"""
Deriv Synthetic Indices Trading Bot
- Par: Step Index
- Estrategia: Quiebre + Retesteo + Rechazo (mentor Joel)
- Temporalidades: Semanal → Diario → H4 → Entrada
- Gestión: Sistema de balas 5%
- Grid/Recovery automático
- Alertas Telegram con botones en español
- 100% automático via Deriv WebSocket API
"""
import os
import json
import uuid
import asyncio
import logging
import websockets
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional, Literal

import httpx
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, HTTPException, Request
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, ConfigDict

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("deriv-bot")

# ── Config ────────────────────────────────────────────────────────────────────
MONGO_URL         = os.environ["MONGO_URL"]
DB_NAME           = os.environ.get("DB_NAME", "deriv_bot")
DERIV_TOKEN       = os.environ.get("DERIV_TOKEN", "")
TELEGRAM_TOKEN    = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT     = os.environ.get("TELEGRAM_CHAT_ID", "")
DERIV_WS_URL      = "wss://ws.derivws.com/websockets/v3?app_id=36544"

# ── Pares disponibles ─────────────────────────────────────────────────────────
PAIRS = {
    "step_index":    {"symbol": "stpRNG",   "name": "Step Index",      "pip": 0.1},
    "volatility_75": {"symbol": "R_75",     "name": "Volatility 75",   "pip": 0.01},
    "volatility_25": {"symbol": "R_25",     "name": "Volatility 25",   "pip": 0.01},
    "volatility_10": {"symbol": "R_10",     "name": "Volatility 10",   "pip": 0.01},
}

mongo_client = AsyncIOMotorClient(MONGO_URL)
db = mongo_client[DB_NAME]

app = FastAPI(title="Deriv Synthetic Bot API")
api = APIRouter(prefix="/api")

# ── Helpers ───────────────────────────────────────────────────────────────────
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def new_id() -> str:
    return str(uuid.uuid4())

def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

# ── Modelos ───────────────────────────────────────────────────────────────────
class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = "singleton"
    pair: str = "volatility_75"
    capital_total: float = 1000.0       # Capital total guardado en banco
    bala_pct: float = 5.0               # % del capital por bala (5%)
    objetivo_multiplicador: float = 20.0 # Objetivo: 20x la bala
    max_operaciones_dia: int = 2        # Máximo 2 operaciones por día
    grid_size: float = 1.0              # Distancia entre operaciones grid
    max_grid_operaciones: int = 5       # Máximo operaciones en grid
    tick_interval_seconds: int = 300    # Cada cuánto analiza

class BotState(BaseModel):
    id: str = "singleton"
    running: bool = False
    started_at: Optional[str] = None
    last_tick_at: Optional[str] = None
    bala_actual: float = 0.0
    bala_numero: int = 0
    operaciones_hoy: int = 0
    operaciones_fecha: str = Field(default_factory=today_utc)
    ganancia_total: float = 0.0
    perdidas_total: float = 0.0
    circuit_breaker: bool = False

class Operacion(BaseModel):
    id: str = Field(default_factory=new_id)
    par: str
    direccion: Literal["BUY", "SELL"]
    precio_entrada: float
    lotaje: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: Literal["ABIERTA", "CERRADA"] = "ABIERTA"
    precio_salida: Optional[float] = None
    ganancia: Optional[float] = None
    motivo_entrada: str = ""
    motivo_salida: Optional[str] = None
    abierta_en: str = Field(default_factory=now_iso)
    cerrada_en: Optional[str] = None
    grid_nivel: int = 0                  # 0 = entrada original, 1,2,3 = recovery

class Senal(BaseModel):
    id: str = Field(default_factory=new_id)
    par: str
    temporalidad: str
    accion: Literal["BUY", "SELL", "HOLD"]
    confianza: float
    precio: float
    motivo: str
    indicadores: dict
    creada_en: str = Field(default_factory=now_iso)

# ── Deriv WebSocket ───────────────────────────────────────────────────────────
async def deriv_request_public(request: dict) -> dict:
    """Envía una petición PÚBLICA a la API de Deriv via WebSocket (sin autenticación)."""
    try:
        async with websockets.connect(DERIV_WS_URL, ping_timeout=30) as ws:
            await ws.send(json.dumps(request))
            response = json.loads(await ws.recv())
            return response
    except Exception as exc:
        logger.warning("Error Deriv WS público: %s", exc)
        return {"error": str(exc)}

async def get_candles(symbol: str, granularity: int, count: int = 250) -> List[dict]:
    """
    Obtiene velas históricas de Deriv SIN autenticación.
    granularity: 86400=diario, 14400=H4, 3600=H1, 300=M5, 60=M1
    """
    resp = await deriv_request_public({
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": count,
        "end": "latest",
        "granularity": granularity,
        "style": "candles",
    })

    if "error" in resp or "candles" not in resp:
        logger.warning("Error obteniendo velas %s: %s", symbol, resp.get("error", "sin datos"))
        return []

    candles = []
    for c in resp["candles"]:
        candles.append({
            "time":   c["epoch"],
            "open":   float(c["open"]),
            "high":   float(c["high"]),
            "low":    float(c["low"]),
            "close":  float(c["close"]),
        })
    return candles

async def get_current_price(symbol: str) -> float:
    """Obtiene el precio actual de un par sin autenticación."""
    resp = await deriv_request_public({"ticks": symbol})
    if "tick" in resp:
        return float(resp["tick"]["quote"])
    return 0.0

async def place_order(
    symbol: str,
    direction: str,
    amount: float,
    duration: int = 5,
    duration_unit: str = "m",
) -> dict:
    """
    Coloca una orden en Deriv.
    Para Step Index usa contratos de tipo 'CALL' (compra) o 'PUT' (venta).
    """
    contract_type = "CALL" if direction == "BUY" else "PUT"
    resp = await deriv_request({
        "buy": 1,
        "price": amount,
        "parameters": {
            "amount": amount,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "duration": duration,
            "duration_unit": duration_unit,
            "symbol": symbol,
        },
    })
    return resp

# ── Indicadores técnicos ──────────────────────────────────────────────────────
def calcular_indicadores(candles: List[dict]) -> dict:
    """
    Calcula EMA 50/100/200, RSI, estructura de mercado, soportes y resistencias.
    """
    if len(candles) < 50:
        return {}

    df = pd.DataFrame(candles)
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)

    # EMAs
    ema_50  = close.ewm(span=50,  adjust=False).mean()
    ema_100 = close.ewm(span=100, adjust=False).mean() if len(close) >= 100 else ema_50
    ema_200 = close.ewm(span=200, adjust=False).mean() if len(close) >= 200 else ema_100

    # RSI 14
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))

    # ATR 14
    tr  = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    # Estructura del mercado
    def estructura():
        if len(close) < 10:
            return "desconocida"
        highs = high.rolling(5).max()
        lows  = low.rolling(5).min()
        ultimo_alto  = float(highs.iloc[-1])
        penultimo_alto = float(highs.iloc[-6]) if len(highs) > 6 else ultimo_alto
        ultimo_bajo  = float(lows.iloc[-1])
        penultimo_bajo = float(lows.iloc[-6]) if len(lows) > 6 else ultimo_bajo
        if ultimo_alto > penultimo_alto and ultimo_bajo > penultimo_bajo:
            return "alcista"
        elif ultimo_alto < penultimo_alto and ultimo_bajo < penultimo_bajo:
            return "bajista"
        return "lateral"

    # Soporte y resistencia recientes
    def soporte_resistencia():
        ventana = 20
        if len(close) < ventana:
            return None, None
        reciente_high = float(high.iloc[-ventana:].max())
        reciente_low  = float(low.iloc[-ventana:].min())
        return reciente_low, reciente_high

    soporte, resistencia = soporte_resistencia()

    def f(x):
        try:
            v = float(x)
            return None if (np.isnan(v) or np.isinf(v)) else round(v, 4)
        except Exception:
            return None

    return {
        "precio":      f(close.iloc[-1]),
        "ema_50":      f(ema_50.iloc[-1]),
        "ema_100":     f(ema_100.iloc[-1]),
        "ema_200":     f(ema_200.iloc[-1]),
        "rsi":         f(rsi.iloc[-1]),
        "atr":         f(atr.iloc[-1]),
        "estructura":  estructura(),
        "soporte":     f(soporte),
        "resistencia": f(resistencia),
        "tendencia":   "alcista" if float(ema_50.iloc[-1]) > float(ema_200.iloc[-1]) else "bajista",
    }

# ── Detección de zonas con múltiples toques ──────────────────────────────────
def detectar_zonas(candles: List[dict], tolerancia_pct: float = 0.002) -> List[dict]:
    """
    Detecta zonas de soporte/resistencia con múltiples toques.
    El mentor dice: zona con varios toques = zona válida.
    tolerancia_pct: 0.2% de tolerancia para agrupar toques.
    """
    if len(candles) < 10:
        return []

    zonas = []
    highs = [c["high"] for c in candles]
    lows  = [c["low"]  for c in candles]

    # Detectar niveles con múltiples toques
    todos_niveles = highs + lows

    for nivel in todos_niveles:
        tolerancia = nivel * tolerancia_pct
        toques = sum(
            1 for c in candles
            if abs(c["high"] - nivel) < tolerancia or abs(c["low"] - nivel) < tolerancia
        )
        if toques >= 2:  # Mínimo 2 toques para ser zona válida
            # Verificar si ya existe zona cercana
            existe = any(abs(z["nivel"] - nivel) < tolerancia * 3 for z in zonas)
            if not existe:
                zonas.append({
                    "nivel":  round(nivel, 4),
                    "toques": toques,
                    "tipo":   "resistencia" if nivel > candles[-1]["close"] else "soporte",
                })

    # Ordenar por número de toques (más toques = más fuerte)
    return sorted(zonas, key=lambda z: z["toques"], reverse=True)[:10]

def detectar_quiebre_contundente(candles: List[dict], zona_nivel: float) -> dict:
    """
    Detecta si hubo un quiebre contundente de una zona.
    Regla del mentor: quiebre con CUERPO de vela, no mecha.
    """
    if len(candles) < 3:
        return {"quiebre": False, "direccion": None}

    tolerancia = zona_nivel * 0.002

    # Revisar las últimas 5 velas
    for i in range(-5, 0):
        vela = candles[i]
        cuerpo_alto = max(vela["open"], vela["close"])
        cuerpo_bajo = min(vela["open"], vela["close"])
        tamaño_cuerpo = abs(vela["close"] - vela["open"])
        tamaño_total  = vela["high"] - vela["low"]

        # El cuerpo debe ser al menos 50% de la vela total
        if tamaño_total == 0:
            continue
        ratio_cuerpo = tamaño_cuerpo / tamaño_total

        if ratio_cuerpo < 0.5:
            continue  # Es mecha, no cuerpo — no cuenta como quiebre

        # Quiebre bajista: cuerpo cierra por debajo de la zona
        if cuerpo_bajo < zona_nivel - tolerancia and vela["close"] < zona_nivel:
            return {
                "quiebre":   True,
                "direccion": "SELL",
                "vela_idx":  i,
                "precio_quiebre": vela["close"],
            }

        # Quiebre alcista: cuerpo cierra por encima de la zona
        if cuerpo_alto > zona_nivel + tolerancia and vela["close"] > zona_nivel:
            return {
                "quiebre":   True,
                "direccion": "BUY",
                "vela_idx":  i,
                "precio_quiebre": vela["close"],
            }

    return {"quiebre": False, "direccion": None}

def detectar_retesteo(candles: List[dict], zona_nivel: float, direccion: str) -> bool:
    """
    Detecta si el precio está retestando la zona quebrada.
    Regla del mentor: entras DURANTE el retesteo, no después.
    """
    if len(candles) < 2:
        return False

    tolerancia  = zona_nivel * 0.003
    precio_actual = candles[-1]["close"]
    precio_anterior = candles[-2]["close"]

    # El precio debe estar cerca de la zona
    dist_zona = abs(precio_actual - zona_nivel)

    if dist_zona > tolerancia * 3:
        return False

    # Para SELL: precio subió hasta la zona (retesteo de resistencia creada)
    if direccion == "SELL":
        return precio_actual >= zona_nivel - tolerancia and precio_anterior < precio_actual

    # Para BUY: precio bajó hasta la zona (retesteo de soporte creado)
    if direccion == "BUY":
        return precio_actual <= zona_nivel + tolerancia and precio_anterior > precio_actual

    return False

# ── Detección de patrón quiebre + retesteo (versión mejorada) ─────────────────
def detectar_patron_quiebre_retesteo(
    candles_h4: List[dict],
    candles_diario: List[dict],
) -> dict:
    """
    Estrategia exacta del mentor Joel:
    1. Zona con múltiples toques (soporte/resistencia válida)
    2. Quiebre contundente con CUERPO de vela (no mecha)
    3. Retesteo de la zona quebrada
    4. Entrada durante el retesteo
    5. Solo H4 y Diario
    """
    if len(candles_h4) < 30 or len(candles_diario) < 10:
        return {"accion": "HOLD", "confianza": 0.0, "motivo": "Datos insuficientes"}

    ind_h4     = calcular_indicadores(candles_h4)
    ind_diario = calcular_indicadores(candles_diario)

    if not ind_h4 or not ind_diario:
        return {"accion": "HOLD", "confianza": 0.0, "motivo": "Indicadores insuficientes"}

    precio_actual  = candles_h4[-1]["close"]
    estructura_d   = ind_diario.get("estructura", "lateral")
    tendencia_h4   = ind_h4.get("tendencia", "bajista")
    tendencia_d    = ind_diario.get("tendencia", "bajista")
    rsi_h4         = ind_h4.get("rsi", 50) or 50

    score   = 0
    motivos = []
    accion  = "HOLD"

    # ── Paso 1: Detectar zonas válidas en H4 ─────────────────────────────────
    zonas_h4 = detectar_zonas(candles_h4)
    zonas_d  = detectar_zonas(candles_diario)

    if not zonas_h4:
        return {"accion": "HOLD", "confianza": 0.0, "motivo": "Sin zonas válidas en H4"}

    # ── Paso 2: Buscar quiebre + retesteo en las zonas más fuertes ───────────
    señal_encontrada = False

    for zona in zonas_h4[:5]:  # Revisar las 5 zonas más fuertes
        nivel  = zona["nivel"]
        toques = zona["toques"]

        # Quiebre contundente con cuerpo de vela
        quiebre = detectar_quiebre_contundente(candles_h4, nivel)

        if not quiebre["quiebre"]:
            continue

        direccion = quiebre["direccion"]

        # Retesteo de la zona
        en_retesteo = detectar_retesteo(candles_h4, nivel, direccion)

        if not en_retesteo:
            continue

        # ── Confirmaciones adicionales ────────────────────────────────────
        score = 0

        # Zona fuerte (más toques = más puntos)
        if toques >= 3:
            score += 3
            motivos.append(f"Zona fuerte con {toques} toques en H4")
        elif toques >= 2:
            score += 2
            motivos.append(f"Zona con {toques} toques en H4")

        # Tendencia alineada
        if tendencia_h4 == tendencia_d:
            score += 2
            motivos.append(f"Tendencia alineada H4+D: {tendencia_h4}")

        # Estructura diaria confirma
        if (direccion == "BUY" and estructura_d == "alcista") or            (direccion == "SELL" and estructura_d == "bajista"):
            score += 2
            motivos.append(f"Estructura diaria confirma: {estructura_d}")

        # RSI confirma
        if direccion == "BUY" and rsi_h4 < 45:
            score += 1
            motivos.append(f"RSI bajo ({rsi_h4:.0f}) — confirma compra")
        elif direccion == "SELL" and rsi_h4 > 55:
            score += 1
            motivos.append(f"RSI alto ({rsi_h4:.0f}) — confirma venta")

        # Zona también existe en diario (más fuerte)
        zona_en_diario = any(
            abs(z["nivel"] - nivel) < nivel * 0.005
            for z in zonas_d
        )
        if zona_en_diario:
            score += 2
            motivos.append("Zona confirmada también en Diario")

        if score >= 5:
            accion = direccion
            señal_encontrada = True
            motivos.insert(0, f"✅ Quiebre+Retesteo @ {nivel:.4f}")
            break

    if not señal_encontrada:
        return {
            "accion":    "HOLD",
            "confianza": 0.0,
            "motivo":    "Sin patrón quiebre+retesteo válido",
            "score":     0,
        }

    confianza = round(min(1.0, score / 10.0), 2)

    return {
        "accion":    accion,
        "confianza": confianza,
        "motivo":    " | ".join(motivos),
        "score":     score,
    }

# ── Sistema de balas ──────────────────────────────────────────────────────────
async def calcular_bala(settings: Settings) -> float:
    """Calcula el monto de la bala actual (5% del capital)."""
    return round(settings.capital_total * (settings.bala_pct / 100), 2)

async def check_objetivo_bala(settings: Settings, state: BotState) -> bool:
    """Verifica si se alcanzó el objetivo de la bala actual."""
    bala    = await calcular_bala(settings)
    objetivo = bala * settings.objetivo_multiplicador
    return state.ganancia_total >= objetivo

# ── Settings & State ──────────────────────────────────────────────────────────
async def get_settings() -> Settings:
    doc = await db.settings.find_one({"id": "singleton"}, {"_id": 0})
    if not doc:
        s = Settings()
        await db.settings.insert_one(s.model_dump())
        return s
    return Settings(**doc)

async def save_settings(s: Settings) -> None:
    await db.settings.update_one({"id": "singleton"}, {"$set": s.model_dump()}, upsert=True)

async def get_state() -> BotState:
    doc = await db.bot_state.find_one({"id": "singleton"}, {"_id": 0})
    if not doc:
        s = BotState()
        await db.bot_state.insert_one(s.model_dump())
        return s
    return BotState(**doc)

async def save_state(s: BotState) -> None:
    await db.bot_state.update_one({"id": "singleton"}, {"$set": s.model_dump()}, upsert=True)

# ── Telegram con botones ──────────────────────────────────────────────────────
MENU_PRINCIPAL = {
    "inline_keyboard": [
        [
            {"text": "▶️ Arrancar Bot",  "callback_data": "start_bot"},
            {"text": "🛑 Parar Bot",     "callback_data": "stop_bot"},
        ],
        [
            {"text": "📊 Portfolio",     "callback_data": "portfolio"},
            {"text": "📈 Señales",       "callback_data": "senales"},
        ],
        [
            {"text": "💰 Nueva Bala",    "callback_data": "nueva_bala"},
            {"text": "⚙️ Estado",        "callback_data": "estado"},
        ],
        [
            {"text": "⚡ PÁNICO — Cerrar Todo", "callback_data": "panico"},
        ],
    ]
}

async def enviar_telegram(mensaje: str, reply_markup: dict = None) -> bool:
    token   = TELEGRAM_TOKEN
    chat_id = TELEGRAM_CHAT
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        payload = {
            "chat_id":    chat_id,
            "text":       mensaje,
            "parse_mode": "Markdown",
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, json=payload)
            return r.status_code == 200
    except Exception as exc:
        logger.warning("Error Telegram: %s", exc)
        return False

async def responder_callback(callback_id: str, texto: str = "") -> None:
    token = TELEGRAM_TOKEN
    if not token:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/answerCallbackQuery",
                json={"callback_query_id": callback_id, "text": texto},
            )
    except Exception:
        pass

async def procesar_telegram(update: dict) -> None:
    """Procesa mensajes y botones de Telegram."""
    settings = await get_settings()
    state    = await get_state()

    # Comandos de texto
    mensaje = update.get("message", {})
    texto   = mensaje.get("text", "")

    if texto in ("/start", "/menu"):
        bala    = await calcular_bala(settings)
        status  = "🟢 Corriendo" if state.running else "🔴 Parado"
        par     = PAIRS.get(settings.pair, {}).get("name", settings.pair)
        await enviar_telegram(
            f"👋 *Bot de Índices Sintéticos*\n\n"
            f"Estado: {status}\n"
            f"Par: {par}\n"
            f"Capital total: ${settings.capital_total:,.2f}\n"
            f"Bala actual: ${bala:,.2f} ({settings.bala_pct}%)\n"
            f"Objetivo bala: ${bala * settings.objetivo_multiplicador:,.2f}\n"
            f"Ganancia total: ${state.ganancia_total:+.2f}\n\n"
            f"¿Qué quieres hacer?",
            reply_markup=MENU_PRINCIPAL,
        )
        return

    # Botones
    callback = update.get("callback_query", {})
    cb_id    = callback.get("id", "")
    cb_data  = callback.get("data", "")

    if not cb_data:
        return

    if cb_data == "start_bot":
        if state.circuit_breaker:
            await responder_callback(cb_id, "⚠️ Circuit breaker activo")
            await enviar_telegram("⚠️ *No se puede arrancar*\nCircuit breaker activo.", reply_markup=MENU_PRINCIPAL)
            return
        state.running    = True
        state.started_at = now_iso()
        await save_state(state)
        await responder_callback(cb_id, "✅ Bot arrancado")
        bala = await calcular_bala(settings)
        par  = PAIRS.get(settings.pair, {}).get("name", settings.pair)
        await enviar_telegram(
            f"🟢 *Bot arrancado*\n"
            f"Par: {par}\n"
            f"Bala: ${bala:.2f}\n"
            f"Analizando Diario + H4 cada {settings.tick_interval_seconds//60} minutos.",
            reply_markup=MENU_PRINCIPAL,
        )

    elif cb_data == "stop_bot":
        state.running = False
        await save_state(state)
        await responder_callback(cb_id, "🛑 Bot parado")
        await enviar_telegram("🔴 *Bot parado*\nLas posiciones abiertas siguen activas.", reply_markup=MENU_PRINCIPAL)

    elif cb_data == "panico":
        state.running = False
        await save_state(state)
        # Cerrar todas las operaciones abiertas
        abiertas = await db.operaciones.find({"status": "ABIERTA"}, {"_id": 0}).to_list(100)
        for op_doc in abiertas:
            await db.operaciones.update_one(
                {"id": op_doc["id"]},
                {"$set": {"status": "CERRADA", "motivo_salida": "⚡ Pánico manual", "cerrada_en": now_iso()}}
            )
        await responder_callback(cb_id, "⚡ Pánico ejecutado")
        await enviar_telegram(
            f"⚡ *PÁNICO EJECUTADO*\n"
            f"Bot parado. {len(abiertas)} operación(es) cerrada(s).",
            reply_markup=MENU_PRINCIPAL,
        )

    elif cb_data == "portfolio":
        ops_cerradas = await db.operaciones.find({"status": "CERRADA"}, {"_id": 0}).to_list(1000)
        ops_abiertas = await db.operaciones.find({"status": "ABIERTA"}, {"_id": 0}).to_list(100)
        ganadas  = len([o for o in ops_cerradas if (o.get("ganancia") or 0) > 0])
        perdidas = len([o for o in ops_cerradas if (o.get("ganancia") or 0) < 0])
        total    = len(ops_cerradas)
        wr       = round(ganadas / total * 100, 1) if total else 0
        bala     = await calcular_bala(settings)
        status   = "🟢 Corriendo" if state.running else "🔴 Parado"
        await responder_callback(cb_id, "📊 Cargando...")
        await enviar_telegram(
            f"📊 *Portfolio Sintéticos*\n\n"
            f"Estado: {status}\n"
            f"Capital total: ${settings.capital_total:,.2f}\n"
            f"Bala #{state.bala_numero}: ${bala:.2f}\n"
            f"Ganancia total: ${state.ganancia_total:+.2f}\n"
            f"Operaciones hoy: {state.operaciones_hoy}/{settings.max_operaciones_dia}\n"
            f"Total operaciones: {total}\n"
            f"Ganadas: {ganadas} | Perdidas: {perdidas}\n"
            f"Win rate: {wr}%",
            reply_markup=MENU_PRINCIPAL,
        )

    elif cb_data == "senales":
        cur   = db.senales.find({}, {"_id": 0}).sort("creada_en", -1).limit(5)
        senas = await cur.to_list(5)
        await responder_callback(cb_id, "📈 Últimas señales")
        if not senas:
            await enviar_telegram("📈 *No hay señales todavía*", reply_markup=MENU_PRINCIPAL)
            return
        msg = "📈 *Últimas señales*\n\n"
        for s in senas:
            e = "🟢" if s["accion"] == "BUY" else "🔴" if s["accion"] == "SELL" else "⚪"
            msg += f"{e} *{s['par']}* {s['accion']} | {s['confianza']*100:.0f}%\n_{s['motivo'][:80]}_\n\n"
        await enviar_telegram(msg, reply_markup=MENU_PRINCIPAL)

    elif cb_data == "nueva_bala":
        state.bala_numero += 1
        state.ganancia_total = 0.0
        await save_state(state)
        bala = await calcular_bala(settings)
        await responder_callback(cb_id, "💰 Nueva bala iniciada")
        await enviar_telegram(
            f"💰 *Nueva bala iniciada*\n"
            f"Bala #{state.bala_numero}\n"
            f"Monto: ${bala:.2f}\n"
            f"Objetivo: ${bala * settings.objetivo_multiplicador:.2f}\n"
            f"¡A facturar! 🎯",
            reply_markup=MENU_PRINCIPAL,
        )

    elif cb_data == "estado":
        price = await get_current_price(PAIRS[settings.pair]["symbol"])
        par   = PAIRS.get(settings.pair, {}).get("name", settings.pair)
        await responder_callback(cb_id, "⚙️ Estado actual")
        await enviar_telegram(
            f"⚙️ *Estado actual*\n\n"
            f"Par: {par}\n"
            f"Precio actual: {price:.4f}\n"
            f"Bot: {'🟢 Corriendo' if state.running else '🔴 Parado'}\n"
            f"Operaciones hoy: {state.operaciones_hoy}/{settings.max_operaciones_dia}\n"
            f"Último análisis: {state.last_tick_at or 'Nunca'}",
            reply_markup=MENU_PRINCIPAL,
        )

# ── Lógica principal del bot ──────────────────────────────────────────────────
async def ejecutar_tick() -> dict:
    """
    Ejecuta un ciclo de análisis:
    1. Obtiene velas Diario y H4
    2. Analiza estructura y zonas
    3. Si hay señal → abre operación
    4. Revisa operaciones abiertas
    """
    settings = await get_settings()
    state    = await get_state()
    state.last_tick_at = now_iso()

    # Reset operaciones diarias
    if state.operaciones_fecha != today_utc():
        state.operaciones_hoy    = 0
        state.operaciones_fecha  = today_utc()

    await save_state(state)

    if not state.running:
        return {"mensaje": "Bot parado"}

    if state.circuit_breaker:
        return {"mensaje": "Circuit breaker activo"}

    par_config = PAIRS.get(settings.pair, PAIRS["step_index"])
    simbolo    = par_config["symbol"]

    # Obtener velas
    logger.info("📊 Obteniendo velas para %s...", simbolo)
    velas_diario = await get_candles(simbolo, 86400, 100)   # Diario
    velas_h4     = await get_candles(simbolo, 14400, 200)   # H4

    logger.info("📊 Velas obtenidas — Diario: %d | H4: %d", len(velas_diario), len(velas_h4))

    if not velas_diario or not velas_h4:
        logger.warning("No se pudieron obtener velas para %s", simbolo)
        return {"mensaje": "Error obteniendo datos de mercado"}

    # Analizar patrón
    senal = detectar_patron_quiebre_retesteo(velas_h4, velas_diario)
    precio = velas_h4[-1]["close"] if velas_h4 else 0
    logger.info("📈 Señal: %s | Confianza: %.0f%% | %s", senal["accion"], senal["confianza"]*100, senal["motivo"][:80])

    # Evitar señales duplicadas — solo guardar si es diferente a la última
    ultima = await db.senales.find_one(
        {"par": settings.pair},
        {"_id": 0},
        sort=[("creada_en", -1)]
    )
    es_duplicada = (
        ultima and
        ultima.get("accion") == senal["accion"] and
        ultima.get("motivo") == senal["motivo"]
    )

    if not es_duplicada:
        s = Senal(
            par=settings.pair,
            temporalidad="H4+D",
            accion=senal["accion"],
            confianza=senal["confianza"],
            precio=precio,
            motivo=senal["motivo"],
            indicadores=calcular_indicadores(velas_h4),
        )
        await db.senales.insert_one(s.model_dump())
    else:
        s = Senal(
            par=settings.pair,
            temporalidad="H4+D",
            accion=senal["accion"],
            confianza=senal["confianza"],
            precio=precio,
            motivo=senal["motivo"],
            indicadores=calcular_indicadores(velas_h4),
        )

    resultado = {"senal": senal, "precio": precio}

    # Abrir operación si hay señal
    if (
        senal["accion"] in ("BUY", "SELL")
        and senal["confianza"] >= 0.65
        and state.operaciones_hoy < settings.max_operaciones_dia
    ):
        bala   = await calcular_bala(settings)
        monto  = round(bala * 0.1, 2)  # 10% de la bala por operación

        # Colocar orden en Deriv
        orden = await place_order(
            symbol=simbolo,
            direction=senal["accion"],
            amount=monto,
            duration=4,
            duration_unit="h",
        )

        if "error" not in orden:
            op = Operacion(
                par=settings.pair,
                direccion=senal["accion"],
                precio_entrada=precio,
                lotaje=monto,
                motivo_entrada=senal["motivo"],
            )
            await db.operaciones.insert_one(op.model_dump())
            state.operaciones_hoy += 1
            await save_state(state)

            emoji = "🟢" if senal["accion"] == "BUY" else "🔴"
            await enviar_telegram(
                f"{emoji} *Nueva operación abierta*\n"
                f"Par: {par_config['name']}\n"
                f"Dirección: {senal['accion']}\n"
                f"Precio: {precio:.4f}\n"
                f"Monto: ${monto:.2f}\n"
                f"Confianza: {senal['confianza']*100:.0f}%\n"
                f"Motivo: {senal['motivo'][:100]}",
                reply_markup=MENU_PRINCIPAL,
            )
            resultado["operacion"] = op.model_dump()
        else:
            logger.warning("Error colocando orden: %s", orden.get("error"))

    return resultado

# ── Scheduler ─────────────────────────────────────────────────────────────────
_scheduler_task = None

async def scheduler_loop():
    logger.info("⏰ Scheduler Deriv iniciado")
    await asyncio.sleep(15)
    elapsed = 0
    while True:
        try:
            state    = await get_state()
            settings = await get_settings()
            interval = max(60, settings.tick_interval_seconds)

            if state.running and elapsed >= interval:
                logger.info("🔄 Analizando mercado sintético...")
                await ejecutar_tick()
                elapsed = 0
            elif not state.running:
                elapsed = 0

            await asyncio.sleep(10)
            elapsed += 10

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Error scheduler: %s", exc)
            await asyncio.sleep(15)

# ── Routes ────────────────────────────────────────────────────────────────────
@api.get("/")
async def root():
    return {"name": "Deriv Synthetic Bot", "status": "ok", "version": "1.0"}

@api.get("/market/precio")
async def precio_actual(par: str = "step_index"):
    simbolo = PAIRS.get(par, PAIRS["step_index"])["symbol"]
    precio  = await get_current_price(simbolo)
    return {"par": par, "precio": precio}

@api.get("/market/analisis")
async def analisis(par: str = "step_index"):
    simbolo      = PAIRS.get(par, PAIRS["step_index"])["symbol"]
    velas_diario = await get_candles(simbolo, 86400, 100)
    velas_h4     = await get_candles(simbolo, 14400, 200)
    ind_h4       = calcular_indicadores(velas_h4)
    ind_diario   = calcular_indicadores(velas_diario)
    senal        = detectar_patron_quiebre_retesteo(velas_h4, velas_diario)
    return {
        "par":        par,
        "indicadores_h4":     ind_h4,
        "indicadores_diario": ind_diario,
        "senal":      senal,
    }

@api.get("/operaciones")
async def listar_operaciones(status: Optional[str] = None):
    q = {}
    if status:
        q["status"] = status.upper()
    cur = db.operaciones.find(q, {"_id": 0}).sort("abierta_en", -1).limit(100)
    return {"operaciones": await cur.to_list(100)}

@api.get("/senales")
async def listar_senales(limit: int = 20):
    cur = db.senales.find({}, {"_id": 0}).sort("creada_en", -1).limit(limit)
    return {"senales": await cur.to_list(limit)}

@api.get("/portfolio")
async def portfolio():
    settings     = await get_settings()
    state        = await get_state()
    ops_cerradas = await db.operaciones.find({"status": "CERRADA"}, {"_id": 0}).to_list(1000)
    ops_abiertas = await db.operaciones.find({"status": "ABIERTA"}, {"_id": 0}).to_list(100)
    ganadas      = len([o for o in ops_cerradas if (o.get("ganancia") or 0) > 0])
    total        = len(ops_cerradas)
    bala         = await calcular_bala(settings)
    return {
        "capital_total":    settings.capital_total,
        "bala_actual":      bala,
        "bala_numero":      state.bala_numero,
        "objetivo_bala":    round(bala * settings.objetivo_multiplicador, 2),
        "ganancia_total":   state.ganancia_total,
        "operaciones_hoy":  state.operaciones_hoy,
        "total_operaciones": total,
        "ganadas":          ganadas,
        "win_rate":         round(ganadas / total * 100, 1) if total else 0,
        "running":          state.running,
    }

@api.get("/bot/status")
async def bot_status():
    state    = await get_state()
    settings = await get_settings()
    return {"state": state.model_dump(), "settings": settings.model_dump()}

@api.post("/bot/start")
async def bot_start():
    state = await get_state()
    state.running    = True
    state.started_at = now_iso()
    await save_state(state)
    await enviar_telegram("🟢 *Bot de Sintéticos iniciado*", reply_markup=MENU_PRINCIPAL)
    return {"state": state.model_dump()}

@api.post("/bot/stop")
async def bot_stop():
    state         = await get_state()
    state.running = False
    await save_state(state)
    await enviar_telegram("🔴 *Bot de Sintéticos parado*", reply_markup=MENU_PRINCIPAL)
    return {"state": state.model_dump()}

@api.post("/bot/tick")
async def bot_tick():
    resultado = await ejecutar_tick()
    return resultado

@api.post("/bot/panico")
async def bot_panico():
    state         = await get_state()
    state.running = False
    await save_state(state)
    abiertas = await db.operaciones.find({"status": "ABIERTA"}, {"_id": 0}).to_list(100)
    for op in abiertas:
        await db.operaciones.update_one(
            {"id": op["id"]},
            {"$set": {"status": "CERRADA", "motivo_salida": "Pánico", "cerrada_en": now_iso()}}
        )
    await enviar_telegram(f"⚡ *PÁNICO* — {len(abiertas)} operación(es) cerrada(s)", reply_markup=MENU_PRINCIPAL)
    return {"cerradas": len(abiertas)}

@api.post("/telegram/test")
async def telegram_test():
    ok = await enviar_telegram(
        "✅ *Bot de Sintéticos conectado*\n"
        "Escribe /start para ver el menú.",
        reply_markup=MENU_PRINCIPAL,
    )
    return {"sent": ok}

@api.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    asyncio.create_task(procesar_telegram(data))
    return {"ok": True}

@api.get("/settings")
async def get_settings_route():
    return (await get_settings()).model_dump()

@api.put("/settings")
async def update_settings(data: dict):
    cur  = await get_settings()
    vals = cur.model_dump()
    vals.update(data)
    new = Settings(**vals)
    await save_settings(new)
    return new.model_dump()

# ── App setup ─────────────────────────────────────────────────────────────────
app.include_router(api)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    global _scheduler_task
    _scheduler_task = asyncio.create_task(scheduler_loop())
    logger.info("🚀 Deriv Bot arrancado")

@app.on_event("shutdown")
async def shutdown():
    global _scheduler_task
    if _scheduler_task:
        _scheduler_task.cancel()
    mongo_client.close()
