# SharkFlow — Polymarket Trading Bot

**Desarrollado por: Carlos David Donoso Cordero (ddchack)**

Bot de trading automatizado para [Polymarket](https://polymarket.com) con motor matemático avanzado, análisis de sentimiento, backtesting histórico, tracking de ballenas (whale tracker), y dashboard web estilo Bloomberg terminal.

> ⚠️ **Disclaimer**: Este software es para fines educativos y de investigación. Las apuestas en prediction markets conllevan riesgo de pérdida total del capital invertido. Úsalo bajo tu propia responsabilidad.

---

## Características

- **Motor matemático**: Kelly Criterion fraccional, Expected Value, KL Divergence, Bayesian Ensemble, Monte Carlo, ELO, Poisson, VPIN, OBI
- **Calibración**: Platt Scaling, Beta Calibration, Isotonic Regression
- **Rush Mode**: Estrategia de 5 minutos BTC con señales RSI + crowd consensus (momentum y mean-reversion)
- **Dump & Hedge**: Detección de dumpeos y cobertura automática
- **Escape Room**: Oportunidades de alto retorno (precio ≤$0.33 → retorno ≥3x) con análisis dual-IA
- **Whale Tracker**: Monitoreo de wallets con historial de ganancias
- **Arbitraje**: Detección YES+NO simultáneos > 100¢
- **LLM Integration**: Claude (Anthropic) + OpenAI para análisis de mercados
- **Dashboard**: Bloomberg terminal dark theme con Chart.js

---

## Arquitectura

```
sharkflow/
├── backend/
│   ├── api_server.py          # FastAPI server — punto de entrada (~5000 líneas)
│   ├── math_engine.py         # Kelly, EV, edge detection, estadísticas
│   ├── bayesian_engine.py     # Motor Bayesiano con ensemble v5.0
│   ├── rush_mode.py           # Estrategia BTC 5-min (momentum + contrarian)
│   ├── market_scanner.py      # Escaneo de mercados via Gamma + CLOB API
│   ├── trading_client.py      # Ejecución CLOB (FOK, limit orders)
│   ├── risk_manager.py        # Gestión de riesgo adaptativa
│   ├── whale_tracker.py       # Tracking de wallets grandes
│   ├── arbitrage_detector.py  # Detector YES+NO simultáneo
│   ├── backtest_engine.py     # Backtesting con mercados resueltos
│   ├── news_sentiment.py      # Análisis de noticias (NewsAPI)
│   ├── llm_engine.py          # Integración Claude + OpenAI
│   ├── calibration_v2.py      # Calibración Platt/Beta/Isotónica
│   ├── category_models.py     # Modelos por categoría (política, deportes, crypto)
│   ├── mean_reversion.py      # Mean reversion engine
│   ├── microstructure.py      # VPIN, OBI, análisis de orderbook
│   ├── dump_hedge.py          # Estrategia Dump & Hedge
│   ├── escape_room.py         # Estrategia alto retorno
│   ├── sports_data.py         # Cliente ESPN + Gamma APIs
│   ├── sports_intel.py        # ELO, lesiones, altitud
│   ├── auto_allocator.py      # Asignación automática de capital
│   ├── advanced_math.py       # Algoritmos matemáticos adicionales
│   ├── ws_client.py           # WebSocket client CLOB
│   ├── extremizer.py          # Extremización de probabilidades
│   ├── swarm_engine.py        # Motor swarm
│   └── dashboard.html         # UI Bloomberg terminal (vanilla JS + Chart.js)
├── config/
│   └── .env.example           # Template de configuración
├── launcher.py                # Punto de entrada — inicia backend
├── start.bat                  # Script de inicio Windows
├── sharkflow_loop.sh          # Loop bash Linux/Mac
├── requirements.txt           # Dependencias Python
└── CLAUDE.md                  # Instrucciones para agente autónomo SharkBot
```

---

## APIs de Polymarket Utilizadas

| API | URL | Auth | Uso |
|-----|-----|------|-----|
| **Gamma** | `gamma-api.polymarket.com` | Ninguna | Mercados, metadata, precios |
| **CLOB** | `clob.polymarket.com` | L1/L2 EIP-712 | Orderbook, ejecución de órdenes |
| **Data** | `data-api.polymarket.com` | Ninguna | Posiciones, historial de trades |

---

## Fórmulas Matemáticas

### Kelly Criterion (fraccional 25%)
```
f* = (b·p - q) / b
donde:
  b = (1/price) - 1   (odds decimales netos)
  p = probabilidad real estimada
  q = 1 - p
```

### Expected Value
```
EV = p_real × (1/price - 1) - (1 - p_real)
```
Solo se apuesta si `EV > 0.05` ($0.05 por cada $1 invertido).

### Confidence Score (0-100)
Compuesto ponderado:
- Edge magnitude: 30%
- EV magnitude: 25%
- Volumen 24h: 15%
- Liquidez: 15%
- Sentimiento de noticias: 10%
- Spread del mercado: 5%

### Rush Mode — Contrarian Signal
```
Señal UP:  RSI < 45 (sobrevendido) + crowd bias ≥ 3¢ + sin tendencia bajista 2 ventanas
Señal DOWN: RSI > 55 (sobrecomprado) + crowd bias ≥ 3¢ + sin tendencia alcista 2 ventanas
Señal Poly-pura: BTC lateral (<0.05%) + crowd ≥ 54¢ en un lado
```

---

## Setup

### 1. Clonar el repositorio

```bash
git clone https://github.com/ddchack/sharkflow.git
cd sharkflow
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar credenciales

```bash
cp config/.env.example .env
```

Editar `.env` con tus datos:

```env
# Polymarket
POLYMARKET_PRIVATE_KEY=0xTU_PRIVATE_KEY_AQUI
POLYMARKET_FUNDER_ADDRESS=0xTU_PROXY_WALLET_AQUI
POLYMARKET_SIGNATURE_TYPE=1   # 1 para cuentas Gmail/Magic.link

# LLM (opcionales — habilitan análisis con IA)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...

# Noticias (opcional — mejora análisis de sentimiento)
NEWSAPI_KEY=tu_newsapi_key

# Servidor
BACKEND_PORT=8888
```

#### Cómo obtener tu Private Key de Polymarket
1. Ve a [reveal.magic.link/polymarket](https://reveal.magic.link/polymarket) (logueado)
2. Autentícate con tu email
3. Copia la private key (empieza con `0x...`)

#### Cómo encontrar tu Proxy Wallet (Funder Address)
- En tu perfil de Polymarket, la dirección mostrada es el **Gnosis Safe proxy** — esa es tu `FUNDER_ADDRESS`
- La EOA (derivada de la private key) es diferente — **no uses la EOA como funder**

### 4. Iniciar el servidor

```bash
# Windows
start.bat

# Linux / Mac
python launcher.py
# o
./sharkflow_loop.sh
```

Dashboard disponible en: `http://localhost:8888`

---

## Endpoints Principales

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/api/status` | Estado del bot y autenticación |
| GET | `/api/markets` | Mercados activos Polymarket |
| GET | `/api/scan` | Escanear oportunidades (Kelly + EV + Bayesian) |
| GET | `/api/balance` | Balance USDC en wallet |
| GET | `/api/risk` | Estado de gestión de riesgo |
| GET | `/api/risk/status` | Pausas y pérdidas consecutivas |
| GET | `/api/whales` | Movimientos de wallets grandes |
| GET | `/api/arbitrage` | Oportunidades de arbitraje |
| GET | `/api/history` | Historial de trades persistente |
| POST | `/api/trade/auto` | Auto-trading con Kelly |
| POST | `/api/auto-allocate` | Asignación automática multi-mercado |
| POST | `/api/rush/start` | Iniciar Rush Mode |
| POST | `/api/rush/stop` | Detener Rush Mode |
| GET | `/api/rush/status` | Estado del bot Rush |
| POST | `/api/escape-room/start` | Iniciar Escape Room bot |
| GET | `/api/escape-room/scan` | Oportunidades ≥3x retorno |
| GET | `/api/sports/markets` | Mercados deportivos |
| GET | `/api/sports/live-scores` | Scores ESPN en vivo |
| WS | `/ws/dashboard` | WebSocket heartbeat + eventos |

---

## Seguridad

- **DRY RUN activado por defecto** — no ejecuta trades reales hasta que lo habilites explícitamente
- Capital máximo configurable (default: $100 USD)
- Stop automático tras 3 pérdidas consecutivas (pausa 24h)
- Las credenciales solo se almacenan en `.env` local — **nunca se suben al repo**
- Firma L1/L2 EIP-712 para autenticación con Polymarket CLOB

---

## Stack Tecnológico

| Componente | Tecnología |
|-----------|-----------|
| Backend | Python 3.11+, FastAPI, uvicorn |
| Frontend | HTML/CSS/JS vanilla, Chart.js 4.4.4 |
| Polymarket | `py-clob-client` |
| ML | `scikit-learn`, `numpy`, `scipy` |
| LLM | `anthropic`, `openai` |
| Async HTTP | `httpx` |
| WebSocket | `websockets`, FastAPI WebSocket |

---

## Créditos

**Desarrollado por Carlos David Donoso Cordero (ddchack)**

- GitHub: [@ddchack](https://github.com/ddchack)
- Proyecto iniciado: 2024

### Librerías y recursos utilizados
- [Polymarket CLOB Client](https://github.com/Polymarket/py-clob-client) — MIT License
- [FastAPI](https://fastapi.tiangolo.com/) — MIT License
- [Chart.js](https://www.chartjs.org/) — MIT License
- [scikit-learn](https://scikit-learn.org/) — BSD License
- Fórmulas de Kelly Criterion basadas en J.L. Kelly Jr. (1956)
- Calibración Platt Scaling: John Platt (1999)

---

## Licencia

MIT License — libre para uso personal y comercial con atribución.

```
Copyright (c) 2024 Carlos David Donoso Cordero (ddchack)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
```
