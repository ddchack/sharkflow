# Polymarket Bot v2.0 — Análisis Completo de Mejoras

**Autor:** Carlos David Donoso Cordero (ddchack)  
**Fecha:** Marzo 2026  
**Basado en:** Investigación de 50+ fuentes incluyendo documentación oficial, papers académicos, bots open-source exitosos, y análisis de traders que han generado $40M+ en arbitraje.

---

## RESUMEN EJECUTIVO

Se identificaron **28 mejoras** organizadas en 7 categorías. Las más impactantes para un capital de $100 USD son:

1. **Modelo Bayesiano** de probabilidad (reemplaza estimación fija)
2. **WebSocket en tiempo real** (reemplaza polling REST)
3. **Detección de arbitraje** intra-mercado (YES+NO < $1.00)
4. **Whale tracking** via Data API (copiar wallets exitosas)
5. **LLM-powered analysis** usando Claude API para evaluar mercados
6. **Circuit breakers** y gestión de riesgo avanzada
7. **Dashboard mejorado** con gráficos en tiempo real

---

## 1. MODELO DE PROBABILIDAD (Crítico)

### Problema actual
El bot v1 estima la probabilidad real como `market_price + sentiment_adjustment`. Esto es circular — usa el precio del mercado para predecir el precio del mercado.

### Mejoras

**1.1 — Bayesian Probability Updater**
- Prior: precio del mercado como distribución Beta(α, β)
- Likelihood: señales externas (noticias, polls, datos)
- Posterior: probabilidad actualizada vía Bayes' theorem
- Implementación: `P(θ|data) ∝ P(data|θ) × P(θ)`
- Se actualiza iterativamente con cada nueva señal

**1.2 — Ensemble de Señales**  
Combinar múltiples fuentes con pesos calibrados:
- Sentimiento de noticias (TextBlob + keyword): peso 20%
- Movimiento de precio 24h (momentum): peso 15%
- Volumen relativo (surge detection): peso 15%
- Posiciones de whales (Data API): peso 25%
- Spread/liquidez (market efficiency): peso 10%
- Consenso cross-platform (vs Kalshi/Metaculus): peso 15%

**1.3 — Calibración Histórica**
- Trackear predicciones vs resultados reales
- Ajustar pesos del ensemble según accuracy histórica
- Usar Brier Score como métrica: `BS = (forecast - outcome)²`

---

## 2. DATOS EN TIEMPO REAL (Crítico)

### Problema actual  
El bot usa REST polling que pierde oportunidades que duran 2-30 segundos.

### Mejoras

**2.1 — WebSocket CLOB**
```
URL: wss://ws-subscriptions-clob.polymarket.com/ws/market
Eventos: book, price_change, last_trade_price, tick_size_change
```
- Suscribirse a mercados monitoreados
- Mantener orderbook local actualizado
- Detectar cambios de precio instantáneamente
- Ping/pong cada 10 segundos para mantener conexión

**2.2 — WebSocket User Channel**  
Para tracking de órdenes propias en tiempo real.

**2.3 — RTDS (Real-Time Data Socket)**
```
URL: wss://ws-live-data.polymarket.com
```
- Precios de crypto en tiempo real
- Feed de actividad y comentarios

---

## 3. ESTRATEGIAS ADICIONALES (Alto Impacto)

### 3.1 — Arbitraje Intra-Mercado
Cuando YES + NO < $1.00, comprar ambos lados garantiza profit.
- Polymarket cobra 2% fee en outcomes ganadores
- Spread mínimo rentable: >2.5-3% después de fees
- Escanear todos los mercados activos cada ciclo
- Ventana típica de oportunidad: 2.7 segundos (2026)

### 3.2 — Arbitraje Combinatorial
- Mercados mutuamente excluyentes que suman >100% o <100%
- Ejemplo: "¿Cuántos recortes del Fed?" → opciones 0,1,2...8+ 
- Si los precios de todas las opciones suman < $1.00 → profit garantizado
- Requiere: agrupar mercados por evento, calcular sumas

### 3.3 — Copy Trading (Whale Following)
- Data API endpoint: `GET /activity?user={wallet_address}`
- Identificar wallets con >75% win rate y >$10K volume
- Monitorear posiciones nuevas cada 4 segundos
- Copiar con ratio proporcional (ej: 0.1x del tamaño del whale)
- Filtrar wash trades (volume alto pero PnL bajo)

### 3.4 — Market Making (Avanzado)
- Colocar órdenes limit en ambos lados (bid/ask)
- Capturar spread como profit (ej: comprar YES@0.58, vender@0.62)
- Requiere: gestión de inventario, ajuste dinámico de spreads
- Riesgo: acumulación de inventario en un lado
- Polymarket ofrece rewards/rebates a market makers

### 3.5 — Momentum/Mean Reversion
- Detectar surges de volumen vs promedio 7 días
- Comprar en momentum positivo confirmado
- Vender cuando el precio revierte a la media
- Usar Bollinger Bands adaptados a prediction markets

---

## 4. ANÁLISIS LLM-POWERED (Diferenciador)

### 4.1 — Claude API para Evaluación
- Enviar pregunta del mercado + noticias recientes a Claude
- Solicitar estimación de probabilidad con razonamiento
- Usar como una señal más en el ensemble
- Costo: ~$0.01-0.05 por evaluación con Sonnet

### 4.2 — Análisis de Noticias Mejorado
- Usar Claude para extraer entidades y sentimiento contextual
- Más preciso que TextBlob para noticias financieras/políticas
- Detectar matices que keyword matching pierde
- Evaluar credibilidad de la fuente

---

## 5. GESTIÓN DE RIESGO (Esencial)

### 5.1 — Circuit Breakers
- Pausar trading si pérdida diaria > 10% del capital
- Cancelar todas las órdenes si drawdown > 15%
- Reducir tamaño de apuesta después de 3 pérdidas consecutivas
- Kill switch manual desde el dashboard

### 5.2 — Position Sizing Avanzado
- Anti-Martingala: aumentar después de wins, reducir después de losses
- Risk parity: igualar riesgo por operación, no capital
- Correlation tracking: evitar sobre-exposición a un tema

### 5.3 — Validación Pre-Trade
- Verificar liquidez mínima antes de ejecutar
- Estimar slippage con orderbook depth
- Verificar que el mercado no cierra en <24h (evitar illiquid end-of-life)
- Confirmar balance USDC suficiente

### 5.4 — Portfolio Tracking
- P&L en tiempo real con trajectory chart
- Exposure por categoría (politics, crypto, sports)
- Correlation matrix entre posiciones activas
- Daily/weekly performance reports

---

## 6. INFRAESTRUCTURA TÉCNICA (Robustez)

### 6.1 — Base de Datos Local (SQLite)
- Persistir historial de trades, backtests, y configuración
- Cache de mercados para reducir API calls
- Log de todas las señales y decisiones del bot
- Permite análisis offline y debugging

### 6.2 — Rate Limiting Inteligente
- CLOB API: 100 req/min (público), 60 órdenes/min (trading)
- Implementar exponential backoff en 429s
- Queue de órdenes con priorización
- Batch orders (hasta 15 por request)

### 6.3 — Heartbeat API
- Mantener conexión activa con CLOB
- Si desconexión detectada → cancelar órdenes abiertas
- Reconexión automática con estado recovery

### 6.4 — Logging Estructurado
- Cada decisión del bot logeada con timestamp
- Separar logs: trading, signals, errors, performance
- Exportar a JSON para análisis posterior

---

## 7. DASHBOARD v2 (UX)

### 7.1 — Gráficos en Tiempo Real
- Recharts para trajectory de capital
- Orderbook depth visualization
- Price history con candlesticks
- Win rate rolling chart

### 7.2 — Controles Mejorados
- Toggle auto-trade on/off
- Selector de estrategias activas
- Filtros por categoría de mercado
- Manual override para cualquier recomendación

### 7.3 — Alertas
- Toast notifications para trades ejecutados
- Alertas de arbitraje detectado
- Warning cuando se acerca al límite diario
- Notificaciones de whale movements

### 7.4 — Mobile-Responsive
- Layout adaptativo para móvil
- Controles touch-friendly
- Notificaciones push (vía service worker)

---

## PRIORIZACIÓN PARA $100 USD DE CAPITAL

| # | Mejora | Impacto | Esfuerzo | Prioridad |
|---|--------|---------|----------|-----------|
| 1 | Bayesian probability model | ALTO | Medio | P0 |
| 2 | WebSocket real-time data | ALTO | Medio | P0 |
| 3 | Arbitraje intra-mercado (YES+NO) | ALTO | Bajo | P0 |
| 4 | Circuit breakers & risk mgmt | ALTO | Bajo | P0 |
| 5 | Whale tracking (Data API) | ALTO | Medio | P1 |
| 6 | Claude API market evaluation | MEDIO | Bajo | P1 |
| 7 | SQLite persistence | MEDIO | Bajo | P1 |
| 8 | Momentum/volume detection | MEDIO | Medio | P1 |
| 9 | Backtest mejorado con datos reales | MEDIO | Medio | P2 |
| 10 | Copy trading execution | MEDIO | Alto | P2 |
| 11 | Dashboard charts (Recharts) | BAJO | Medio | P2 |
| 12 | Market making | BAJO | Alto | P3 |
| 13 | Cross-platform arbitrage (Kalshi) | BAJO | Alto | P3 |

---

## MÉTRICAS DE ÉXITO

Para un capital de $100:
- **Target mensual conservador:** +5-15% ($5-15 profit)
- **Win rate objetivo:** >55% (break-even ~50% en binary markets)
- **Max drawdown aceptable:** -20% ($20)
- **Sharpe ratio objetivo:** >1.0
- **Número de trades/día:** 3-10

---

*Documento generado por Carlos David Donoso Cordero (ddchack)*
