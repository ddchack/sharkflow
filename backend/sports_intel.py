# SharkFlow by Carlos David Donoso Cordero (ddchack)
"""
══════════════════════════════════════════════════════════════
SharkFlow Sports Intelligence Module
Developed by: Carlos David Donoso Cordero (ddchack)
══════════════════════════════════════════════════════════════

Construye señales deportivas reales para el ensemble de predicción:
  - Ratings ELO desde historial ESPN (2 temporadas)
  - Partes de lesiones via ESPN API
  - Ajuste por altitud del estadio (home advantage)
  - Señales de noticias via NewsAPI (clave opcional)

Uso:
    si = SportsIntelligence(newsapi_key=os.getenv("NEWSAPI_KEY"))
    await si.async_init()                  # carga caché o construye desde ESPN
    ctx = si.get_context(question)         # llamada sync desde evaluate()
    # → {"elo_a": 1542, "elo_b": 1488, "injury_adj": -0.02, ...}
"""

import asyncio
import json
import logging
import math
import os
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────

_CACHE_FILE = Path(__file__).parent / "sports_elo_cache.json"
_CACHE_TTL_HOURS = 24          # Refrescar ELO cada 24 horas
_BASE_ELO = 1500.0
_K_FACTOR = 32.0               # K-factor ELO estándar
_K_ESTABLISHED = 16.0          # K para equipos con >50 partidos en caché

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

# Ligas soportadas → ruta ESPN
ESPN_LEAGUES: dict[str, str] = {
    "nba":             "basketball/nba",
    "nfl":             "football/nfl",
    "mlb":             "baseball/mlb",
    "nhl":             "hockey/nhl",
    "ncaab":           "basketball/mens-college-basketball",
    "soccer_epl":      "soccer/eng.1",
    "soccer_laliga":   "soccer/esp.1",
    "soccer_champions":"soccer/uefa.champions",
    "soccer_mls":      "soccer/usa.1",
}

# ─────────────────────────────────────────────────────────────
# ALTITUD DE ESTADIOS (metros sobre el nivel del mar)
# Fuente: Wikipedia / datos públicos
# Ventaja estimada por altitud: +0.5% win-rate por cada 500m > 500m
# ─────────────────────────────────────────────────────────────

STADIUM_ALTITUDE_M: dict[str, int] = {
    # NBA
    "DEN": 1609,   # Nuggets — Ball Arena, Denver CO
    # NFL
    "DEN": 1609,   # Broncos — Empower Field, Denver CO
    # MLB
    "COL": 1577,   # Rockies — Coors Field (record con aires raros)
    "ARI": 340,    # Diamondbacks — Chase Field, Phoenix AZ
    # NHL
    "COL": 1609,   # Avalanche — Ball Arena, Denver CO
    # Soccer / Internacional
    "Bolivia": 3640,   # Estadio Hernando Siles, La Paz
    "Colombia": 2600,  # Estadio El Campín, Bogotá
    "Ecuador": 2850,   # Estadio Olímpico, Quito
    "Peru": 3170,      # Estadio Nacional, Lima
    "Mexico": 2240,    # Azteca, CDMX
}

_SEA_LEVEL = 500   # metros — por debajo no hay ventaja significativa


def _altitude_boost(team_abbr: str, is_home: bool = True) -> float:
    """
    BUG 5 fix: retorna ajuste de probabilidad por altitud teniendo en cuenta
    si el equipo es LOCAL o VISITANTE.

    - Local en altitud alta → ventaja positiva (aclimatación, presión, etc.)
    - Visitante que va a jugar en altitud alta → desventaja (50% del efecto local)

    +0.5% win-rate por cada 500m sobre el umbral de 500m. Máximo +8% (local).
    """
    alt = STADIUM_ALTITUDE_M.get(team_abbr.upper(), 0)
    if alt <= _SEA_LEVEL:
        return 0.0
    extra_m = alt - _SEA_LEVEL
    boost_magnitude = min(0.08, (extra_m / 500) * 0.005)
    if is_home:
        return round(boost_magnitude, 4)         # Local en altitud: ventaja completa
    else:
        return round(-boost_magnitude * 0.5, 4)  # Visitante en altitud: desventaja parcial


# ─────────────────────────────────────────────────────────────
# MAPEO NOMBRE EQUIPO → ABREVIATURA ESPN
# Cubre las variaciones más comunes en preguntas de Polymarket
# ─────────────────────────────────────────────────────────────

_TEAM_ALIASES: dict[str, str] = {
    # ─── BUG 6: COLISIONES CONOCIDAS ────────────────────────────────────────
    # Los aliases de nombre corto sin deporte prefijado tienen colisiones:
    #   "cardinals"  → ARI (NFL Arizona Cardinals) o STL (MLB St. Louis Cardinals)
    #   "giants"     → NYG (NFL New York Giants) o SF (MLB San Francisco Giants)
    #   "rangers"    → TEX (MLB Texas Rangers) o NYR (NHL New York Rangers)
    #   "jets"       → NYJ (NFL New York Jets) o WPG (NHL Winnipeg Jets)
    #   "kings"      → SAC (NBA Sacramento Kings) o LAK (NHL Los Angeles Kings)
    #   "bears/bulls/hawks/blackhawks" → CHI (comparten abreviatura Chicago)
    #   "panthers"   → CAR (NFL Carolina Panthers) o FLA (NHL Florida Panthers)
    #   "blues/blues" → STL (NHL St. Louis Blues)
    #   "dolphins"   → MIA (NBA Miami Heat) y MIA (NFL Miami Dolphins) — mismo valor, OK
    #
    # La colisión se resuelve por ÚLTIMO alias escrito (el de MLB/NHL overwrite al NFL/NBA).
    # TODO: implementar aliases por deporte (_TEAM_ALIASES_NFL, _TEAM_ALIASES_MLB, etc.)
    # y detectar el deporte desde la pregunta antes de hacer la búsqueda.
    # ──────────────────────────────────────────────────────────────────────────
    # NBA
    "lakers": "LAL", "los angeles lakers": "LAL",
    "celtics": "BOS", "boston celtics": "BOS",
    "warriors": "GSW", "golden state warriors": "GSW",
    "bulls": "CHI", "chicago bulls": "CHI",
    "nets": "BKN", "brooklyn nets": "BKN",
    "heat": "MIA", "miami heat": "MIA",
    "bucks": "MIL", "milwaukee bucks": "MIL",
    "76ers": "PHI", "sixers": "PHI", "philadelphia 76ers": "PHI",
    "suns": "PHX", "phoenix suns": "PHX",
    "nuggets": "DEN", "denver nuggets": "DEN",
    "mavericks": "DAL", "mavs": "DAL", "dallas mavericks": "DAL",
    "clippers": "LAC", "la clippers": "LAC", "los angeles clippers": "LAC",
    "knicks": "NYK", "new york knicks": "NYK",
    "thunder": "OKC", "oklahoma city thunder": "OKC",
    "spurs": "SAS", "san antonio spurs": "SAS",
    "jazz": "UTA", "utah jazz": "UTA",
    "blazers": "POR", "trail blazers": "POR", "portland trail blazers": "POR",
    "timberwolves": "MIN", "minnesota timberwolves": "MIN",
    "grizzlies": "MEM", "memphis grizzlies": "MEM",
    "pelicans": "NOP", "new orleans pelicans": "NOP",
    "hawks": "ATL", "atlanta hawks": "ATL",
    "raptors": "TOR", "toronto raptors": "TOR",
    "cavaliers": "CLE", "cavs": "CLE", "cleveland cavaliers": "CLE",
    "pacers": "IND", "indiana pacers": "IND",
    "pistons": "DET", "detroit pistons": "DET",
    "hornets": "CHA", "charlotte hornets": "CHA",
    "wizards": "WAS", "washington wizards": "WAS",
    "magic": "ORL", "orlando magic": "ORL",
    "kings": "SAC", "sacramento kings": "SAC",
    "rockets": "HOU", "houston rockets": "HOU",
    # NFL
    "chiefs": "KC", "kansas city chiefs": "KC",
    "patriots": "NE", "new england patriots": "NE",
    "eagles": "PHI", "philadelphia eagles": "PHI",
    "cowboys": "DAL", "dallas cowboys": "DAL",
    "49ers": "SF", "san francisco 49ers": "SF",
    "ravens": "BAL", "baltimore ravens": "BAL",
    "bills": "BUF", "buffalo bills": "BUF",
    "bengals": "CIN", "cincinnati bengals": "CIN",
    "chargers": "LAC", "los angeles chargers": "LAC",
    "packers": "GB", "green bay packers": "GB",
    "seahawks": "SEA", "seattle seahawks": "SEA",
    "titans": "TEN", "tennessee titans": "TEN",
    "bears": "CHI", "chicago bears": "CHI",
    "lions": "DET", "detroit lions": "DET",
    "vikings": "MIN", "minnesota vikings": "MIN",
    "broncos": "DEN", "denver broncos": "DEN",
    "raiders": "LV", "las vegas raiders": "LV",
    "steelers": "PIT", "pittsburgh steelers": "PIT",
    "giants": "NYG", "new york giants": "NYG",
    "jets": "NYJ", "new york jets": "NYJ",
    "dolphins": "MIA", "miami dolphins": "MIA",
    "colts": "IND", "indianapolis colts": "IND",
    "texans": "HOU", "houston texans": "HOU",
    "jaguars": "JAX", "jacksonville jaguars": "JAX",
    "cardinals": "ARI", "arizona cardinals": "ARI",
    "rams": "LAR", "los angeles rams": "LAR",
    "buccaneers": "TB", "tampa bay buccaneers": "TB",
    "falcons": "ATL", "atlanta falcons": "ATL",
    "saints": "NO", "new orleans saints": "NO",
    "panthers": "CAR", "carolina panthers": "CAR",
    "commanders": "WAS", "washington commanders": "WAS",
    "browns": "CLE", "cleveland browns": "CLE",
    # MLB
    "yankees": "NYY", "new york yankees": "NYY",
    "dodgers": "LAD", "los angeles dodgers": "LAD",
    "red sox": "BOS", "boston red sox": "BOS",
    "astros": "HOU", "houston astros": "HOU",
    "cubs": "CHC", "chicago cubs": "CHC",
    "white sox": "CWS", "chicago white sox": "CWS",
    "mets": "NYM", "new york mets": "NYM",
    "braves": "ATL", "atlanta braves": "ATL",
    "nationals": "WSH", "washington nationals": "WSH",
    "phillies": "PHI", "philadelphia phillies": "PHI",
    "cardinals": "STL", "st. louis cardinals": "STL",
    "brewers": "MIL", "milwaukee brewers": "MIL",
    "reds": "CIN", "cincinnati reds": "CIN",
    "pirates": "PIT", "pittsburgh pirates": "PIT",
    "rockies": "COL", "colorado rockies": "COL",
    "padres": "SD", "san diego padres": "SD",
    "giants": "SF", "san francisco giants": "SF",
    "diamondbacks": "ARI", "arizona diamondbacks": "ARI",
    "mariners": "SEA", "seattle mariners": "SEA",
    "athletics": "OAK", "oakland athletics": "OAK",
    "rangers": "TEX", "texas rangers": "TEX",
    "angels": "LAA", "los angeles angels": "LAA",
    "rays": "TB", "tampa bay rays": "TB",
    "orioles": "BAL", "baltimore orioles": "BAL",
    "blue jays": "TOR", "toronto blue jays": "TOR",
    "twins": "MIN", "minnesota twins": "MIN",
    "guardians": "CLE", "cleveland guardians": "CLE",
    "tigers": "DET", "detroit tigers": "DET",
    "royals": "KC", "kansas city royals": "KC",
    # NHL
    "blackhawks": "CHI", "chicago blackhawks": "CHI",
    "penguins": "PIT", "pittsburgh penguins": "PIT",
    "rangers": "NYR", "new york rangers": "NYR",
    "maple leafs": "TOR", "toronto maple leafs": "TOR",
    "canadiens": "MTL", "montreal canadiens": "MTL",
    "bruins": "BOS", "boston bruins": "BOS",
    "red wings": "DET", "detroit red wings": "DET",
    "flyers": "PHI", "philadelphia flyers": "PHI",
    "capitals": "WSH", "washington capitals": "WSH",
    "hurricanes": "CAR", "carolina hurricanes": "CAR",
    "lightning": "TB", "tampa bay lightning": "TB",
    "panthers": "FLA", "florida panthers": "FLA",
    "senators": "OTT", "ottawa senators": "OTT",
    "sabres": "BUF", "buffalo sabres": "BUF",
    "islanders": "NYI", "new york islanders": "NYI",
    "devils": "NJD", "new jersey devils": "NJD",
    "avalanche": "COL", "colorado avalanche": "COL",
    "stars": "DAL", "dallas stars": "DAL",
    "predators": "NSH", "nashville predators": "NSH",
    "blues": "STL", "st. louis blues": "STL",
    "jets": "WPG", "winnipeg jets": "WPG",
    "wild": "MIN", "minnesota wild": "MIN",
    "coyotes": "ARI", "arizona coyotes": "ARI",
    "ducks": "ANA", "anaheim ducks": "ANA",
    "sharks": "SJS", "san jose sharks": "SJS",
    "kings": "LAK", "los angeles kings": "LAK",
    "oilers": "EDM", "edmonton oilers": "EDM",
    "flames": "CGY", "calgary flames": "CGY",
    "canucks": "VAN", "vancouver canucks": "VAN",
    # Soccer — selecciones nacionales (Polymarket suele usar nombres de países)
    "portugal": "POR_NT", "brazil": "BRA_NT", "argentina": "ARG_NT",
    "france": "FRA_NT", "spain": "ESP_NT", "germany": "GER_NT",
    "england": "ENG_NT", "italy": "ITA_NT", "netherlands": "NED_NT",
    "usa": "USA_NT", "united states": "USA_NT",
    "mexico": "MEX_NT", "colombia": "COL_NT", "uruguay": "URU_NT",
    # Clubes EPL (nombres cortos más usados en Polymarket)
    "man city": "MCI", "manchester city": "MCI",
    "man utd": "MUN", "manchester united": "MUN",
    "liverpool": "LIV",
    "arsenal": "ARS",
    "chelsea": "CHE",
    "tottenham": "TOT", "spurs": "TOT",
    "aston villa": "AVL",
    "newcastle": "NEW",
    "real madrid": "RMA",
    "barcelona": "BAR", "barca": "BAR",
    "atletico madrid": "ATM", "atletico": "ATM",
    "juventus": "JUV",
    "inter milan": "INT", "inter": "INT",
    "ac milan": "MIL", "milan": "MIL",
    "napoli": "NAP",
    "psg": "PSG", "paris saint-germain": "PSG",
    "bayern munich": "FCB", "bayern": "FCB",
    "dortmund": "BVB", "borussia dortmund": "BVB",
}


def _extract_teams(question: str) -> tuple[str, str]:
    """
    Extrae dos abreviaturas de equipo de una pregunta de Polymarket.
    Retorna ("", "") si no puede identificar los equipos.

    Estrategias:
    1. Patrón "Team A vs Team B" explícito
    2. Scan de aliases en toda la pregunta
    """
    q = question.lower().strip()

    # Estrategia 1: buscar "X vs Y" o "X beat Y" o "X win over Y"
    vs_patterns = [
        r"(.+?)\s+vs\.?\s+(.+?)(?:\s+to\s+|\s+will\s+|\s+on\s+|\?|$)",
        r"(.+?)\s+beat\s+(.+?)(?:\s+to\s+|\s+will\s+|\?|$)",
        r"(.+?)\s+defeats?\s+(.+?)(?:\s+to\s+|\s+will\s+|\?|$)",
    ]
    for pattern in vs_patterns:
        m = re.search(pattern, q)
        if m:
            left = m.group(1).strip()
            right = m.group(2).strip()
            abbr_a = _match_alias(left)
            abbr_b = _match_alias(right)
            if abbr_a and abbr_b:
                return abbr_a, abbr_b

    # Estrategia 2: scan completo, tomar los primeros dos hits
    found = []
    # Ordenar aliases por longitud desc para que "boston red sox" matchee antes que "red"
    for alias, abbr in sorted(_TEAM_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in q:
            if abbr not in found:
                found.append(abbr)
            if len(found) == 2:
                break

    if len(found) == 2:
        return found[0], found[1]
    return "", ""


def _match_alias(text: str) -> str:
    """Busca el alias más largo que matchee en el texto."""
    text = text.strip().lower()
    best = ""
    best_abbr = ""
    for alias, abbr in _TEAM_ALIASES.items():
        if alias in text and len(alias) > len(best):
            best = alias
            best_abbr = abbr
    return best_abbr


# ─────────────────────────────────────────────────────────────
# CLASE PRINCIPAL
# ─────────────────────────────────────────────────────────────

class SportsIntelligence:
    """
    Motor de inteligencia deportiva para SharkFlow.
    Construye ELO desde historial ESPN, lesiones, altitud y noticias.
    """

    def __init__(self, newsapi_key: Optional[str] = None):
        self._newsapi_key = newsapi_key
        self._http = httpx.AsyncClient(
            timeout=15.0,
            headers={"User-Agent": "SharkFlow/6.0 (ddchack Research Bot)"},
            follow_redirects=True,
        )
        # ELO en memoria: abbr → {"elo": float, "games": int}
        self._elo: dict[str, dict] = {}
        # Lesiones: abbr → lista de jugadores lesionados
        self._injuries: dict[str, list] = {}
        # Noticias cache: query → {"sentiment": float, "ts": float}
        self._news_cache: dict[str, dict] = {}
        self._last_elo_refresh: float = 0.0
        self._initialized: bool = False

    # ─── INIT ─────────────────────────────────────────────────

    async def async_init(self):
        """
        Carga el caché de ELO del disco si es fresco (< 24h),
        de lo contrario construye desde ESPN. Luego refresca lesiones.
        """
        if self._try_load_cache():
            logger.info("[SportsIntel] ELO cargado desde caché en disco.")
        else:
            logger.info("[SportsIntel] Construyendo ELO desde ESPN (primera vez)...")
            await self._build_elo_from_records()
            self._save_cache()
        await self._refresh_injuries()
        self._initialized = True

    async def refresh(self):
        """Refresca ELO e injuries (llamar cada 24h)."""
        await self._build_elo_from_records()
        self._save_cache()
        await self._refresh_injuries()
        logger.info("[SportsIntel] ELO e injuries actualizados.")

    # ─── MÉTODO PRINCIPAL (SYNC) ──────────────────────────────

    def get_context(self, question: str) -> dict:
        """
        Extrae contexto deportivo de la pregunta para pasarlo a CategoryEvaluator.
        Retorna dict con elo_a, elo_b, injury_adj, altitude_adj, etc.
        Llamada SÍNCRONA — usa datos en memoria.
        """
        team_a, team_b = _extract_teams(question)

        elo_a = self._elo.get(team_a, {}).get("elo", 0.0) if team_a else 0.0
        elo_b = self._elo.get(team_b, {}).get("elo", 0.0) if team_b else 0.0

        # Lesiones: -2% por jugador titular lesionado (estimación conservadora)
        inj_a = len(self._injuries.get(team_a, []))
        inj_b = len(self._injuries.get(team_b, []))
        # injury_adj: positivo si team_b tiene más lesiones (favorece a team_a)
        injury_adj = round((inj_b - inj_a) * 0.02, 3)

        # Altitud: si alguno de los equipos juega en estadio de alta altitud,
        # ajustar P(team_a wins). Positivo si team_a está en altitud, negativo si team_b.
        # BUG 5 fix: team_a = local (first mentioned = home), team_b = visitante.
        # _extract_teams sigue el orden de aparición en la pregunta.
        # El local recibe boost positivo; el visitante que juega en altitud ajena recibe penalización.
        alt_a = _altitude_boost(team_a, is_home=True)  if team_a else 0.0
        alt_b = _altitude_boost(team_b, is_home=False) if team_b else 0.0
        # alt_b ya es negativo si team_b viaja a altitud alta, así que sumamos (no restamos)
        altitude_adj = round(alt_a + alt_b, 4)  # positivo → favorece team_a

        # Si tenemos ELO real, aplicar injury + altitude al ELO directamente
        if elo_a > 0 and elo_b > 0:
            # Transformar ELO a probabilidad, luego ajustar
            elo_prob = 1.0 / (1.0 + 10 ** (-(elo_a - elo_b) / 400.0))
            elo_prob = max(0.05, min(0.95, elo_prob + injury_adj + altitude_adj))
        else:
            elo_prob = 0.0

        return {
            "elo_a": round(elo_a, 1),
            "elo_b": round(elo_b, 1),
            "elo_prob": round(elo_prob, 4),      # P(team_a wins) según ELO
            "injury_adj": injury_adj,
            "altitude_adj": altitude_adj,
            "injuries_a": self._injuries.get(team_a, [])[:5],
            "injuries_b": self._injuries.get(team_b, [])[:5],
            "team_a": team_a,
            "team_b": team_b,
            "teams_found": bool(team_a and team_b),
            "has_elo": elo_a > 0 and elo_b > 0,
        }

    async def get_news_sentiment(self, question: str) -> float:
        """
        Obtiene sentimiento de noticias para la pregunta (-1 a 1).
        Usa NewsAPI si hay clave, Google RSS como fallback.
        Cachea por 2 horas.
        """
        # Extraer palabras clave del título del evento
        team_a, team_b = _extract_teams(question)
        if not team_a and not team_b:
            return 0.0

        query_parts = []
        if team_a:
            query_parts.append(team_a)
        if team_b:
            query_parts.append(team_b)
        query = " ".join(query_parts)

        # Caché de 2 horas
        cached = self._news_cache.get(query)
        if cached and (time.time() - cached["ts"]) < 7200:
            return cached["sentiment"]

        sentiment = await self._fetch_news_sentiment(query)
        self._news_cache[query] = {"sentiment": sentiment, "ts": time.time()}
        return sentiment

    async def _fetch_news_sentiment(self, query: str) -> float:
        """Fetch y analiza sentimiento de noticias para un query."""
        articles = []

        # Intento 1: NewsAPI (requiere key)
        if self._newsapi_key and self._newsapi_key not in ("", "your_newsapi_key_here"):
            try:
                from_date = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
                resp = await self._http.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": query,
                        "from": from_date,
                        "sortBy": "relevancy",
                        "pageSize": 10,
                        "language": "en",
                        "apiKey": self._newsapi_key,
                    }
                )
                if resp.status_code == 200:
                    articles = resp.json().get("articles", [])
            except Exception as e:
                logger.debug(f"[SportsIntel] NewsAPI error: {e}")

        # Intento 2: Google News RSS (sin key)
        if not articles:
            try:
                encoded = query.replace(" ", "+")
                url = f"https://news.google.com/rss/search?q={encoded}+sports&hl=en-US&gl=US&ceid=US:en"
                resp = await self._http.get(url)
                if resp.status_code == 200:
                    titles = re.findall(r"<title>(.*?)</title>", resp.text)
                    articles = [{"title": t, "description": ""} for t in titles[1:11]]
            except Exception as e:
                logger.debug(f"[SportsIntel] Google RSS error: {e}")

        if not articles:
            return 0.0

        # Análisis simple de sentimiento por keywords
        positive_kw = ["win", "wins", "victory", "dominant", "healthy", "returns",
                        "comeback", "strong", "unbeaten", "streak", "hot", "favor"]
        negative_kw = ["loss", "loses", "injury", "injured", "out", "suspend",
                        "struggle", "slump", "weak", "crisis", "doubt", "scratch"]

        total_sentiment = 0.0
        count = 0
        for a in articles:
            text = (a.get("title", "") + " " + a.get("description", "")).lower()
            pos = sum(1 for k in positive_kw if k in text)
            neg = sum(1 for k in negative_kw if k in text)
            if pos + neg > 0:
                total_sentiment += (pos - neg) / (pos + neg)
                count += 1

        return round(total_sentiment / max(count, 1), 3)

    # ─── ESPN: CONSTRUIR ELO DESDE REGISTROS DE VICTORIA/DERROTA ──

    async def _build_elo_from_records(self):
        """
        Obtiene registros actuales de todos los equipos via ESPN /teams
        y estima ELO desde win_pct usando la fórmula inversa de ELO.

        Luego intenta enriquecer con datos de temporada reciente (juegos).
        """
        tasks = []
        for league, path in ESPN_LEAGUES.items():
            tasks.append(self._fetch_league_records(league, path))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                logger.warning(f"[SportsIntel] Error construyendo ELO: {res}")
            elif res:
                for abbr, data in res.items():
                    if abbr not in self._elo:
                        self._elo[abbr] = data
                    else:
                        # Promediar si el equipo aparece en múltiples ligas
                        self._elo[abbr]["elo"] = (self._elo[abbr]["elo"] + data["elo"]) / 2
                        self._elo[abbr]["games"] = max(self._elo[abbr]["games"], data["games"])

        self._last_elo_refresh = time.time()
        logger.info(f"[SportsIntel] ELO construido para {len(self._elo)} equipos.")

    async def _fetch_league_records(self, league: str, path: str) -> dict[str, dict]:
        """
        Obtiene records win/loss de ESPN via team detail endpoints.
        Paso 1: lista de team IDs. Paso 2: detalle de cada equipo (paralelo).
        Retorna dict abbr → {"elo": float, "games": int, "league": str}
        """
        base_url = f"{ESPN_BASE}/{path}/teams"
        result: dict[str, dict] = {}
        try:
            # Paso 1: obtener lista de teams con sus IDs
            resp = await self._http.get(base_url, params={"limit": 50})
            if resp.status_code != 200:
                return result

            data = resp.json()
            raw_teams = (
                data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
                or data.get("teams", [])
            )

            team_ids = []
            for tw in raw_teams:
                t = tw.get("team", tw)
                tid = t.get("id", "")
                abbr = t.get("abbreviation", "").upper()
                if tid and abbr:
                    team_ids.append((tid, abbr))

            if not team_ids:
                return result

            # Paso 2: fetch detalle en paralelo (incluye record con wins/losses)
            async def fetch_one(tid: str, abbr: str) -> tuple[str, dict]:
                try:
                    r = await self._http.get(f"{base_url}/{tid}")
                    if r.status_code != 200:
                        return abbr, {}
                    td = r.json().get("team", {})
                    record_items = td.get("record", {}).get("items", [])
                    wins = losses = 0
                    for rec in record_items:
                        if rec.get("type") in ("total", "ovr"):
                            stats = {s["name"]: s["value"] for s in rec.get("stats", [])}
                            wins = int(stats.get("wins", 0))
                            losses = int(stats.get("losses", 0))
                            break
                    return abbr, {"wins": wins, "losses": losses}
                except Exception:
                    return abbr, {}

            tasks = [fetch_one(tid, abbr) for tid, abbr in team_ids]
            records = await asyncio.gather(*tasks, return_exceptions=True)

            for res in records:
                if isinstance(res, Exception):
                    continue
                abbr, rec = res
                if not rec or not abbr:
                    continue
                wins = rec.get("wins", 0)
                losses = rec.get("losses", 0)
                games = wins + losses
                if games > 0:
                    win_pct_clamped = max(0.02, min(0.98, wins / games))
                    elo = _BASE_ELO + 400 * math.log10(win_pct_clamped / (1.0 - win_pct_clamped))
                else:
                    elo = _BASE_ELO
                result[abbr] = {
                    "elo": round(elo, 1),
                    "games": games,
                    "wins": wins,
                    "losses": losses,
                    "league": league,
                }

        except Exception as e:
            logger.debug(f"[SportsIntel] {league} teams error: {e}")

        return result

    # ─── ESPN: LESIONES ───────────────────────────────────────

    async def _refresh_injuries(self):
        """
        Obtiene partes de lesiones para las ligas principales.
        ESPN endpoint: /sports/{path}/injuries (si existe)
        """
        tasks = []
        injury_leagues = ["nba", "nfl", "mlb", "nhl"]
        for league in injury_leagues:
            path = ESPN_LEAGUES[league]
            tasks.append(self._fetch_league_injuries(league, path))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, dict):
                for abbr, players in res.items():
                    self._injuries[abbr] = players

        count = sum(len(v) for v in self._injuries.values())
        logger.info(f"[SportsIntel] {count} jugadores lesionados cargados.")

    async def _fetch_league_injuries(self, league: str, path: str) -> dict[str, list]:
        """Fetch injury report para una liga. Retorna abbr → [jugadores]."""
        result: dict[str, list] = {}
        # ESPN no tiene un endpoint público /injuries — usamos scoreboard actual
        # y parseamos el estado de los players de cada equipo.
        # Alternativa: /teams/{id}/roster?season=2025 y filtrar "questionable"/"out"
        # Por ahora usamos los atletas del scoreboard reciente.
        try:
            url = f"{ESPN_BASE}/{path}/scoreboard"
            resp = await self._http.get(url, params={"limit": 5})
            if resp.status_code != 200:
                return result

            data = resp.json()
            events = data.get("events", [])
            for event in events:
                for comp in event.get("competitions", []):
                    for competitor in comp.get("competitors", []):
                        abbr = competitor.get("team", {}).get("abbreviation", "").upper()
                        if not abbr:
                            continue
                        # Buscar jugadores con estado "Questionable", "Out", "Doubtful"
                        injured = []
                        for athlete in competitor.get("athletes", []):
                            status = athlete.get("status", {})
                            if isinstance(status, dict):
                                status_type = status.get("type", {}).get("name", "")
                            else:
                                status_type = str(status)
                            if any(s in status_type.lower() for s in ["questionable", "out", "doubtful", "injured"]):
                                name = athlete.get("athlete", {}).get("displayName", "?")
                                injured.append(name)
                        if injured:
                            result[abbr] = result.get(abbr, []) + injured

        except Exception as e:
            logger.debug(f"[SportsIntel] {league} injuries error: {e}")

        return result

    # ─── CACHÉ EN DISCO ───────────────────────────────────────

    def _try_load_cache(self) -> bool:
        """
        Carga el caché de ELO desde disco.
        Retorna True si el caché existe y es fresco (< 24h).
        """
        if not _CACHE_FILE.exists():
            return False
        try:
            with open(_CACHE_FILE, encoding="utf-8") as f:
                data = json.load(f)
            ts = data.get("timestamp", 0)
            age_h = (time.time() - ts) / 3600
            if age_h > _CACHE_TTL_HOURS:
                logger.info(f"[SportsIntel] Caché tiene {age_h:.1f}h — demasiado viejo, reconstruyendo.")
                return False
            self._elo = data.get("elo", {})
            self._last_elo_refresh = ts
            return True
        except Exception as e:
            logger.warning(f"[SportsIntel] Error leyendo caché: {e}")
            return False

    def _save_cache(self):
        """Persiste ELO en disco."""
        try:
            with open(_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": time.time(),
                    "generated": datetime.now().isoformat(),
                    "teams_count": len(self._elo),
                    "elo": self._elo,
                }, f, indent=2)
            logger.info(f"[SportsIntel] Caché guardado: {len(self._elo)} equipos.")
        except Exception as e:
            logger.warning(f"[SportsIntel] Error guardando caché: {e}")

    # ─── DIAGNÓSTICO ─────────────────────────────────────────

    def get_status(self) -> dict:
        """Retorna estado del módulo para /api/status."""
        age_h = (time.time() - self._last_elo_refresh) / 3600 if self._last_elo_refresh else None
        return {
            "initialized": self._initialized,
            "teams_with_elo": len(self._elo),
            "teams_with_injuries": len(self._injuries),
            "injured_players_total": sum(len(v) for v in self._injuries.values()),
            "cache_age_hours": round(age_h, 1) if age_h is not None else None,
            "cache_file": str(_CACHE_FILE),
            "newsapi_configured": bool(
                self._newsapi_key and self._newsapi_key not in ("", "your_newsapi_key_here")
            ),
        }

    async def close(self):
        await self._http.aclose()


# ─────────────────────────────────────────────────────────────
# SINGLETON — importar en api_server.py y category_models.py
# ─────────────────────────────────────────────────────────────

_sports_intel: Optional[SportsIntelligence] = None


def get_sports_intel() -> Optional[SportsIntelligence]:
    """Retorna el singleton de SportsIntelligence (puede ser None si no está inicializado)."""
    return _sports_intel


def init_sports_intel(newsapi_key: Optional[str] = None) -> SportsIntelligence:
    """
    Crea (o retorna) el singleton de SportsIntelligence.
    Llamar en startup de api_server.py, luego await si.async_init().
    """
    global _sports_intel
    if _sports_intel is None:
        _sports_intel = SportsIntelligence(newsapi_key=newsapi_key)
    return _sports_intel
