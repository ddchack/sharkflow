"""
SharkFlow Sports Data Client
Integra APIs gratuitas sin autenticación para datos deportivos:
- Polymarket Gamma API: mercados deportivos activos
- ESPN API (no-oficial): scores en vivo NBA/NFL/MLB/NHL/Soccer

NOTA: Este módulo solo obtiene datos (scores, mercados). NO incluye modelos
de predicción deportiva — predecir resultados con alta precisión viola la
naturaleza estocástica de los deportes (especialmente baseball/MLB).
"""

import httpx
import asyncio
import json as _json
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def _parse_outcome_price(outcome_prices, idx: int, default: float = 0.5) -> float:
    """Parsea outcomePrices que puede venir como list o como string JSON."""
    if outcome_prices is None:
        return default
    try:
        if isinstance(outcome_prices, str):
            outcome_prices = _json.loads(outcome_prices)
        if isinstance(outcome_prices, (list, tuple)) and len(outcome_prices) > idx:
            return float(outcome_prices[idx] or default)
    except (ValueError, TypeError, _json.JSONDecodeError):
        pass
    return default

ESPN_LEAGUES = {
    "nba": "basketball/nba",
    "nfl": "football/nfl",
    "mlb": "baseball/mlb",
    "nhl": "hockey/nhl",
    "soccer_epl": "soccer/eng.1",
    "soccer_laliga": "soccer/esp.1",
    "soccer_champions": "soccer/uefa.champions",
    "ncaab": "basketball/mens-college-basketball",
    "ufc": "mma/ufc",
}


class SportsDataClient:
    """Cliente para APIs deportivas gratuitas sin autenticación."""

    ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
    GAMMA_BASE = "https://gamma-api.polymarket.com"
    MLB_BASE = "https://statsapi.mlb.com/api/v1"
    NHL_BASE = "https://api-web.nhle.com/v1"

    def __init__(self):
        self._http = httpx.AsyncClient(timeout=10.0, headers={
            "User-Agent": "SharkFlow/4.0 (Polymarket Research Bot)"
        })

    async def get_espn_scoreboard(self, league: str) -> dict:
        """Obtiene scoreboard en vivo de ESPN para una liga."""
        league_key = league.lower()
        if league_key not in ESPN_LEAGUES:
            return {"league": league, "games": [], "count": 0, "error": "Liga no soportada"}
        espn_path = ESPN_LEAGUES[league_key]
        url = f"{self.ESPN_BASE}/{espn_path}/scoreboard"
        try:
            resp = await self._http.get(url)
            resp.raise_for_status()
            data = resp.json()
            games = []
            for event in data.get("events", []):
                comp = event.get("competitions", [{}])[0]
                competitors = comp.get("competitors", [])
                status = event.get("status", {})
                game = {
                    "id": event.get("id"),
                    "name": event.get("name"),
                    "date": event.get("date"),
                    "status": status.get("type", {}).get("description", "Scheduled"),
                    "status_short": status.get("type", {}).get("shortDetail", ""),
                    "is_live": status.get("type", {}).get("state") == "in",
                    "is_final": status.get("type", {}).get("completed", False),
                    "teams": [],
                }
                for c in competitors:
                    team_info = c.get("team", {})
                    game["teams"].append({
                        "id": team_info.get("id"),
                        "name": team_info.get("displayName", team_info.get("name")),
                        "abbreviation": team_info.get("abbreviation"),
                        "score": c.get("score", "0"),
                        "home_away": c.get("homeAway"),
                        "winner": c.get("winner", False),
                        "logo": team_info.get("logo", ""),
                    })
                games.append(game)
            return {"league": league, "games": games, "count": len(games)}
        except Exception as e:
            logger.warning(f"ESPN {league} error: {e}")
            return {"league": league, "games": [], "count": 0, "error": str(e)}

    async def get_polymarket_sports_markets(self, limit: int = 100) -> list:
        """Obtiene mercados deportivos activos de Polymarket via Gamma API."""
        try:
            resp = await self._http.get(
                f"{self.GAMMA_BASE}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": limit,
                    "tag_slug": "sports",
                }
            )
            if resp.status_code == 200:
                data = resp.json()
                markets = data if isinstance(data, list) else data.get("markets", [])
                result = []
                for m in markets:
                    result.append({
                        "id": m.get("id"),
                        "question": m.get("question", ""),
                        "slug": m.get("slug", ""),
                        "yes_price": _parse_outcome_price(m.get("outcomePrices"), 0),
                        "no_price":  _parse_outcome_price(m.get("outcomePrices"), 1),
                        "volume": float(m.get("volume", 0) or 0),
                        "liquidity": float(m.get("liquidity", 0) or 0),
                        "end_date": m.get("endDate", ""),
                        "tags": [t.get("slug", "") for t in m.get("tags", []) if isinstance(t, dict)],
                    })
                return result
            return []
        except Exception as e:
            logger.warning(f"Gamma sports markets error: {e}")
            return []

    async def get_live_scores_all(self) -> dict:
        """Obtiene scores en vivo de todas las ligas principales."""
        tasks = {
            league: self.get_espn_scoreboard(league)
            for league in ["nba", "nfl", "mlb", "nhl", "soccer_epl"]
        }
        results = {}
        responses = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for league, resp in zip(tasks.keys(), responses):
            if isinstance(resp, Exception):
                results[league] = {"league": league, "games": [], "error": str(resp)}
            else:
                results[league] = resp
        return results

    async def close(self):
        await self._http.aclose()
