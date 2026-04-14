"""
SharkFlow CalibrationTracker v3.0
Brier Score dinámico por persona del swarm — tracking de accuracy por arquetipo
# SharkFlow by Carlos David Donoso Cordero (ddchack)
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

TRACKER_FILE = os.path.join(os.path.dirname(__file__), "data", "persona_brier_scores.json")
BRIER_EMA_ALPHA = 0.15  # EMA α: ~6 últimas predicciones con peso > 50%


def _load_state() -> dict:
    try:
        os.makedirs(os.path.dirname(TRACKER_FILE), exist_ok=True)
        if os.path.exists(TRACKER_FILE):
            with open(TRACKER_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"CalibrationTracker load error: {e}")
    return {}


def _save_state(state: dict):
    try:
        os.makedirs(os.path.dirname(TRACKER_FILE), exist_ok=True)
        with open(TRACKER_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"CalibrationTracker save error: {e}")


class CalibrationTracker:
    """
    Rastrea el Brier Score EMA de cada persona del swarm.
    Se actualiza cuando un mercado se resuelve.
    # SharkFlow by Carlos David Donoso Cordero (ddchack)
    """

    def __init__(self):
        self._state: dict = _load_state()
        # state[name] = {
        #   "brier_ema": float,
        #   "n_predictions": int,
        #   "total_brier": float,
        #   "last_updated": str,
        #   "archetype": str,
        # }

    def _get_or_init(self, name: str, archetype: str = "", initial_brier: float = 0.25) -> dict:
        if name not in self._state:
            self._state[name] = {
                "brier_ema": initial_brier,
                "n_predictions": 0,
                "total_brier": 0.0,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "archetype": archetype,
            }
        return self._state[name]

    def record_resolution(self, name: str, archetype: str,
                          predicted_prob: float, outcome: int):
        """
        Registra la resolución de un mercado para una persona.

        predicted_prob: probabilidad que dio la persona (0-1)
        outcome: 1 si resolvió YES, 0 si resolvió NO
        """
        brier_score = (predicted_prob - outcome) ** 2
        entry = self._get_or_init(name, archetype)

        old_ema = entry["brier_ema"]
        new_ema = BRIER_EMA_ALPHA * brier_score + (1 - BRIER_EMA_ALPHA) * old_ema

        entry["brier_ema"] = round(new_ema, 5)
        entry["n_predictions"] += 1
        entry["total_brier"] += brier_score
        entry["last_updated"] = datetime.now(timezone.utc).isoformat()

        _save_state(self._state)
        return new_ema

    def get_brier_ema(self, name: str) -> float:
        entry = self._state.get(name, {})
        return entry.get("brier_ema", 0.25)

    def get_all_scores(self) -> list:
        """Devuelve tabla completa para dashboard."""
        rows = []
        for name, entry in self._state.items():
            n = entry.get("n_predictions", 0)
            rows.append({
                "name": name,
                "archetype": entry.get("archetype", ""),
                "brier_ema": round(entry.get("brier_ema", 0.25), 4),
                "n_predictions": n,
                "avg_brier": round(entry.get("total_brier", 0) / max(1, n), 4),
                "last_updated": entry.get("last_updated", ""),
                "tier": self._classify_tier(entry.get("brier_ema", 0.25)),
            })
        rows.sort(key=lambda x: x["brier_ema"])
        return rows

    def get_archetype_summary(self) -> list:
        """Agrupa scores por arquetipo para mostrar en dashboard."""
        archetype_data: dict[str, list] = {}
        for name, entry in self._state.items():
            arch = entry.get("archetype", "Unknown")
            # Simplificar arquetipo al grupo principal
            group = arch.split(" — ")[0] if " — " in arch else arch
            if group not in archetype_data:
                archetype_data[group] = []
            archetype_data[group].append(entry.get("brier_ema", 0.25))

        summary = []
        for group, briers in archetype_data.items():
            avg = sum(briers) / len(briers)
            summary.append({
                "archetype_group": group,
                "avg_brier_ema": round(avg, 4),
                "n_personas": len(briers),
                "tier": self._classify_tier(avg),
            })
        summary.sort(key=lambda x: x["avg_brier_ema"])
        return summary

    @staticmethod
    def _classify_tier(brier_ema: float) -> str:
        if brier_ema < 0.15:
            return "ELITE"
        elif brier_ema < 0.20:
            return "GOOD"
        elif brier_ema < 0.25:
            return "AVERAGE"
        elif brier_ema < 0.30:
            return "POOR"
        else:
            return "BAD"

    def apply_to_swarm_engine(self, swarm_engine) -> int:
        """
        Aplica los Brier scores actualizados al SwarmEngine.
        Retorna el número de personas actualizadas.
        """
        updated = 0
        for name, entry in self._state.items():
            brier = entry.get("brier_ema", 0.25)
            swarm_engine.update_brier(name, brier)
            updated += 1
        return updated


# Singleton
_tracker_instance: Optional[CalibrationTracker] = None


def get_calibration_tracker() -> CalibrationTracker:
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = CalibrationTracker()
    return _tracker_instance
