"""
══════════════════════════════════════════════════════════════
SHARKFLOW — Smart Capital Auto-Allocator
Developed by: Carlos David Donoso Cordero (ddchack)
══════════════════════════════════════════════════════════════

Takes a total budget (e.g. $20) and distributes across 5-20 bets:
- Higher confidence/KL = proportionally more capital
- Minimum $1 per bet, maximum 30% of total on single bet
- Time horizon filter: SHORT (<7d), MEDIUM (7-30d), LONG (>30d)
- Uses Multi-Kelly for correlated position sizing
- Respects category diversification limits

Based on research:
- Proportional staking produces 230% more profit than fixed
- Optimal diversification: 5-10 simultaneous positions
- Quarter Kelly recommended for small bankrolls
"""

import math
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional

from advanced_math import KLDivergence, MultiKelly, GaussianCopula, BinarySharpe


@dataclass
class AllocationSlot:
    """A single bet in the auto-allocation portfolio."""
    market_id: str
    question: str
    category: str
    side: str                 # YES or NO
    market_price: float
    ensemble_prob: float
    kl_divergence: float
    binary_sharpe: float
    edge_pct: float
    confidence: float
    allocated_usd: float      # Final allocation amount
    allocation_pct: float     # % of total budget
    priority_rank: int
    time_horizon: str         # SHORT, MEDIUM, LONG
    days_to_resolution: int
    yes_token: str
    no_token: str
    reasoning: list


@dataclass
class AllocationPlan:
    """Complete auto-allocation plan for the user."""
    plan_id: str
    total_budget: float
    time_horizon: str
    min_bets: int
    max_bets: int
    slots: list               # list[AllocationSlot]
    total_allocated: float
    reserve_usd: float        # Unallocated cash buffer
    expected_ev: float        # Portfolio expected value
    portfolio_sharpe: float
    effective_exposure: float
    category_breakdown: dict  # {category: total_usd}
    created_at: str


class AutoAllocator:
    """
    Distributes capital across multiple bets optimally.
    __signature__ = "Carlos David Donoso Cordero (ddchack)"
    """

    # Hard limits
    MIN_BET_USD = 1.00
    MAX_SINGLE_BET_PCT = 0.30   # Max 30% of budget on one bet
    RESERVE_PCT = 0.05           # Keep 5% as cash reserve
    MAX_PER_CATEGORY_PCT = 0.40  # Max 40% in any single category

    def __init__(self, kelly_fraction: float = 0.25):
        self.kelly_fraction = kelly_fraction

    def allocate(self, recommendations: list[dict], total_budget: float,
                 time_horizon: str = "ALL", min_bets: int = 5,
                 max_bets: int = 20) -> AllocationPlan:
        """
        Main allocation method.

        Args:
            recommendations: list from /api/scan endpoint
            total_budget: total USD to distribute (e.g. $20)
            time_horizon: "SHORT" (<7d), "MEDIUM" (7-30d), "LONG" (>30d), "ALL"
            min_bets: minimum number of bets (default 5)
            max_bets: maximum number of bets (default 20)

        Returns: AllocationPlan with optimized distribution
        """
        # Validar input: filtrar recomendaciones corruptas sin los campos mínimos
        recommendations = [r for r in recommendations
                           if isinstance(r, dict) and "ensemble_prob" in r and "market_price" in r]
        if not recommendations:
            return self._empty_plan(total_budget, time_horizon, min_bets, max_bets)

        # ── Step 1: Filter by time horizon ──
        filtered = self._filter_by_horizon(recommendations, time_horizon)
        if len(filtered) < min_bets:
            # Relax filter if not enough markets
            filtered = recommendations

        # ── Step 2: Filter actionable (KL > threshold) ──
        actionable = [r for r in filtered if r.get("kl_divergence", 0) > 0.005]
        if len(actionable) < min_bets:
            actionable = filtered[:max_bets]

        # ── Step 3: Select top N candidates ──
        # Already sorted by KL from scan endpoint
        candidates = actionable[:max_bets]
        n_bets = max(min_bets, min(len(candidates), max_bets))
        candidates = candidates[:n_bets]

        if not candidates:
            return self._empty_plan(total_budget, time_horizon, min_bets, max_bets)

        # ── Step 4: Calculate raw allocations via Multi-Kelly ──
        kelly_bets = [{"p": r.get("ensemble_prob", 0.5),
                        "price": r.get("market_price", 0.5)}
                       for r in candidates]
        raw_fracs = MultiKelly.multi_kelly_optimize(kelly_bets, self.kelly_fraction)

        # ── Step 5: Apply proportional KL-weighted adjustment ──
        # Higher KL = proportionally more capital
        kl_values = [max(0.001, r.get("kl_divergence", 0.001)) for r in candidates]
        kl_total = sum(kl_values)
        kl_weights = [kl / kl_total for kl in kl_values]

        # Blend Kelly fractions with KL weights (60% Kelly, 40% KL)
        blended = []
        for i in range(len(candidates)):
            blend = raw_fracs[i] * 0.6 + kl_weights[i] * 0.4
            blended.append(blend)

        # ── Step 6: Normalize and apply budget constraints ──
        reserve = total_budget * self.RESERVE_PCT
        allocatable = total_budget - reserve
        max_single = total_budget * self.MAX_SINGLE_BET_PCT

        blend_total = sum(blended)
        if blend_total <= 0:
            blend_total = 1.0

        # First pass: raw USD allocations
        raw_usd = [(b / blend_total) * allocatable for b in blended]

        # Apply constraints
        slots = []
        category_totals = {}

        for i, rec in enumerate(candidates):
            cat = rec.get("category", "other")
            cat_max = total_budget * self.MAX_PER_CATEGORY_PCT
            cat_current = category_totals.get(cat, 0)

            usd = raw_usd[i]
            # Cap at max single bet
            usd = min(usd, max_single)
            # Cap at category limit
            usd = min(usd, cat_max - cat_current)
            # Floor at minimum
            if usd < self.MIN_BET_USD:
                if allocatable - sum(s.allocated_usd for s in slots) >= self.MIN_BET_USD:
                    usd = self.MIN_BET_USD
                else:
                    continue
            usd = round(usd, 2)

            # Parse time horizon for this market
            days = self._days_to_resolution(rec.get("end_date", ""))
            if days < 7:
                th = "SHORT"
            elif days < 30:
                th = "MEDIUM"
            else:
                th = "LONG"

            slot = AllocationSlot(
                market_id=rec.get("market_id", ""),
                question=rec.get("question", ""),
                category=cat,
                side=rec.get("side", "YES"),
                market_price=rec.get("market_price", 0),
                ensemble_prob=rec.get("ensemble_prob", 0.5),
                kl_divergence=rec.get("kl_divergence", 0),
                binary_sharpe=rec.get("binary_sharpe", 0),
                edge_pct=rec.get("edge_pct", 0),
                confidence=rec.get("confidence", 0),
                allocated_usd=usd,
                allocation_pct=round(usd / total_budget * 100, 1),
                priority_rank=i + 1,
                time_horizon=th,
                days_to_resolution=days,
                yes_token=rec.get("yes_token", ""),
                no_token=rec.get("no_token", ""),
                reasoning=rec.get("reasoning", []),
            )
            slots.append(slot)
            category_totals[cat] = category_totals.get(cat, 0) + usd

        # ── Step 7: Calculate portfolio metrics ──
        total_allocated = sum(s.allocated_usd for s in slots)
        portfolio_ev = sum(
            s.allocated_usd * (s.ensemble_prob * (1 / s.market_price - 1) - (1 - s.ensemble_prob))
            for s in slots if s.market_price > 0
        )

        # Portfolio Sharpe
        if slots:
            bets_for_sharpe = [
                {"p": s.ensemble_prob, "m": s.market_price,
                 "w": s.allocated_usd / max(1, total_allocated)}
                for s in slots
            ]
            port_sharpe = BinarySharpe.portfolio(bets_for_sharpe)
        else:
            port_sharpe = 0

        # Effective exposure
        positions = [{"size": s.allocated_usd, "category": s.category,
                       "market_id": s.market_id} for s in slots]
        eff_exp = GaussianCopula.effective_exposure(positions) if positions else 0

        # Category breakdown
        cat_breakdown = {}
        for s in slots:
            cat_breakdown[s.category] = cat_breakdown.get(s.category, 0) + s.allocated_usd

        return AllocationPlan(
            plan_id=f"plan_{datetime.now().strftime('%H%M%S')}",
            total_budget=total_budget,
            time_horizon=time_horizon,
            min_bets=min_bets,
            max_bets=max_bets,
            slots=[self._slot_to_dict(s) for s in slots],
            total_allocated=round(total_allocated, 2),
            reserve_usd=round(total_budget - total_allocated, 2),
            expected_ev=round(portfolio_ev, 2),
            portfolio_sharpe=round(port_sharpe, 3),
            effective_exposure=round(eff_exp, 2),
            category_breakdown={k: round(v, 2) for k, v in cat_breakdown.items()},
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    # ─── Helpers ─────────────────────────────────────────

    def _filter_by_horizon(self, recs: list[dict], horizon: str) -> list[dict]:
        if horizon == "ALL":
            return recs
        filtered = []
        for r in recs:
            days = self._days_to_resolution(r.get("end_date", ""))
            if horizon == "SHORT" and days <= 7:
                filtered.append(r)
            elif horizon == "MEDIUM" and 7 < days <= 30:
                filtered.append(r)
            elif horizon == "LONG" and days > 30:
                filtered.append(r)
        return filtered

    @staticmethod
    def _days_to_resolution(end_date: str) -> int:
        if not end_date:
            return 30  # Default
        try:
            end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            delta = end - datetime.now(timezone.utc)
            return max(0, delta.days)
        except Exception:
            return 30

    @staticmethod
    def _slot_to_dict(slot: AllocationSlot) -> dict:
        return {
            "market_id": slot.market_id,
            "question": slot.question,
            "category": slot.category,
            "side": slot.side,
            "market_price": slot.market_price,
            "ensemble_prob": slot.ensemble_prob,
            "kl_divergence": round(slot.kl_divergence, 4),
            "binary_sharpe": round(slot.binary_sharpe, 3),
            "edge_pct": slot.edge_pct,
            "confidence": slot.confidence,
            "allocated_usd": slot.allocated_usd,
            "allocation_pct": slot.allocation_pct,
            "priority_rank": slot.priority_rank,
            "time_horizon": slot.time_horizon,
            "days_to_resolution": slot.days_to_resolution,
            "yes_token": slot.yes_token,
            "no_token": slot.no_token,
            "reasoning": slot.reasoning[:3],
        }

    @staticmethod
    def _empty_plan(budget, horizon, min_b, max_b) -> AllocationPlan:
        return AllocationPlan(
            plan_id="empty", total_budget=budget, time_horizon=horizon,
            min_bets=min_b, max_bets=max_b, slots=[], total_allocated=0,
            reserve_usd=budget, expected_ev=0, portfolio_sharpe=0,
            effective_exposure=0, category_breakdown={},
            created_at=datetime.now(timezone.utc).isoformat())
