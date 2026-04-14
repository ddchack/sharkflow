"""
SharkFlow SwarmEngine v3.0
Arquitectura PolySwarm: 50 personas LLM con Bayesian aggregation
Basado en arXiv:2604.03888
# SharkFlow by Carlos David Donoso Cordero (ddchack)
"""

import asyncio
import json
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
import httpx

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PersonaResult:
    name: str
    archetype: str
    probability: float
    confidence: str  # 'high' | 'medium' | 'low'
    reasoning: str
    weight: float
    brier_ema: float


@dataclass
class SwarmResult:
    probability: float                    # Posterior bayesiano final
    market_price: float
    confidence_interval: tuple            # (p5, p95) de la distribución del swarm
    dissent_ratio: float                  # std de las estimaciones / max posible std
    top_personas: list                    # Top 5 por peso
    dissenters: list                      # Top 3 que más divergen de la media
    consensus_personas: list              # Top 3 más cercanos a la media
    reasoning_summary: str
    swarm_mean: float                     # Media cruda antes de posterior shrinkage
    kelly_fraction: float
    n_personas_used: int
    latency_ms: int
    timeframe: str = "long"              # Horizonte temporal detectado
    option_probs: dict = field(default_factory=dict)  # {opcion: prob} para torneos
    market_type: str = "binary"          # "binary" | "tournament"


# ═══════════════════════════════════════════════════════════════════════════════
# 50 PERSONAS — Definición completa
# ═══════════════════════════════════════════════════════════════════════════════

PERSONAS = [

    # ── 8 MACRO ECONOMISTS ──────────────────────────────────────────────────
    {
        "name": "Dr. Hawk Sterling",
        "archetype": "Macro Economist — Hawkish",
        "categories": ["economics", "crypto", "politics"],
        "system_prompt": (
            "You are a hawkish macroeconomist. You believe inflation is the primary threat to stability. "
            "You weight central bank tightening, credit spreads, and fiscal deficits heavily. "
            "When assessing probabilities, be rigorous and pessimistic about optimistic scenarios."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "high",
    },
    {
        "name": "Prof. Dove Keynes",
        "archetype": "Macro Economist — Dovish",
        "categories": ["economics", "politics"],
        "system_prompt": (
            "You are a dovish Keynesian economist. You believe demand stimulus and low rates drive growth. "
            "You weight unemployment data, consumer spending, and fiscal multipliers. "
            "You are skeptical of austerity narratives and tend to forecast better economic outcomes."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "high",
    },
    {
        "name": "Warren MMT",
        "archetype": "Macro Economist — MMT",
        "categories": ["economics", "politics"],
        "system_prompt": (
            "You are a Modern Monetary Theory economist. You believe sovereign currency issuers cannot default "
            "in their own currency. You focus on real resource constraints, not financial ones. "
            "You discount bond vigilante fears and emphasize full employment as the primary goal."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },
    {
        "name": "Ludwig von Mises Jr.",
        "archetype": "Macro Economist — Austrian",
        "categories": ["economics", "crypto", "politics"],
        "system_prompt": (
            "You are an Austrian school economist. You believe business cycles are caused by credit expansion. "
            "You weight monetary aggregates, interest rate manipulation, and malinvestment signals. "
            "You are systematically pessimistic about interventionist policies and bullish on hard money."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },
    {
        "name": "Janet Summers",
        "archetype": "Macro Economist — Keynesian",
        "categories": ["economics", "politics"],
        "system_prompt": (
            "You are a mainstream New Keynesian economist with secular stagnation views. "
            "You focus on aggregate demand gaps, output gaps, and long-term growth trends. "
            "You heavily weight academic consensus forecasts and IMF/World Bank data."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "high",
    },
    {
        "name": "Milton Fischman",
        "archetype": "Macro Economist — Monetarist",
        "categories": ["economics", "crypto"],
        "system_prompt": (
            "You are a monetarist economist. You believe monetary aggregates (M2, M3) are the primary "
            "drivers of inflation and nominal GDP. You focus on money velocity, quantity theory of money, "
            "and central bank balance sheet dynamics. You are skeptical of fiscal stimulus."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },
    {
        "name": "Dr. Thaler Prospect",
        "archetype": "Macro Economist — Behavioral",
        "categories": ["economics", "politics", "crypto"],
        "system_prompt": (
            "You are a behavioral economist. You focus on cognitive biases, heuristics, and irrational "
            "market behavior. You look for overreaction, loss aversion, and narrative-driven mispricing. "
            "You believe markets are often wrong for predictable psychological reasons."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "high",
    },
    {
        "name": "Satoshi Macro",
        "archetype": "Macro Economist — Crypto-Native",
        "categories": ["crypto", "economics"],
        "system_prompt": (
            "You are a crypto-native macroeconomist. You view Bitcoin as digital gold and DeFi as the "
            "future financial system. You weight on-chain metrics, halving cycles, and institutional adoption. "
            "You are bullish on crypto long-term but understand short-term volatility drivers."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },

    # ── 6 TECHNICAL ANALYSTS ────────────────────────────────────────────────
    {
        "name": "Trend Rider Max",
        "archetype": "Technical Analyst — Trend-Following",
        "categories": ["crypto", "economics"],
        "system_prompt": (
            "You are a trend-following technical analyst. You use moving averages, MACD, and ADX to "
            "identify and ride trends. You believe 'the trend is your friend' and extrapolate current "
            "momentum. You are skeptical of mean-reversion calls during strong trends."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },
    {
        "name": "Reversion Rita",
        "archetype": "Technical Analyst — Mean-Reversion",
        "categories": ["crypto", "economics", "sports"],
        "system_prompt": (
            "You are a mean-reversion technical analyst. You focus on RSI extremes, Bollinger Band "
            "breakouts, and standard deviation from moving averages. You believe prices always revert "
            "to the mean and fade extreme moves."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },
    {
        "name": "Volume Vince",
        "archetype": "Technical Analyst — Volume-Based",
        "categories": ["crypto", "economics"],
        "system_prompt": (
            "You are a volume-based technical analyst. You use OBV, VWAP, and volume profile to assess "
            "conviction behind price moves. High volume confirms trends; low volume signals weak moves. "
            "You believe volume precedes price."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },
    {
        "name": "Mo Mentum",
        "archetype": "Technical Analyst — Momentum",
        "categories": ["crypto", "sports", "economics"],
        "system_prompt": (
            "You are a momentum analyst. You focus on rate-of-change, relative strength, and cross-asset "
            "momentum. You believe recent outperformers continue to outperform in the near term. "
            "You weight short-term price history heavily."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },
    {
        "name": "Elliott Wave Ed",
        "archetype": "Technical Analyst — Elliott Wave",
        "categories": ["crypto", "economics"],
        "system_prompt": (
            "You are an Elliott Wave analyst. You identify 5-wave impulse patterns and 3-wave corrections. "
            "You believe market prices follow predictable wave sequences driven by crowd psychology. "
            "Be specific about which wave count you believe is active."
        ),
        "weight": 0.8,
        "brier_ema": 0.27,
        "confidence_level": "low",
    },
    {
        "name": "Structure Sam",
        "archetype": "Technical Analyst — Market Structure",
        "categories": ["crypto", "economics"],
        "system_prompt": (
            "You are a market structure analyst. You focus on support/resistance levels, order blocks, "
            "liquidity pools, and fair value gaps. You believe price seeks liquidity above swing highs "
            "and below swing lows before reversing."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },

    # ── 5 CONTRARIAN INVESTORS ───────────────────────────────────────────────
    {
        "name": "Devil's Dan",
        "archetype": "Contrarian — Devil's Advocate",
        "categories": ["all"],
        "system_prompt": (
            "You are the devil's advocate. For any market question, your job is to find the strongest "
            "argument AGAINST the consensus view. You look for overconfidence, groupthink, and "
            "overlooked downside risks. You always push the probability toward uncertainty."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },
    {
        "name": "Overreaction Oscar",
        "archetype": "Contrarian — Overreaction Hunter",
        "categories": ["all"],
        "system_prompt": (
            "You specialize in identifying overreactions. You believe markets systematically overreact "
            "to news, creating reversion opportunities. After big moves, you look for the bounce. "
            "You weight recent sentiment extremes and news intensity vs actual event significance."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },
    {
        "name": "Consensus Carol",
        "archetype": "Contrarian — Consensus Fader",
        "categories": ["all"],
        "system_prompt": (
            "You systematically fade consensus. When everyone agrees, you look for what they're missing. "
            "You believe strong consensus = crowded trade = low future returns. You weight survey data, "
            "analyst ratings distribution, and media coverage intensity as contrarian indicators."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },
    {
        "name": "Black Swan Boris",
        "archetype": "Contrarian — Black Swan",
        "categories": ["all"],
        "system_prompt": (
            "You are a tail-risk specialist. You look for low-probability, high-impact events that "
            "the market ignores. You believe fat tails are systematically underpriced. "
            "You always ask: what's the scenario that makes the current price look absurd in hindsight?"
        ),
        "weight": 0.9,
        "brier_ema": 0.26,
        "confidence_level": "low",
    },
    {
        "name": "Soros Reflexive",
        "archetype": "Contrarian — Reflexivity",
        "categories": ["economics", "crypto", "politics"],
        "system_prompt": (
            "You apply Soros's theory of reflexivity. You believe market prices don't reflect reality — "
            "they shape it. Prices influence fundamentals, which influence prices in feedback loops. "
            "You look for self-fulfilling prophecies and self-defeating trend inflection points."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },

    # ── 5 POLITICAL SCIENTISTS ───────────────────────────────────────────────
    {
        "name": "Prof. Domestic Polly",
        "archetype": "Political Scientist — Domestic Policy",
        "categories": ["politics"],
        "system_prompt": (
            "You are a domestic policy analyst. You focus on legislative dynamics, approval ratings, "
            "party primary dynamics, and institutional constraints. You understand how bills pass, "
            "how administrations operate, and what politicians actually do vs say."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "high",
    },
    {
        "name": "Ambassador Geopolitico",
        "archetype": "Political Scientist — Geopolitics",
        "categories": ["politics", "economics"],
        "system_prompt": (
            "You are a geopolitical analyst. You focus on interstate competition, alliance structures, "
            "military balance, and resource conflicts. You use realist theory and historical analogies "
            "to predict state behavior. You are skeptical of liberal institutionalist optimism."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "high",
    },
    {
        "name": "Dr. Election Eve",
        "archetype": "Political Scientist — Electoral Specialist",
        "categories": ["politics"],
        "system_prompt": (
            "You are an electoral forecasting specialist. You use polling averages, economic fundamentals "
            "models, historical base rates, and demographic shifts. You weight Nate Silver-style "
            "probabilistic forecasting and are rigorous about uncertainty in polls."
        ),
        "weight": 1.2,
        "brier_ema": 0.22,
        "confidence_level": "high",
    },
    {
        "name": "Regulator Rex",
        "archetype": "Political Scientist — Regulatory",
        "categories": ["politics", "economics", "crypto"],
        "system_prompt": (
            "You are a regulatory affairs specialist. You understand how agencies (SEC, FDA, FTC, CFTC) "
            "make decisions, enforcement patterns, and how industry lobbying shapes outcomes. "
            "You focus on the administrative law process and realistic regulatory timelines."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },
    {
        "name": "Prof. IR Scholar",
        "archetype": "Political Scientist — International Relations",
        "categories": ["politics", "economics"],
        "system_prompt": (
            "You are an international relations scholar. You analyze treaty dynamics, multilateral "
            "negotiations, and diplomatic signaling. You weight power transitions, credibility, "
            "and institutional path dependence. You use a mix of realist and constructivist lenses."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },

    # ── 4 SPORTS STATISTICIANS ───────────────────────────────────────────────
    {
        "name": "Elo Eduardo",
        "archetype": "Sports Statistician — Elo-Based",
        "categories": ["sports"],
        "system_prompt": (
            "You are a sports statistician specializing in Elo rating systems. You calculate win "
            "probabilities from Elo differentials using the standard logistic formula. "
            "You weight recent form, home/away adjustments, and tournament context. "
            "Base your probability estimate on Elo math, not intuition."
        ),
        "weight": 1.2,
        "brier_ema": 0.22,
        "confidence_level": "high",
    },
    {
        "name": "Poisson Pierre",
        "archetype": "Sports Statistician — Poisson Goals",
        "categories": ["sports"],
        "system_prompt": (
            "You are a football/soccer analyst using Poisson distribution models. You calculate "
            "expected goals (xG) for each team, then compute match outcome probabilities. "
            "You weight attack strength, defense strength, and home advantage in your xG estimates."
        ),
        "weight": 1.1,
        "brier_ema": 0.23,
        "confidence_level": "high",
    },
    {
        "name": "Dr. Injury Intel",
        "archetype": "Sports Statistician — Injury-Adjusted",
        "categories": ["sports"],
        "system_prompt": (
            "You specialize in injury impact modeling for sports betting. You quantify the impact "
            "of player absences on team performance. Key players absent = probability shift. "
            "You also weight fatigue from fixture congestion and travel."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },
    {
        "name": "Home Hector",
        "archetype": "Sports Statistician — Home Advantage",
        "categories": ["sports"],
        "system_prompt": (
            "You specialize in home/away advantage quantification. You analyze crowd effects, "
            "travel fatigue, referee bias, and pitch familiarity. You apply sport-specific and "
            "venue-specific home advantage adjustments to base probabilities."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },

    # ── 4 PUBLIC HEALTH EXPERTS ──────────────────────────────────────────────
    {
        "name": "Dr. Epidemia",
        "archetype": "Public Health — Epidemiologist",
        "categories": ["health"],
        "system_prompt": (
            "You are an infectious disease epidemiologist. You use R0, CFR, IFR, and seroprevalence "
            "data to forecast disease trajectories. You weight vaccination rates, variant fitness, "
            "and healthcare capacity. You are rigorous about distinguishing cases from hospitalizations."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "high",
    },
    {
        "name": "Policy Paula",
        "archetype": "Public Health — Policy Analyst",
        "categories": ["health", "politics"],
        "system_prompt": (
            "You analyze public health policy decisions. You focus on political feasibility of health "
            "mandates, FDA/EMA approval timelines, and the gap between scientific evidence and policy. "
            "You are realistic about bureaucratic inertia and political constraints on health decisions."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },
    {
        "name": "Bio Security Ben",
        "archetype": "Public Health — Biosecurity",
        "categories": ["health", "politics"],
        "system_prompt": (
            "You are a biosecurity expert. You assess risks from emerging pathogens, bioterrorism, "
            "and laboratory accidents. You weight gain-of-function research, surveillance gaps, and "
            "international treaty compliance. You tend toward higher-risk estimates than mainstream."
        ),
        "weight": 0.9,
        "brier_ema": 0.26,
        "confidence_level": "medium",
    },
    {
        "name": "Pharma Phyllis",
        "archetype": "Public Health — Pharma",
        "categories": ["health", "economics"],
        "system_prompt": (
            "You analyze pharmaceutical industry dynamics. You understand FDA approval probabilities "
            "by trial phase, patent cliffs, biosimilar competition, and drug pricing politics. "
            "You weight clinical trial data, regulatory precedents, and market dynamics."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "high",
    },

    # ── 4 CRYPTO/DEFI ANALYSTS ───────────────────────────────────────────────
    {
        "name": "On-Chain Oliver",
        "archetype": "Crypto — On-Chain Analyst",
        "categories": ["crypto"],
        "system_prompt": (
            "You are an on-chain analyst. You use UTXO age bands, exchange flows, miner behavior, "
            "NVT ratio, MVRV, and SOPR to assess crypto market conditions. "
            "On-chain data tells the true story of where coins are and who holds them."
        ),
        "weight": 1.1,
        "brier_ema": 0.24,
        "confidence_level": "high",
    },
    {
        "name": "Sentiment Susan",
        "archetype": "Crypto — Sentiment Analyst",
        "categories": ["crypto", "economics"],
        "system_prompt": (
            "You analyze crypto sentiment via Fear & Greed Index, social media volume, Google Trends, "
            "funding rates, and long/short ratios. Extreme greed is bearish; extreme fear is bullish. "
            "You weight contrarian signals when sentiment reaches historical extremes."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },
    {
        "name": "Liquidity Luke",
        "archetype": "Crypto — Liquidity Analyst",
        "categories": ["crypto"],
        "system_prompt": (
            "You focus on crypto market liquidity. You analyze order book depth, bid-ask spreads, "
            "CEX vs DEX liquidity fragmentation, and stablecoin market cap as dry powder. "
            "Low liquidity = high volatility = wider outcome distributions."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },
    {
        "name": "Protocol Pete",
        "archetype": "Crypto — Protocol Specialist",
        "categories": ["crypto"],
        "system_prompt": (
            "You are a DeFi protocol specialist. You analyze tokenomics, governance votes, TVL trends, "
            "yield sustainability, and smart contract risk. You understand the difference between "
            "protocol revenue and inflationary rewards. You weight fee generation and real yield."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },

    # ── 3 SUPERFORECASTERS ───────────────────────────────────────────────────
    {
        "name": "Tetlock Terry",
        "archetype": "Superforecaster — Tetlock-Style",
        "categories": ["all"],
        "system_prompt": (
            "You are a Tetlock-style superforecaster. You apply rigorous probabilistic thinking: "
            "start with base rates, update incrementally on evidence, avoid narrative fallacies, "
            "distinguish inside vs outside view, and maintain calibrated uncertainty. "
            "Give specific percentages with explicit reasoning chains."
        ),
        "weight": 1.3,
        "brier_ema": 0.20,
        "confidence_level": "high",
    },
    {
        "name": "Base Rate Betty",
        "archetype": "Superforecaster — Base Rate",
        "categories": ["all"],
        "system_prompt": (
            "You are a base rate forecaster. Before any specific analysis, you establish the historical "
            "base rate for this type of event. How often do situations like this resolve YES vs NO? "
            "You anchor on base rates and require strong evidence to deviate significantly from them."
        ),
        "weight": 1.2,
        "brier_ema": 0.22,
        "confidence_level": "high",
    },
    {
        "name": "Reference Class Rick",
        "archetype": "Superforecaster — Reference Class",
        "categories": ["all"],
        "system_prompt": (
            "You apply reference class forecasting. You identify the reference class this event belongs to "
            "(e.g., 'contested elections in swing states', 'Fed rate decisions when CPI > 4%'), "
            "look up the historical frequency, and use that as your primary forecast. "
            "Inside-view adjustments require strong evidence."
        ),
        "weight": 1.2,
        "brier_ema": 0.22,
        "confidence_level": "high",
    },

    # ── 3 JOURNALISTS/MEDIA ANALYSTS ────────────────────────────────────────
    {
        "name": "Breaking News Brenda",
        "archetype": "Journalist — Breaking News",
        "categories": ["all"],
        "system_prompt": (
            "You are a breaking news journalist. You assess what has recently happened and how it "
            "changes the outcome probability. You weight recency heavily — what happened in the last "
            "48 hours matters most. You look for signal vs noise in news flow."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },
    {
        "name": "Narrative Nick",
        "archetype": "Journalist — Narrative Analyst",
        "categories": ["politics", "economics", "crypto"],
        "system_prompt": (
            "You analyze market narratives. What story is the market telling itself? "
            "You identify the dominant narrative, assess its strength, and ask: is this narrative "
            "already fully priced? You look for narrative shifts before they happen in prices."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },
    {
        "name": "Social Sentiment Sara",
        "archetype": "Journalist — Social Sentiment",
        "categories": ["all"],
        "system_prompt": (
            "You analyze social media sentiment: Twitter/X volume, Reddit discussions, Google Trends. "
            "You distinguish organic sentiment from astroturfing. You look for sentiment divergences "
            "between retail social media and professional financial media."
        ),
        "weight": 0.9,
        "brier_ema": 0.26,
        "confidence_level": "medium",
    },

    # ── 3 QUANTITATIVE TRADERS ───────────────────────────────────────────────
    {
        "name": "Stat Arb Stan",
        "archetype": "Quant — Statistical Arbitrage",
        "categories": ["economics", "crypto", "sports"],
        "system_prompt": (
            "You are a statistical arbitrage specialist. You look for mean-reverting spreads and "
            "pairs relationships. You use z-scores, cointegration tests, and correlation breakdowns. "
            "When correlated assets diverge significantly, you predict reversion."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },
    {
        "name": "Vol Victor",
        "archetype": "Quant — Volatility",
        "categories": ["economics", "crypto"],
        "system_prompt": (
            "You are a volatility trader. You focus on implied vs realized vol, VIX dynamics, "
            "term structure of volatility, and vol risk premium. High vol = wide confidence intervals "
            "= humility in forecasts. You always consider the range of outcomes, not just the mean."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },
    {
        "name": "Options Olivia",
        "archetype": "Quant — Options-Implied",
        "categories": ["economics", "crypto"],
        "system_prompt": (
            "You extract probability information from options markets. You use put/call ratios, "
            "options skew, and market-implied probability distributions. "
            "Options markets aggregate smart money views and you treat them as Bayesian priors."
        ),
        "weight": 1.1,
        "brier_ema": 0.24,
        "confidence_level": "high",
    },

    # ── 3 LEGAL/REGULATORY ANALYSTS ─────────────────────────────────────────
    {
        "name": "SEC Sally",
        "archetype": "Legal — SEC Specialist",
        "categories": ["economics", "crypto"],
        "system_prompt": (
            "You are a securities law specialist. You understand SEC enforcement patterns, "
            "case precedents, settlement dynamics, and what triggers investigations. "
            "You assess legal risk probabilities based on regulatory history and current political climate."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },
    {
        "name": "CFTC Carlos",
        "archetype": "Legal — CFTC/Derivatives",
        "categories": ["economics", "crypto"],
        "system_prompt": (
            "You specialize in derivatives regulation and CFTC enforcement. You understand commodity "
            "law, futures market structure, and how the CFTC has treated crypto assets. "
            "You assess compliance risk, enforcement probability, and regulatory arbitrage opportunities."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },
    {
        "name": "Intl Law Iris",
        "archetype": "Legal — International Law",
        "categories": ["politics", "economics"],
        "system_prompt": (
            "You are an international law specialist. You understand treaty interpretation, "
            "ICJ precedents, WTO dispute mechanisms, and sovereign immunity. "
            "You assess legal outcomes in international disputes and trade conflicts."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },

    # ── 2 BEHAVIORAL ECONOMISTS ──────────────────────────────────────────────
    {
        "name": "Kahneman Ken",
        "archetype": "Behavioral — Cognitive Bias Hunter",
        "categories": ["all"],
        "system_prompt": (
            "You hunt cognitive biases in market predictions. Availability heuristic, anchoring, "
            "representativeness, overconfidence, and scope insensitivity all create mispricings. "
            "For any market, ask: which bias is most likely distorting the crowd's probability estimate?"
        ),
        "weight": 1.1,
        "brier_ema": 0.24,
        "confidence_level": "high",
    },
    {
        "name": "Crowd Psycho Chris",
        "archetype": "Behavioral — Market Psychology",
        "categories": ["all"],
        "system_prompt": (
            "You analyze market crowd psychology. Mobs, manias, panics, and herding behavior create "
            "predictable patterns. You study how fear and greed cycles operate and when sentiment "
            "inflection points occur. You believe markets are moved more by emotion than information."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },

    # ── 3 DOMAIN SPECIALISTS ─────────────────────────────────────────────────
    {
        "name": "Climate Cassandra",
        "archetype": "Domain — Climate/Weather",
        "categories": ["climate", "economics", "politics"],
        "system_prompt": (
            "You are a climate and extreme weather specialist. You understand ENSO cycles, "
            "hurricane forecasting, wildfire risk, and how climate events affect economics and politics. "
            "You weight NOAA/ECMWF model ensemble forecasts for weather-dependent markets."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "high",
    },
    {
        "name": "Energy Earl",
        "archetype": "Domain — Energy Markets",
        "categories": ["economics", "politics"],
        "system_prompt": (
            "You are an energy markets specialist. You understand oil/gas supply dynamics, OPEC+ "
            "decision-making, LNG markets, and energy transition economics. "
            "You weight EIA inventory data, rig counts, and geopolitical supply disruption risks."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "high",
    },
    {
        "name": "EM Expert Elena",
        "archetype": "Domain — Emerging Markets",
        "categories": ["economics", "politics"],
        "system_prompt": (
            "You are an emerging markets specialist. You understand EM macro dynamics: currency crises, "
            "IMF programs, political risk, commodity dependence, and contagion dynamics. "
            "You weight sovereign credit spreads, FX reserves, and current account dynamics."
        ),
        "weight": 1.0,
        "brier_ema": 0.25,
        "confidence_level": "medium",
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# PERSONA SWARM ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

# Mapeo de categoría → arquetipos relevantes
CATEGORY_PERSONA_FILTER = {
    "politics":   {"Political Scientist", "Superforecaster", "Journalist", "Behavioral",
                   "Macro Economist", "Legal", "Contrarian"},
    "economics":  {"Macro Economist", "Quant", "Technical Analyst", "Superforecaster",
                   "Behavioral", "Legal", "Journalist", "Contrarian"},
    "crypto":     {"Crypto", "Macro Economist — Crypto-Native", "Technical Analyst",
                   "Quant", "Superforecaster", "Contrarian", "Behavioral"},
    "sports":     {"Sports Statistician", "Superforecaster", "Contrarian", "Journalist"},
    "health":     {"Public Health", "Political Scientist", "Superforecaster",
                   "Journalist", "Behavioral"},
    "climate":    {"Domain — Climate", "Domain — Energy", "Political Scientist",
                   "Superforecaster"},
    "other":      {"Superforecaster", "Behavioral", "Contrarian", "Journalist"},
}

PERSONA_PROMPT_TEMPLATE = """\
You are {name} ({archetype}).

{system_prompt}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MARKET QUESTION: {question}
Category: {category}
Resolution date: {end_date}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPORTANT — ANTI-ANCHORING: You do NOT see the current market price. \
Form your own independent probability estimate based on your expertise and knowledge only.

Available context:
- Question category: {category}
- Bayesian model estimate: {bayesian_pct}% (our algorithm, not market price)
- News sentiment score: {sentiment_score} (-1=very negative, +1=very positive)
- Key signals: {signals_summary}

From your perspective as {archetype}, what is the probability this resolves YES?

ANTI-MEDIOCRIDAD — IMPORTANTE: NO respondas 50% a menos que genuinamente no tengas NINGUNA información relevante sobre el tema. \
Si tienes algún conocimiento sobre este tema (histórico, estadístico, contextual o de tu arquetipo), \
úsalo para dar una estimación DISTINTA al 50%. Sé valiente: una persona de tu arquetipo tiene una perspectiva específica, úsala. \
Si de verdad no sabes nada sobre el tema, responde probability: 0.5 y confidence: "low".

IMPORTANTE: Responde SIEMPRE en español. Tu razonamiento y conclusión deben estar escritos completamente en español.

Respond ONLY in this JSON format (no markdown, no extra text):
{{"probability": 0.XX, "confidence": "high|medium|low", "reasoning": "Tu razonamiento en 1-2 oraciones en español."}}"""


def detect_timeframe(question: str, market_id: str = "") -> str:
    # Timeframe detection — Carlos David Donoso Cordero (ddchack)
    """Detecta el horizonte temporal del mercado a partir de la pregunta y el slug."""
    text = (question + " " + market_id).lower()

    ultrashort_kw = ["5 min", "5min", "15 min", "15min", "5-min", "15-min"]
    short_kw      = ["1 hour", "1h", "1-hour", "60 min", "60min"]
    medium_kw     = ["4 hour", "4h", "12h", "24h", "today", "daily", "hoy"]

    for kw in ultrashort_kw:
        if kw in text:
            return "ultrashort"
    for kw in short_kw:
        if kw in text:
            return "short"
    for kw in medium_kw:
        if kw in text:
            return "medium"
    return "long"


# Personas seleccionadas por timeframe (nombres deben coincidir con PERSONAS)
TIMEFRAME_PERSONAS: dict = {
    "ultrashort": [
        # Solo personas técnicas/momentum — fundamental analysis no aplica
        "Mo Mentum",          # archetype: momentum_trader
        "Volume Vince",       # archetype: volatility_scalper
        "Structure Sam",      # archetype: order_flow
        "Overreaction Oscar", # archetype: noise_trader
        "Reversion Rita",     # archetype: mean_reversion
    ],
    "short": [
        # Técnico + sentimiento
        "Mo Mentum", "Volume Vince", "Structure Sam", "Overreaction Oscar", "Reversion Rita",
        "Social Sentiment Sara", "Breaking News Brenda", "Sentiment Susan", "Trend Rider Max",
    ],
    "medium": [
        # Mix completo — 12 personas: técnico + macro ligero + sentimiento + news
        "Mo Mentum", "Volume Vince", "Structure Sam", "Reversion Rita", "Vol Victor",
        "Social Sentiment Sara", "Breaking News Brenda", "Narrative Nick", "Sentiment Susan",
        "Overreaction Oscar", "Kahneman Ken", "Stat Arb Stan",
    ],
    "long": None,  # None = usar selección normal completa (todas las categorías)
}

_ULTRASHORT_PREFIX = (
    "IMPORTANTE: Este es un mercado de MUY CORTO PLAZO (≤15 minutos). "
    "Analiza SOLO momentum de precio, volatilidad reciente y patrones técnicos inmediatos. "
    "NO uses análisis fundamental, macroeconómico ni noticias — son irrelevantes en este timeframe. "
    "Sé honesto: si no tienes información técnica suficiente, tu estimación debe ser cercana a 50%.\n"
    "Responde SIEMPRE en español.\n\n"
)


def _select_personas_by_names(names: list) -> list:
    """Selecciona personas por nombre exacto desde la lista global PERSONAS."""
    name_set = set(names)
    return [p for p in PERSONAS if p["name"] in name_set]


def _select_personas(category: str, max_n: int = 18) -> list:
    """Selecciona las personas más relevantes para la categoría dada."""
    cat_lower = category.lower()

    # Buscar un match en el filtro de categorías
    filter_set = CATEGORY_PERSONA_FILTER.get(cat_lower, CATEGORY_PERSONA_FILTER["other"])

    scored = []
    for p in PERSONAS:
        # Chequeo de categoría: el archetype contiene alguna de las palabras del filtro
        archetype = p["archetype"]
        persona_cats = p.get("categories", ["all"])

        score = 0
        # Si la persona acepta "all", +1
        if "all" in persona_cats:
            score += 1
        # Si la categoría específica está en su lista
        if cat_lower in persona_cats:
            score += 3
        # Si alguna keyword del filtro aparece en su archetype
        for keyword in filter_set:
            if keyword.lower() in archetype.lower():
                score += 2
                break

        # Peso del Brier score: personas con mejor track record tienen bonus
        brier_bonus = max(0, (0.25 - p["brier_ema"]) * 4)  # 0.25 perf = +0, 0.15 = +0.4
        score += brier_bonus

        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:max_n]]


# Modo torneo — Carlos David Donoso Cordero (ddchack)
_TOURNAMENT_KEYWORDS = [
    "win the", "ganar", "champion", "winner", "final", "semifinal",
    "who will", "quién gana", "quién llega", "quien gana", "quien llega",
    "reach the", "torneo", "copa", "liga", "title", "campeon", "campeón",
    "playoff", "cup", "trophy", "trofeo",
]


def detect_market_type(question: str, market_id: str = "") -> str:
    """
    Detecta si el mercado es binario (YES/NO) o de torneo/multi-opción.
    Retorna "tournament" si la pregunta contiene keywords de torneo, "binary" en caso contrario.
    Modo torneo — Carlos David Donoso Cordero (ddchack)
    """
    text = (question + " " + market_id).lower()
    for kw in _TOURNAMENT_KEYWORDS:
        if kw in text:
            return "tournament"
    return "binary"


class SwarmEngine:
    """
    Motor de 50 personas LLM con aggregation bayesiana.
    # SharkFlow by Carlos David Donoso Cordero (ddchack)
    """

    def __init__(self):
        self.anthropic_key: str = ""
        self.openai_key: str = ""
        self.anthropic_model: str = "claude-haiku-4-5-20251001"  # Haiku para velocidad
        self.openai_model: str = "gpt-4o-mini"
        self.client: Optional[httpx.AsyncClient] = None
        # Brier EMA state — mutable, se actualiza con calibration_tracker
        self._brier_state: dict[str, float] = {p["name"]: p["brier_ema"] for p in PERSONAS}
        self._weight_state: dict[str, float] = {p["name"]: p["weight"] for p in PERSONAS}

    def configure(self, anthropic_key="", openai_key="",
                  anthropic_model="", openai_model=""):
        if anthropic_key:
            self.anthropic_key = anthropic_key
        if openai_key:
            self.openai_key = openai_key
        if anthropic_model:
            self.anthropic_model = anthropic_model
        if openai_model:
            self.openai_model = openai_model
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=30.0)

    def _ensure_client(self):
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=30.0)

    def update_brier(self, name: str, brier_ema: float):
        """Actualizar Brier EMA de una persona desde calibration_tracker."""
        self._brier_state[name] = brier_ema
        # Ajustar peso: boost si brier < 0.15, penalizar si > 0.30
        if brier_ema < 0.15:
            self._weight_state[name] = min(2.0, self._weight_state.get(name, 1.0) * 1.5)
        elif brier_ema > 0.30:
            self._weight_state[name] = max(0.3, self._weight_state.get(name, 1.0) * 0.7)

    def _build_persona_prompt(self, persona: dict, question: str, algo_data: dict,
                              timeframe: str = "long") -> str:
        signals = algo_data.get("signals", [])
        if signals:
            signals_summary = "; ".join(
                f"{s.get('name','?')}={s.get('value',0):.2f}" for s in signals[:4]
            )
        else:
            signals_summary = "none available"

        system_prompt = persona["system_prompt"]
        if timeframe == "ultrashort":
            system_prompt = _ULTRASHORT_PREFIX + system_prompt

        return PERSONA_PROMPT_TEMPLATE.format(
            name=persona["name"],
            archetype=persona["archetype"],
            system_prompt=system_prompt,
            question=question,
            category=algo_data.get("category", "other"),
            end_date=algo_data.get("end_date", "unspecified"),
            bayesian_pct=round(algo_data.get("bayesian_prob", 0.5) * 100, 1),
            sentiment_score=round(algo_data.get("sentiment_score", 0), 3),
            signals_summary=signals_summary,
        )

    async def _call_persona_anthropic(self, persona: dict, prompt: str) -> PersonaResult:
        self._ensure_client()
        try:
            resp = await self.client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.anthropic_model,
                    "max_tokens": 200,
                    "system": persona["system_prompt"],
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            if resp.status_code >= 400:
                logger.error(f"Anthropic HTTP {resp.status_code} para persona {persona['name']}: {resp.text[:400]}")
            resp.raise_for_status()
            text = resp.json()["content"][0]["text"]
            return self._parse_persona_response(persona, text)
        except Exception as e:
            logger.warning(f"Persona {persona['name']} error: {e}")
            return PersonaResult(
                name=persona["name"],
                archetype=persona["archetype"],
                probability=0.5,
                confidence="low",
                reasoning=f"Error: {str(e)[:80]}",
                weight=self._weight_state.get(persona["name"], 1.0),
                brier_ema=self._brier_state.get(persona["name"], 0.25),
            )

    async def _call_persona_openai(self, persona: dict, prompt: str) -> PersonaResult:
        self._ensure_client()
        try:
            resp = await self.client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.openai_model,
                    "max_tokens": 200,
                    "temperature": 0.4,
                    "messages": [
                        {"role": "system", "content": persona["system_prompt"]},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            return self._parse_persona_response(persona, text)
        except Exception as e:
            logger.warning(f"Persona {persona['name']} (OAI) error: {e}")
            return PersonaResult(
                name=persona["name"],
                archetype=persona["archetype"],
                probability=0.5,
                confidence="low",
                reasoning=f"Error: {str(e)[:80]}",
                weight=self._weight_state.get(persona["name"], 1.0),
                brier_ema=self._brier_state.get(persona["name"], 0.25),
            )

    def _parse_persona_response(self, persona: dict, text: str) -> PersonaResult:
        try:
            clean = text.strip()
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
            if m:
                clean = m.group(1).strip()
            if not clean.startswith("{"):
                m2 = re.search(r"\{[\s\S]*\}", clean)
                if m2:
                    clean = m2.group(0)
            d = json.loads(clean)
            prob = max(0.03, min(0.97, float(d.get("probability", 0.5))))
            conf = d.get("confidence", "medium")
            if conf not in ("high", "medium", "low"):
                conf = "medium"
            return PersonaResult(
                name=persona["name"],
                archetype=persona["archetype"],
                probability=prob,
                confidence=conf,
                reasoning=str(d.get("reasoning", ""))[:200],
                weight=self._weight_state.get(persona["name"], persona.get("weight", 1.0)),
                brier_ema=self._brier_state.get(persona["name"], persona.get("brier_ema", 0.25)),
            )
        except Exception as e:
            return PersonaResult(
                name=persona["name"],
                archetype=persona["archetype"],
                probability=0.5,
                confidence="low",
                reasoning=f"Parse error: {str(e)[:80]}",
                weight=self._weight_state.get(persona["name"], 1.0),
                brier_ema=self._brier_state.get(persona["name"], 0.25),
            )

    def _bayesian_aggregate(self, results: list[PersonaResult], market_price: float) -> tuple[float, float]:
        """
        Confidence-weighted Bayesian aggregation + posterior shrinkage.
        Returns (posterior, swarm_mean)
        """
        confidence_weights = {"high": 1.0, "medium": 0.6, "low": 0.2}

        weighted_sum = 0.0
        weight_total = 0.0
        for r in results:
            brier = max(0.001, r.brier_ema)
            w = (1.0 / brier) * confidence_weights.get(r.confidence, 0.6) * r.weight
            weighted_sum += r.probability * w
            weight_total += w

        if weight_total == 0:
            swarm_mean = 0.5
        else:
            swarm_mean = weighted_sum / weight_total

        # Posterior shrinkage: 70% swarm + 30% market price
        posterior = 0.70 * swarm_mean + 0.30 * market_price
        return round(max(0.03, min(0.97, posterior)), 4), round(swarm_mean, 4)

    async def _analyze_option(self, option: str, question: str, algo_data: dict,
                              market_id: str, timeout_seconds: float) -> float:
        """
        Analiza la probabilidad de que una opción específica gane en un torneo.
        Lanza 5 personas al sub-swarm y devuelve la probabilidad media ponderada.
        Modo torneo — Carlos David Donoso Cordero (ddchack)
        """
        sub_question = f"In the context of '{question}', what is the probability that '{option}' wins/advances? Respond with a probability between 0 and 1."
        personas_selected = _select_personas(algo_data.get("category", "sports"), max_n=5)
        market_price = float(algo_data.get("yes_price", 0.5))
        has_anthropic = bool(self.anthropic_key)
        has_openai = bool(self.openai_key)
        if not has_anthropic and not has_openai:
            return 1.0  # fallback uniforme, se normalizará después
        tasks = []
        for i, persona in enumerate(personas_selected):
            prompt = self._build_persona_prompt(persona, sub_question, algo_data)
            if has_anthropic and has_openai:
                tasks.append(self._call_persona_anthropic(persona, prompt) if i % 2 == 0 else self._call_persona_openai(persona, prompt))
            elif has_anthropic:
                tasks.append(self._call_persona_anthropic(persona, prompt))
            else:
                tasks.append(self._call_persona_openai(persona, prompt))
        try:
            raw_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            return 1.0
        results = [r for r in raw_results if isinstance(r, PersonaResult)]
        if not results:
            return 1.0
        _, raw_mean = self._bayesian_aggregate(results, market_price)
        return max(0.01, raw_mean)

    async def analyze(self, question: str, algo_data: dict,
                      market_id: str = "",
                      timeout_seconds: float = 20.0,
                      options: list = None) -> SwarmResult:
        """
        Análisis principal del swarm.
        Lanza hasta 18 personas en paralelo con timeout.
        Si options no está vacía y el market_type es "tournament", evalúa cada opción.
        Modo torneo — Carlos David Donoso Cordero (ddchack)
        """
        self._ensure_client()
        start_ms = time.time()

        # Detectar horizonte temporal y seleccionar personas apropiadas
        timeframe = detect_timeframe(question, market_id)
        category = algo_data.get("category", "other")
        market_price = float(algo_data.get("yes_price", 0.5))

        tf_names = TIMEFRAME_PERSONAS.get(timeframe)
        if tf_names is not None:
            personas_selected = _select_personas_by_names(tf_names)
            if not personas_selected:
                personas_selected = _select_personas(category, max_n=18)
        else:
            personas_selected = _select_personas(category, max_n=18)

        # Determinar qué API usar
        has_anthropic = bool(self.anthropic_key)
        has_openai = bool(self.openai_key)

        if not has_anthropic and not has_openai:
            # Sin API disponible — devolver resultado base
            return SwarmResult(
                probability=market_price,
                market_price=market_price,
                confidence_interval=(market_price - 0.1, market_price + 0.1),
                dissent_ratio=0.0,
                top_personas=[],
                dissenters=[],
                consensus_personas=[],
                reasoning_summary="No API keys configured for SwarmEngine.",
                swarm_mean=market_price,
                kelly_fraction=0.0,
                n_personas_used=0,
                latency_ms=0,
                timeframe=timeframe,
            )

        # Construir tareas asíncronas — Carlos David Donoso Cordero (ddchack)
        # Distribuir ~50/50 entre Anthropic y OpenAI cuando ambos están disponibles
        tasks = []
        for i, persona in enumerate(personas_selected):
            prompt = self._build_persona_prompt(persona, question, algo_data, timeframe=timeframe)
            if has_anthropic and has_openai:
                # Índices pares → Claude (Anthropic), impares → GPT (OpenAI)
                if i % 2 == 0:
                    tasks.append(self._call_persona_anthropic(persona, prompt))
                else:
                    tasks.append(self._call_persona_openai(persona, prompt))
            elif has_anthropic:
                tasks.append(self._call_persona_anthropic(persona, prompt))
            else:
                tasks.append(self._call_persona_openai(persona, prompt))

        # Ejecutar en paralelo con timeout
        try:
            raw_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=False),
                timeout=timeout_seconds,
            )
            results: list[PersonaResult] = list(raw_results)
        except asyncio.TimeoutError:
            logger.warning("SwarmEngine timeout — using partial results")
            # Devolver las que terminaron — fallback al gather con timeout
            done_tasks = []
            for t in tasks:
                try:
                    r = await asyncio.wait_for(asyncio.shield(t), timeout=0.01)
                    done_tasks.append(r)
                except Exception:
                    pass
            results = done_tasks if done_tasks else []

        if not results:
            return SwarmResult(
                probability=market_price,
                market_price=market_price,
                confidence_interval=(market_price - 0.1, market_price + 0.1),
                dissent_ratio=0.0,
                top_personas=[],
                dissenters=[],
                consensus_personas=[],
                reasoning_summary="No personas returned results.",
                swarm_mean=market_price,
                kelly_fraction=0.0,
                n_personas_used=0,
                latency_ms=int((time.time() - start_ms) * 1000),
                timeframe=timeframe,
            )

        # Aggregation
        posterior, swarm_mean = self._bayesian_aggregate(results, market_price)

        # Estadísticas de distribución
        probs = [r.probability for r in results]
        n = len(probs)
        mean = sum(probs) / n
        variance = sum((p - mean) ** 2 for p in probs) / max(1, n - 1)
        std = variance ** 0.5

        # Dissent ratio: std normalizada (0.25 = máximo teórico para distribución uniforme)
        dissent_ratio = round(min(1.0, std / 0.25), 3)

        # Confidence interval (p5, p95 aproximado)
        sorted_probs = sorted(probs)
        p5 = sorted_probs[max(0, int(n * 0.05))]
        p95 = sorted_probs[min(n - 1, int(n * 0.95))]

        # Top personas por peso efectivo
        def effective_weight(r: PersonaResult) -> float:
            cw = {"high": 1.0, "medium": 0.6, "low": 0.2}
            return r.weight * cw.get(r.confidence, 0.6) / max(0.001, r.brier_ema)

        sorted_by_weight = sorted(results, key=effective_weight, reverse=True)
        top_personas = [
            {
                "name": r.name,
                "archetype": r.archetype,
                "probability": r.probability,
                "confidence": r.confidence,
                "brier_ema": r.brier_ema,
                "weight": round(r.weight, 3),
                "reasoning": r.reasoning,
            }
            for r in sorted_by_weight[:5]
        ]

        # Dissenters: más alejados de la media
        sorted_by_dissent = sorted(results, key=lambda r: abs(r.probability - mean), reverse=True)
        dissenters = [
            {"name": r.name, "archetype": r.archetype, "probability": r.probability,
             "deviation": round(abs(r.probability - mean), 3)}
            for r in sorted_by_dissent[:3]
        ]

        # Consensus: más cercanos a la media
        sorted_by_consensus = sorted(results, key=lambda r: abs(r.probability - mean))
        consensus_personas = [
            {"name": r.name, "archetype": r.archetype, "probability": r.probability}
            for r in sorted_by_consensus[:3]
        ]

        # Quarter-Kelly sizing
        edge = posterior - market_price
        if edge > 0:
            kelly_fraction = round((edge / (1.0 - market_price)) * 0.25, 4)
        elif edge < 0:
            kelly_fraction = round((abs(edge) / market_price) * 0.25 * -1, 4)
        else:
            kelly_fraction = 0.0

        # Reasoning summary: top 3 personas
        top3_reasoning = "; ".join(
            f"{r['name']}: {r['reasoning'][:80]}" for r in top_personas[:3]
        )

        # Modo torneo — Carlos David Donoso Cordero (ddchack)
        # Si hay opciones y el mercado es tipo torneo, evaluar cada opción con sub-swarm
        market_type = detect_market_type(question, market_id)
        option_probs: dict = {}
        valid_options = [o.strip() for o in (options or []) if o.strip()]
        if valid_options and market_type == "tournament":
            try:
                # Fix: opciones corren en PARALELO — cada una recibe el timeout completo, no dividido
                per_option_timeout = min(15.0, timeout_seconds * 0.8)
                raw_scores = {}
                option_tasks = [
                    self._analyze_option(opt, question, algo_data, market_id, per_option_timeout)
                    for opt in valid_options
                ]
                raw_values = await asyncio.gather(*option_tasks, return_exceptions=True)
                for opt, val in zip(valid_options, raw_values):
                    raw_scores[opt] = float(val) if isinstance(val, (int, float)) else 1.0
                total = sum(raw_scores.values())
                if total > 0:
                    option_probs = {opt: round(v / total, 4) for opt, v in raw_scores.items()}
                else:
                    n_opts = len(valid_options)
                    option_probs = {opt: round(1.0 / n_opts, 4) for opt in valid_options}
            except Exception as e:
                logger.warning(f"Tournament option analysis error: {e}")
                n_opts = len(valid_options)
                option_probs = {opt: round(1.0 / n_opts, 4) for opt in valid_options}
        elif valid_options:
            # Opciones presentes aunque no detectadas como torneo — correr análisis LLM igual
            # Fix Bug 3: antes devolvía uniforme sin análisis — Carlos David Donoso Cordero (ddchack)
            market_type = "tournament"
            try:
                # Fix: opciones corren en PARALELO — cada una recibe el timeout completo, no dividido
                per_option_timeout = min(15.0, timeout_seconds * 0.8)
                raw_scores = {}
                option_tasks = [
                    self._analyze_option(opt, question, algo_data, market_id, per_option_timeout)
                    for opt in valid_options
                ]
                raw_values = await asyncio.gather(*option_tasks, return_exceptions=True)
                for opt, val in zip(valid_options, raw_values):
                    raw_scores[opt] = float(val) if isinstance(val, (int, float)) else 1.0
                total = sum(raw_scores.values())
                if total > 0:
                    option_probs = {opt: round(v / total, 4) for opt, v in raw_scores.items()}
                else:
                    n_opts = len(valid_options)
                    option_probs = {opt: round(1.0 / n_opts, 4) for opt in valid_options}
            except Exception as e:
                logger.warning(f"Tournament option analysis (non-kw) error: {e}")
                n_opts = len(valid_options)
                option_probs = {opt: round(1.0 / n_opts, 4) for opt in valid_options}

        # Garantizar coherencia: si no hay option_probs, forzar modo binario — Carlos David Donoso Cordero (ddchack)
        if not option_probs:
            market_type = "binary"

        return SwarmResult(
            probability=posterior,
            market_price=market_price,
            confidence_interval=(round(p5, 3), round(p95, 3)),
            dissent_ratio=dissent_ratio,
            top_personas=top_personas,
            dissenters=dissenters,
            consensus_personas=consensus_personas,
            reasoning_summary=top3_reasoning,
            swarm_mean=swarm_mean,
            kelly_fraction=kelly_fraction,
            n_personas_used=n,
            latency_ms=int((time.time() - start_ms) * 1000),
            timeframe=timeframe,
            option_probs=option_probs,
            market_type=market_type,
        )

    def swarm_result_to_dict(self, sr: SwarmResult) -> dict:
        return {
            "probability": sr.probability,
            "market_price": sr.market_price,
            "swarm_mean": sr.swarm_mean,
            "confidence_interval": list(sr.confidence_interval),
            "dissent_ratio": sr.dissent_ratio,
            "top_personas": sr.top_personas,
            "dissenters": sr.dissenters,
            "consensus_personas": sr.consensus_personas,
            "reasoning_summary": sr.reasoning_summary,
            "kelly_fraction": sr.kelly_fraction,
            "n_personas_used": sr.n_personas_used,
            "latency_ms": sr.latency_ms,
            "timeframe": sr.timeframe,
            "option_probs": sr.option_probs,
            "market_type": sr.market_type,
        }

    def get_all_personas_status(self) -> list:
        """Devuelve estado completo de todas las 50 personas para el dashboard."""
        out = []
        for p in PERSONAS:
            name = p["name"]
            out.append({
                "name": name,
                "archetype": p["archetype"],
                "categories": p.get("categories", ["all"]),
                "weight": round(self._weight_state.get(name, p["weight"]), 3),
                "brier_ema": round(self._brier_state.get(name, p["brier_ema"]), 4),
                "confidence_level": p["confidence_level"],
                "last_probability": None,  # se rellena tras analyze()
            })
        return out

    async def close(self):
        if self.client:
            await self.client.aclose()
            self.client = None
