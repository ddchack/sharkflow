"""
══════════════════════════════════════════════════════════════
Polymarket Bot - News Sentiment Analyzer
Developed by: Carlos David Donoso Cordero (ddchack)
══════════════════════════════════════════════════════════════

Analyzes news headlines to generate sentiment signals
that feed into the math engine for probability estimation.

Uses:
- NewsAPI (free tier) for headlines
- TextBlob for NLP sentiment
- Custom keyword extraction for market matching
"""

import httpx
import asyncio
import re
import xml.etree.ElementTree as _ET
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict

# TextBlob import with fallback
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False


@dataclass
class NewsArticle:
    title: str
    description: str
    source: str
    url: str
    published_at: str
    sentiment: float = 0.0  # -1.0 to 1.0
    relevance: float = 0.0  # 0.0 to 1.0


@dataclass
class SentimentReport:
    keyword: str
    articles_found: int
    avg_sentiment: float
    sentiment_label: str  # "BULLISH", "BEARISH", "NEUTRAL"
    top_articles: list
    confidence: float  # How confident we are in this signal


class NewsSentimentAnalyzer:
    """
    Analyzes external news to generate sentiment scores for markets.
    __signature__ = "ddchack"
    """

    # Free news API endpoints (no key required for some)
    NEWSAPI_URL = "https://newsapi.org/v2/everything"
    GNEWS_URL = "https://gnews.io/api/v4/search"

    # Common prediction market topics and their keywords
    TOPIC_KEYWORDS = {
        "politics": ["election", "president", "congress", "senate", "vote", "poll",
                      "democrat", "republican", "trump", "biden", "governor"],
        "crypto": ["bitcoin", "ethereum", "crypto", "btc", "eth", "blockchain",
                    "defi", "nft", "solana", "binance"],
        "sports": ["nba", "nfl", "soccer", "football", "championship", "playoffs",
                    "world cup", "olympics", "super bowl"],
        "economics": ["fed", "interest rate", "inflation", "gdp", "recession",
                       "unemployment", "stock market", "s&p", "nasdaq"],
        "tech": ["ai", "artificial intelligence", "openai", "google", "apple",
                  "microsoft", "meta", "spacex", "tesla"],
        "geopolitics": ["war", "conflict", "sanctions", "nato", "china", "russia",
                         "ukraine", "middle east", "iran"],
    }

    _CACHE_MAX_SIZE = 200   # Máximo de entradas en cache para evitar memory leak
    _CACHE_TTL_S    = 3600  # TTL en segundos (1 hora)

    def __init__(self, newsapi_key: str = None):
        self.newsapi_key = newsapi_key
        self.client = httpx.AsyncClient(timeout=15.0)
        self._sentiment_cache: dict = {}  # key → {"report": ..., "ts": datetime}

    async def close(self):
        await self.client.aclose()

    # ─────────────────────────────────────────────────────────
    # NEWS FETCHING
    # ─────────────────────────────────────────────────────────

    async def fetch_news_newsapi(self, query: str, days_back: int = 3,
                                  page_size: int = 20) -> list[dict]:
        """Fetch news from NewsAPI.org (requires free API key)."""
        if not self.newsapi_key:
            return []

        from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        params = {
            "q": query,
            "from": from_date,
            "sortBy": "relevancy",
            "pageSize": page_size,
            "language": "en",
            "apiKey": self.newsapi_key,
        }

        try:
            resp = await self.client.get(self.NEWSAPI_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            return data.get("articles", [])
        except Exception as e:
            print(f"[Sentiment] NewsAPI error: {e}")
            return []

    async def fetch_news_free(self, query: str) -> list[dict]:
        """
        Fallback: scrape Google News RSS (no API key needed).
        Returns basic article info.
        """
        try:
            encoded_q = query.replace(" ", "+")
            url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-US&gl=US&ceid=US:en"
            resp = await self.client.get(url, follow_redirects=True)
            resp.raise_for_status()

            # XML parsing para RSS (más robusto que regex)
            articles = []
            try:
                root = _ET.fromstring(resp.text)
                ns = {"media": "http://search.yahoo.com/mrss/"}
                channel = root.find("channel")
                items = (channel or root).findall("item") if (channel or root) is not None else []
                for item in items[:15]:
                    title = (item.findtext("title") or "").strip()
                    link  = (item.findtext("link") or "").strip()
                    pub   = (item.findtext("pubDate") or "").strip()
                    src_el = item.find("source")
                    src   = src_el.text.strip() if src_el is not None and src_el.text else "Unknown"
                    if title:
                        articles.append({
                            "title": title,
                            "description": title,
                            "source": {"name": src},
                            "url": link,
                            "publishedAt": pub,
                        })
            except _ET.ParseError:
                # Fallback a regex si el XML está malformado
                items_re = re.findall(r"<item[^>]*>(.*?)</item>", resp.text, re.DOTALL)
                for item in items_re[:15]:
                    title_m = re.search(r"<title[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", item, re.DOTALL)
                    if title_m:
                        articles.append({
                            "title": title_m.group(1).strip(),
                            "description": title_m.group(1).strip(),
                            "source": {"name": "Unknown"},
                            "url": "", "publishedAt": "",
                        })
            return articles
        except Exception as e:
            print(f"[Sentiment] Free news fetch error: {e}")
            return []

    # ─────────────────────────────────────────────────────────
    # SENTIMENT ANALYSIS
    # ─────────────────────────────────────────────────────────

    def analyze_text_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of a text string.
        Returns float from -1.0 (very negative) to 1.0 (very positive).
        """
        if not text:
            return 0.0

        if HAS_TEXTBLOB:
            blob = TextBlob(text)
            return round(blob.sentiment.polarity, 4)

        # Fallback: simple keyword-based sentiment
        return self._keyword_sentiment(text)

    def _keyword_sentiment(self, text: str) -> float:
        """Simple keyword-based sentiment as fallback."""
        text_lower = text.lower()

        positive_words = [
            "surge", "rise", "gain", "win", "success", "positive", "growth",
            "boost", "strong", "bullish", "approve", "pass", "victory",
            "breakthrough", "record", "soar", "rally", "up", "increase",
            "support", "confident", "optimistic", "agree", "deal"
        ]
        negative_words = [
            "crash", "fall", "loss", "fail", "negative", "decline", "drop",
            "weak", "bearish", "reject", "block", "defeat", "crisis",
            "collapse", "plunge", "down", "decrease", "concern", "risk",
            "fear", "pessimistic", "oppose", "ban", "war"
        ]

        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        return round((pos_count - neg_count) / total, 4)

    # ─────────────────────────────────────────────────────────
    # MARKET-MATCHED SENTIMENT
    # ─────────────────────────────────────────────────────────

    def extract_keywords_from_question(self, question: str) -> list[str]:
        """
        Extract meaningful keywords from a market question for news search.
        Prioritizes proper nouns (capitalized tokens) over generic terms.
        """
        stop_words = {
            "will", "the", "be", "in", "by", "on", "at", "to", "of", "a", "an",
            "is", "are", "was", "were", "has", "have", "had", "do", "does",
            "this", "that", "these", "those", "before", "after", "yes", "no",
            "above", "below", "more", "less", "than", "or", "and", "not",
            "what", "when", "where", "who", "how", "which", "if", "end",
            "market", "prediction", "2024", "2025", "2026", "2027",
            "reach", "exceed", "close", "hit", "pass", "win", "lose",
            "remain", "sign", "accept", "resign", "qualify",
        }

        # Proper nouns: words that start with uppercase in the original question (not sentence start)
        proper_nouns = []
        tokens = re.findall(r'\b[A-Z][a-z]{1,}\b', question)
        for t in tokens:
            if t.lower() not in stop_words and len(t) > 2:
                proper_nouns.append(t)

        # Lowercase keywords (substantive terms longer than 3 chars)
        words = re.findall(r'\b[a-zA-Z]+\b', question.lower())
        generic = [w for w in words if w not in stop_words and len(w) > 3]

        # Merge: proper nouns first, then generic, deduplicate preserving order
        seen = set()
        merged = []
        for kw in (proper_nouns + generic):
            kl = kw.lower()
            if kl not in seen:
                seen.add(kl)
                merged.append(kw)

        return merged[:6]

    async def get_sentiment_for_market(self, question: str) -> SentimentReport:
        """Full sentiment analysis pipeline for a market question."""
        keywords = self.extract_keywords_from_question(question)
        query = " ".join(keywords[:3])  # Use top 3 keywords for search

        # Check cache
        cache_key = query.lower()
        _now = datetime.now(timezone.utc)
        if cache_key in self._sentiment_cache:
            cached = self._sentiment_cache[cache_key]
            if (_now - cached["ts"]).total_seconds() < self._CACHE_TTL_S:
                return cached["report"]
            else:
                del self._sentiment_cache[cache_key]  # Expirado → limpiar

        # Evitar memory leak: limpiar entradas expiradas si cache está lleno
        if len(self._sentiment_cache) >= self._CACHE_MAX_SIZE:
            expired = [k for k, v in self._sentiment_cache.items()
                       if (_now - v["ts"]).total_seconds() >= self._CACHE_TTL_S]
            for k in expired:
                del self._sentiment_cache[k]
            # Si aún está lleno, eliminar la entrada más antigua
            if len(self._sentiment_cache) >= self._CACHE_MAX_SIZE:
                oldest = min(self._sentiment_cache, key=lambda k: self._sentiment_cache[k]["ts"])
                del self._sentiment_cache[oldest]

        # Fetch news
        articles_raw = await self.fetch_news_newsapi(query)
        if not articles_raw:
            articles_raw = await self.fetch_news_free(query)

        # Analyze each article
        articles = []
        sentiments = []
        for art in articles_raw[:15]:
            title = art.get("title", "")
            desc = art.get("description", "") or ""
            text = f"{title}. {desc}"
            sent = self.analyze_text_sentiment(text)
            sentiments.append(sent)

            articles.append(NewsArticle(
                title=title,
                description=desc[:200],
                source=art.get("source", {}).get("name", "Unknown"),
                url=art.get("url", ""),
                published_at=art.get("publishedAt", ""),
                sentiment=sent,
            ))

        # Compute aggregate
        avg_sentiment = sum(sentiments) / max(1, len(sentiments))

        if avg_sentiment > 0.15:
            label = "BULLISH"
        elif avg_sentiment < -0.15:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        # Confidence based on number of articles and agreement
        if len(sentiments) > 0:
            import numpy as np
            std = float(np.std(sentiments)) if len(sentiments) > 1 else 0.5
            article_factor = min(1.0, len(sentiments) / 10.0)
            agreement_factor = max(0, 1.0 - std)
            confidence = round((article_factor * 0.5 + agreement_factor * 0.5) * 100, 1)
        else:
            confidence = 0.0

        report = SentimentReport(
            keyword=query,
            articles_found=len(articles),
            avg_sentiment=round(avg_sentiment, 4),
            sentiment_label=label,
            top_articles=[{
                "title": a.title,
                "source": a.source,
                "sentiment": a.sentiment,
                "url": a.url,
            } for a in sorted(articles, key=lambda x: abs(x.sentiment), reverse=True)[:5]],
            confidence=confidence,
        )

        # Cache it
        self._sentiment_cache[cache_key] = {"report": report, "ts": datetime.now(timezone.utc)}
        return report

    async def batch_sentiment(self, questions: list[str]) -> dict[str, float]:
        """
        Get sentiment scores for multiple market questions.
        Returns dict: {keyword: sentiment_score}
        """
        results = {}
        tasks = [self.get_sentiment_for_market(q) for q in questions[:10]]  # Limit concurrent
        reports = await asyncio.gather(*tasks, return_exceptions=True)

        for q, report in zip(questions[:10], reports):
            if isinstance(report, SentimentReport):
                keywords = self.extract_keywords_from_question(q)
                for kw in keywords:
                    results[kw] = report.avg_sentiment
            else:
                print(f"[Sentiment] Error analyzing '{q[:50]}': {report}")

        return results
