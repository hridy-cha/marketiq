"""
News sentiment.

Two analyzers share one interface:

- `LexiconSentimentAnalyzer` (default, offline). A compact finance-oriented word list in the spirit
  of Loughran-McDonald, with simple negation handling. It is deliberately small and transparent;
  it is not the full LM dictionary and should be read as a coarse signal.
- `OpenAISentimentAnalyzer` (optional). Adapted from FinRL-X `src/data/data_fetcher.py`
  (Apache 2.0): asks a chat model for a JSON label and confidence. Enabled only when OPENAI_API_KEY
  is set.

`aggregate_sentiment` turns per-article scores into one ticker-level score in [-1, 1] with
exponential recency decay, and reports how many articles contributed.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    label: str  # positive | neutral | negative
    score: float  # -1 .. 1
    model: str


class SentimentAnalyzer(Protocol):
    def analyze(self, text: str) -> SentimentResult: ...


POSITIVE = {
    "beat", "beats", "beating", "exceed", "exceeds", "exceeded", "strong", "stronger", "strongest", "growth", "grow",
    "grew", "record", "surge", "surges", "surged", "soar", "soars", "soared", "rally", "rallies", "rallied", "gain",
    "gains", "gained", "profit", "profitable", "profits", "upgrade", "upgraded", "upgrades", "outperform",
    "outperforms", "outperformed", "bullish", "optimistic", "optimism", "momentum", "innovative", "innovation",
    "expand", "expands", "expansion", "raise", "raised", "raises", "boost", "boosts", "boosted", "improve",
    "improves", "improved", "improvement", "robust", "solid", "accelerate", "accelerates", "accelerating",
    "demand", "opportunity", "opportunities", "buyback", "dividend", "approval", "approved", "win", "wins", "won",
    "partnership", "milestone", "breakthrough", "recovery", "rebound", "rebounds", "rebounded", "positive",
    "success", "successful", "leadership", "upside", "resilient", "high", "higher", "highs",
}
NEGATIVE = {
    "miss", "misses", "missed", "weak", "weaker", "weakest", "weakness", "decline", "declines", "declined",
    "drop", "drops", "dropped", "fall", "falls", "fell", "plunge", "plunges", "plunged", "slump", "slumps",
    "slumped", "loss", "losses", "lose", "loses", "lost", "downgrade", "downgraded", "downgrades",
    "underperform", "underperforms", "underperformed", "bearish", "pessimistic", "pessimism", "concern",
    "concerns", "concerned", "risk", "risks", "risky", "lawsuit", "lawsuits", "litigation", "investigation",
    "investigates", "probe", "recall", "recalls", "layoff", "layoffs", "cut", "cuts", "cutting", "warning",
    "warns", "warned", "delay", "delays", "delayed", "disappoint", "disappoints", "disappointed", "disappointing",
    "fraud", "penalty", "fine", "fined", "default", "bankruptcy", "bankrupt", "debt", "slowdown", "slowing",
    "headwind", "headwinds", "volatile", "volatility", "uncertain", "uncertainty", "negative", "fail", "fails",
    "failed", "failure", "shortfall", "low", "lower", "lows", "sell-off", "selloff", "tariff", "tariffs",
    "inflation", "recession", "crash", "crashes", "crashed",
}
NEGATORS = {"not", "no", "never", "without", "hardly", "barely", "isn't", "wasn't", "doesn't", "didn't", "won't", "cannot", "can't"}

_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z\-']+")


class LexiconSentimentAnalyzer:
    name = "lexicon-v1"

    def analyze(self, text: str) -> SentimentResult:
        tokens = [t.lower() for t in _TOKEN.findall(text or "")]
        if not tokens:
            return SentimentResult("neutral", 0.0, self.name)
        pos = neg = 0
        for i, tok in enumerate(tokens):
            negated = any(t in NEGATORS for t in tokens[max(0, i - 3) : i])
            if tok in POSITIVE:
                neg += 1 if negated else 0
                pos += 0 if negated else 1
            elif tok in NEGATIVE:
                pos += 1 if negated else 0
                neg += 0 if negated else 1
        hits = pos + neg
        if hits == 0:
            return SentimentResult("neutral", 0.0, self.name)
        # Net polarity, damped by document length so one word in a long article is not a strong signal.
        raw = (pos - neg) / hits
        strength = min(1.0, hits / max(4.0, len(tokens) / 25.0))
        score = raw * strength
        label = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
        return SentimentResult(label, round(score, 4), self.name)


class OpenAISentimentAnalyzer:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", timeout: int = 30):
        from openai import OpenAI  # imported lazily so the dependency stays optional

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.timeout = timeout
        self.name = f"openai:{model}"
        self._fallback = LexiconSentimentAnalyzer()

    def analyze(self, text: str) -> SentimentResult:
        prompt = (
            "Classify the overall sentiment of this financial news for the company's stock as positive, "
            "neutral or negative. Reply with JSON only, e.g. {\"sentiment\": \"neutral\", \"confidence\": 0.65}.\n\n"
            f"{(text or '')[:1500]}"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial news sentiment classifier. Output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=60,
                timeout=self.timeout,
            )
            content = (resp.choices[0].message.content or "").strip()
            parsed = json.loads(content)
            label = str(parsed.get("sentiment", "neutral")).lower()
            conf = float(parsed.get("confidence", 0.5))
            if label not in {"positive", "neutral", "negative"}:
                raise ValueError(label)
            score = conf if label == "positive" else -conf if label == "negative" else 0.0
            return SentimentResult(label, round(score, 4), self.name)
        except Exception as exc:
            logger.warning("OpenAI sentiment failed, using lexicon: %s", exc)
            return self._fallback.analyze(text)


def default_analyzer() -> SentimentAnalyzer:
    from marketiq.config.settings import get_settings

    s = get_settings().openai
    if s.api_key:
        try:
            return OpenAISentimentAnalyzer(s.api_key.get_secret_value(), s.model, s.request_timeout)
        except Exception as exc:  # openai not installed, etc.
            logger.warning("OpenAI analyzer unavailable (%s); using lexicon", exc)
    return LexiconSentimentAnalyzer()


@dataclass
class AggregateSentiment:
    score: Optional[float]  # -1 .. 1, None when below min_articles
    article_count: int
    positive: int
    negative: int
    neutral: int
    available: bool
    reason: str


def aggregate_sentiment(
    articles: Iterable, as_of: datetime, half_life_days: float = 5.0, min_articles: int = 2
) -> AggregateSentiment:
    """Recency-weighted mean of article scores. `articles` have .published_at and .sentiment_score."""
    weights: List[float] = []
    scores: List[float] = []
    pos = neg = neu = 0
    for a in articles:
        if a.sentiment_score is None:
            continue
        age_days = max(0.0, (as_of - a.published_at).total_seconds() / 86400.0)
        w = math.pow(0.5, age_days / half_life_days) if half_life_days > 0 else 1.0
        weights.append(w)
        scores.append(float(a.sentiment_score))
        if a.sentiment_label == "positive":
            pos += 1
        elif a.sentiment_label == "negative":
            neg += 1
        else:
            neu += 1
    n = len(scores)
    if n < min_articles:
        return AggregateSentiment(None, n, pos, neg, neu, False, f"Only {n} scored article(s); need {min_articles}")
    total_w = sum(weights)
    score = sum(w * s for w, s in zip(weights, scores)) / total_w if total_w else 0.0
    return AggregateSentiment(round(score, 4), n, pos, neg, neu, True,
                              f"{n} articles: {pos} positive, {neg} negative, {neu} neutral (recency weighted)")
