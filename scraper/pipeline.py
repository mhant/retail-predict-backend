"""
Scraper pipeline — runs in GitHub Actions twice daily.

Flow:
  1. Bluesky / Mastodon → raw_mentions (with VADER scores)
  3. Ticker sentiment → ticker_sentiment_summary + hype_signals
  5. yFinance         → price_snapshots (OHLCV + RSI/MACD/BB indicators)
                      + institutional_data + ticker_metadata
  6. XGBoost          → model_predictions (technical + institutional features)
  7. CNBC / SA RSS   → news_articles
  8. SEC EDGAR        → insider_trades
  9. Record           → pipeline_runs + scraper_events
"""
from __future__ import annotations

import logging
import math
import os
import re
import sys
import time
import warnings
import calendar
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import feedparser
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from scraper import config
from scraper import d1_client as d1

# Suppress yfinance's noisy stderr warnings — we log them to D1 instead
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")

# ── VADER setup ────────────────────────────────────────────────────────────────
_vader = SentimentIntensityAnalyzer()
_vader.lexicon.update(config.FINANCE_LEXICON)

# ── HTTP session ──────────────────────────────────────────────────────────────
_BOT_UA    = "retail-predict-bot/1.0 (+https://github.com/mhant/retail-predict-test)"
_EDGAR_UA  = "retail-predict-research contact@retail-predict-research.com"
_session   = requests.Session()
_session.headers["User-Agent"] = _BOT_UA

# ── Ticker extraction ──────────────────────────────────────────────────────────
_CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")
_CAPS_RE    = re.compile(r"\b([A-Z]{2,5})\b")


def extract_tickers(text: str) -> list[str]:
    """
    Three-stage ticker extraction:
      1. Cashtag regex ($GME) — highest precision, always trusted
      2. ALL-CAPS words filtered against watchlist — moderate precision
      3. Company name lookup (case-insensitive) gated by financial context words —
         catches "tesla stock" / "gamestop shares" without matching "I ate an apple"
    """
    if not text:
        return []
    found: dict[str, int] = {}

    # Stage 1: cashtags
    for m in _CASHTAG_RE.finditer(text):
        sym = m.group(1)
        if sym not in found:
            found[sym] = m.start()

    # Stage 2: ALL-CAPS watchlist words
    for m in _CAPS_RE.finditer(text):
        sym = m.group(1)
        if sym in found or sym in config.STOPWORDS:
            continue
        if sym in config.WATCHLIST:
            found[sym] = m.start()

    # Stage 3: company name matching — only when financial context is present
    lower = text.lower()
    if any(w in lower for w in config.FINANCIAL_CONTEXT):
        for name, ticker in config.NAME_TO_TICKER.items():
            if ticker in found:
                continue
            pos = lower.find(name)
            if pos == -1:
                continue
            # Whole-word check: character before and after must not be alpha
            before_ok = (pos == 0 or not lower[pos - 1].isalpha())
            after_ok  = (pos + len(name) >= len(lower) or not lower[pos + len(name)].isalpha())
            if before_ok and after_ok:
                found[ticker] = pos

    return [s for s, _ in sorted(found.items(), key=lambda x: x[1])]


def _bsky_auth() -> str | None:
    """Exchange Bluesky app-password for a bearer token. Returns None if not configured."""
    identifier = os.environ.get('BSKY_IDENTIFIER', '').strip()
    password   = os.environ.get('BSKY_APP_PASSWORD', '').strip()
    if not identifier or not password:
        print(f"  [bluesky] missing secrets (BSKY_IDENTIFIER={'set' if identifier else 'MISSING'}, "
              f"BSKY_APP_PASSWORD={'set' if password else 'MISSING'}) — skipping")
        return None
    print(f"  [bluesky] authenticating as {identifier}")
    try:
        resp = _session.post(
            'https://bsky.social/xrpc/com.atproto.server.createSession',
            json={'identifier': identifier, 'password': password},
            timeout=15,
        )
        if not resp.ok:
            print(f"  [bluesky] auth failed {resp.status_code}: {resp.text[:200]}")
            return None
        token = resp.json().get('accessJwt')
        print(f"  [bluesky] auth ok")
        return token
    except Exception as exc:
        print(f"  [bluesky] auth error: {exc}")
        return None


# ── Ticker sentiment summary ───────────────────────────────────────────────────

def _parse_iso(s: str, default: float) -> float:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return default


def news_to_mentions(news_rows: list[dict]) -> list[dict]:
    """
    Convert news articles into raw_mention rows so they feed into sentiment.
    yfinance news already has a ticker; RSS rows use whatever extract_tickers found.
    Score is set to 5 so news gets a small but non-zero engagement weight.
    """
    rows: list[dict] = []
    now  = time.time()
    seen: set[tuple] = set()
    for article in news_rows:
        ticker = article.get("ticker")
        if not ticker:
            continue
        source_id = str(article.get("article_id") or article.get("url") or "")[:200]
        key = (source_id, ticker)
        if key in seen:
            continue
        seen.add(key)
        text   = f"{article.get('title', '')} {article.get('summary', '')}".strip()
        scores = _vader.polarity_scores(text[:2000])
        rows.append({
            "source":         f"news_{article.get('source', 'rss')}",
            "source_id":      source_id,
            "subreddit":      article.get("source", "news"),
            "ticker":         ticker,
            "title":          article.get("title", "")[:500],
            "selftext":       article.get("summary", "")[:2000],
            "author":         "",
            "score":          5,
            "ups":            0,
            "upvote_ratio":   0.5,
            "num_comments":   0,
            "url":            article.get("url", "")[:500],
            "created_utc":    article.get("published_utc", now),
            "scraped_utc":    now,
            "vader_compound": scores["compound"],
            "vader_positive": scores["pos"],
            "vader_negative": scores["neg"],
            "vader_neutral":  scores["neu"],
        })
    return rows


_MASTODON_TAGS = ["investing", "stocks", "wallstreetbets", "stockmarket", "options"]


def fetch_mastodon() -> list[dict]:
    """Fetch finance hashtag timelines from Mastodon (public API, no auth)."""
    rows: list[dict] = []
    now  = time.time()
    seen: set[str] = set()

    for tag in _MASTODON_TAGS:
        try:
            url  = f"https://mastodon.social/api/v1/timelines/tag/{tag}?limit=40"
            resp = _session.get(url, timeout=15, headers={"User-Agent": _BOT_UA})
            resp.raise_for_status()
            for status in resp.json():
                sid = str(status.get("id", ""))
                if sid in seen:
                    continue
                seen.add(sid)
                text = re.sub(r"<[^>]+>", " ", status.get("content", "")).strip()
                if not text:
                    continue
                tickers = extract_tickers(text)
                if not tickers:
                    continue
                scores  = _vader.polarity_scores(text[:2000])
                created = _parse_iso(status.get("created_at", ""), now)
                acct    = (status.get("account") or {}).get("username", "")
                for ticker in tickers:
                    rows.append({
                        "source":         "mastodon",
                        "source_id":      sid,
                        "subreddit":      f"mastodon_{tag}",
                        "ticker":         ticker,
                        "title":          text[:500],
                        "selftext":       "",
                        "author":         acct,
                        "score":          status.get("favourites_count", 0),
                        "ups":            0,
                        "upvote_ratio":   0.5,
                        "num_comments":   status.get("replies_count", 0),
                        "url":            status.get("url", ""),
                        "created_utc":    created,
                        "scraped_utc":    now,
                        "vader_compound": scores["compound"],
                        "vader_positive": scores["pos"],
                        "vader_negative": scores["neg"],
                        "vader_neutral":  scores["neu"],
                    })
            time.sleep(1.0)
        except Exception as exc:
            print(f"  [mastodon] {tag}: {exc}")

    print(f"  [mastodon] {len(rows)} posts")
    return rows


def fetch_bluesky(tickers: list[str]) -> list[dict]:
    """Search Bluesky for cashtag mentions. Requires BSKY_IDENTIFIER + BSKY_APP_PASSWORD."""
    token = _bsky_auth()
    if not token:
        print("  [bluesky] skipping — no auth token")
        return []
    headers = {'User-Agent': _BOT_UA, 'Authorization': f'Bearer {token}'}

    rows: list[dict] = []
    now  = time.time()
    seen: set[str] = set()

    for ticker in tickers[:30]:
        try:
            url  = f"https://bsky.social/xrpc/app.bsky.feed.searchPosts?q=%24{ticker}&limit=25"
            resp = _session.get(url, timeout=15, headers=headers)
            resp.raise_for_status()
            for post in resp.json().get("posts", []):
                uri = post.get("uri", "")
                if uri in seen:
                    continue
                seen.add(uri)
                record  = post.get("record", {})
                text    = record.get("text", "")
                if not text:
                    continue
                author  = post.get("author", {})
                handle  = author.get("handle", "")
                post_id = uri.split("/")[-1]
                created = _parse_iso(record.get("createdAt", ""), now)
                scores  = _vader.polarity_scores(text[:2000])
                rows.append({
                    "source":         "bluesky",
                    "source_id":      uri[:200],
                    "subreddit":      "bluesky",
                    "ticker":         ticker,
                    "title":          text[:500],
                    "selftext":       "",
                    "author":         handle,
                    "score":          post.get("likeCount", 0),
                    "ups":            0,
                    "upvote_ratio":   0.5,
                    "num_comments":   post.get("replyCount", 0),
                    "url":            f"https://bsky.app/profile/{handle}/post/{post_id}",
                    "created_utc":    created,
                    "scraped_utc":    now,
                    "vader_compound": scores["compound"],
                    "vader_positive": scores["pos"],
                    "vader_negative": scores["neg"],
                    "vader_neutral":  scores["neu"],
                })
            time.sleep(0.3)
        except Exception as exc:
            print(f"  [bluesky] {ticker}: {exc}")

    print(f"  [bluesky] {len(rows)} posts across {len(tickers[:30])} tickers")
    return rows


def ingest_user_tips(all_mentions: list[dict]) -> None:
    """Fetch approved user tips and add them to the mention pool with low trust weight."""
    tips = d1.fetch_user_tips()
    if not tips:
        print("  [tips] 0 user tips")
        return
    for tip in tips:
        scored = _vader.polarity_scores(tip["body"][:2000])
        all_mentions.append({
            "source":         "user_tip",
            "source_id":      f"tip_{tip['id']}",
            "subreddit":      "user_tip",
            "ticker":         tip["ticker"],
            "title":          tip["body"][:500],
            "selftext":       "",
            "author":         "anonymous",
            "score":          1,
            "ups":            0,
            "upvote_ratio":   0.5,
            "num_comments":   0,
            "url":            "",
            "created_utc":    tip["submitted_at"],
            "scraped_utc":    time.time(),
            "vader_compound": scored["compound"],
            "vader_positive": scored["pos"],
            "vader_negative": scored["neg"],
            "vader_neutral":  scored["neu"],
        })
    print(f"  [tips] {len(tips)} user tips added to mention pool")


def compute_hype_signals(mentions: list[dict]) -> list[dict]:
    """Compute hype buy/sell signals from sentiment data per ticker."""
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for m in mentions:
        by_ticker[m["ticker"]].append(m)

    signals: list[dict] = []
    now = time.time()

    for ticker, rows in by_ticker.items():
        sentiments = [r["vader_compound"] for r in rows if r.get("vader_compound") is not None]
        if not sentiments:
            continue
        scores  = [r["score"] for r in rows]
        weights = [math.log1p(max(s, 0)) for s in scores]
        total_w = sum(weights) or 1.0
        wt_sent = sum(s * w for s, w in zip(sentiments, weights)) / total_w
        avg_s   = sum(sentiments) / len(sentiments)
        abs_s   = abs(avg_s)

        if abs_s > 0.5:   sig = "strong_buy"   if avg_s > 0 else "strong_sell"
        elif abs_s > 0.15: sig = "buy"          if avg_s > 0 else "sell"
        else:              sig = "neutral"

        signals.append({
            "ticker":                    ticker,
            "computed_at":               now,
            "window_hours":              24,
            "signal":                    sig,
            "avg_sentiment":             avg_s,
            "upvote_weighted_sentiment": wt_sent,
            "mention_count":             len(rows),
            "mention_velocity":          len(rows) / 24.0,   # mentions per hour
            "source_count":              len({r["subreddit"] for r in rows}),
        })

    return signals


# ── XGBoost model inference ────────────────────────────────────────────────────

_MODELS_DIR   = Path(__file__).resolve().parents[1] / "models"
_FEATURE_COLS = [
    "mention_count_1h", "mention_count_4h", "mention_count_24h",
    "mention_velocity_zscore", "vader_compound_mean_1h", "vader_compound_mean_24h",
    "vader_compound_std_1h", "upvote_weighted_sentiment", "bull_bear_ratio_1h",
    "finbert_compound_mean", "source_diversity_score", "stocktwits_bull_ratio",
    "sentiment_momentum", "retail_pressure_score",
    "institutional_ownership_pct", "short_interest_pct", "short_ratio",
    "put_call_ratio", "insider_net_signal", "counter_pressure_score", "squeeze_candidate",
    "rsi_14", "macd_histogram", "bb_position", "volume_ratio_20d",
    "price_momentum_1d", "price_momentum_5d", "atr_normalized",
]
_HORIZONS = {"intraday": "xgb_intraday.json", "short": "xgb_short.json", "medium": "xgb_medium.json"}
_SIGNAL_MAP = {
    (True, 0.75): "strong_buy",  (True, 0.60): "buy",
    (False, 0.25): "strong_sell", (False, 0.40): "sell",
}


def _classify(prob_up: float) -> str:
    if prob_up >= 0.75: return "strong_buy"
    if prob_up >= 0.60: return "buy"
    if prob_up <= 0.25: return "strong_sell"
    if prob_up <= 0.40: return "sell"
    return "neutral"


def run_predictions(
    price_rows:   list[dict],
    inst_rows:    list[dict],
    hype_signals: list[dict],
) -> list[dict]:
    """
    Run XGBoost inference across all historical price bars and blend with hype signals.

    For each ticker:
    - ALL historical bars get model-only predictions (backfill, hype_weight=0)
    - The LATEST bar gets a 70/30 blend of model + hype when hype data exists
    predicted_at = bar timestamp so dots align with actual price dates.
    UNIQUE(ticker, horizon, predicted_at) means INSERT OR IGNORE deduplicates reruns.
    """
    if not price_rows:
        return []

    # Normalise hype avg_sentiment (-1..+1) to probability (0..1)
    hype_map: dict[str, float] = {}
    for h in hype_signals:
        s = h.get("avg_sentiment") or 0.0
        hype_map[h["ticker"]] = (float(s) + 1.0) / 2.0

    # Group ALL bars by ticker, sorted oldest → newest
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for row in price_rows:
        by_ticker[row["ticker"]].append(row)
    for bars in by_ticker.values():
        bars.sort(key=lambda r: r["ts"])

    inst_map = {r["ticker"]: r for r in inst_rows}
    predictions: list[dict] = []

    for horizon, model_file in _HORIZONS.items():
        model_path = _MODELS_DIR / model_file
        if not model_path.exists():
            print(f"  [xgb] {horizon} model not found at {model_path}")
            continue
        try:
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
        except Exception as exc:
            print(f"  [xgb] failed to load {horizon}: {exc}")
            continue

        all_fvs: list[dict]  = []
        meta:    list[tuple] = []   # (ticker, bar_ts, is_current)

        for ticker, bars in by_ticker.items():
            inst    = inst_map.get(ticker, {})
            last_ts = bars[-1]["ts"]
            for price in bars:
                is_current = (price["ts"] == last_ts)
                fv = {col: np.nan for col in _FEATURE_COLS}
                fv["rsi_14"]            = price.get("rsi_14")
                fv["macd_histogram"]    = price.get("macd_histogram")
                fv["bb_position"]       = price.get("bb_position")
                fv["volume_ratio_20d"]  = price.get("volume_ratio_20d")
                fv["price_momentum_1d"] = price.get("price_momentum_1d")
                fv["price_momentum_5d"] = price.get("price_momentum_5d")
                fv["atr_normalized"]    = price.get("atr_normalized")
                if is_current:
                    fv["short_interest_pct"]          = inst.get("short_interest_pct")
                    fv["short_ratio"]                 = inst.get("short_ratio")
                    fv["institutional_ownership_pct"] = inst.get("institutional_ownership_pct")
                all_fvs.append(fv)
                meta.append((ticker, price["ts"], is_current))

        if not all_fvs:
            continue

        try:
            X     = pd.DataFrame(all_fvs, columns=_FEATURE_COLS).astype(float)
            proba = model.predict_proba(X)[:, 1]

            for (ticker, bar_ts, is_current), model_prob in zip(meta, proba):
                model_prob = float(model_prob)
                if is_current and ticker in hype_map:
                    hype_prob    = hype_map[ticker]
                    blended      = model_prob * 0.7 + hype_prob * 0.3
                    model_weight = 0.7
                    hype_weight  = 0.3
                else:
                    blended      = model_prob
                    model_weight = 1.0
                    hype_weight  = 0.0
                predictions.append({
                    "ticker":             ticker,
                    "horizon":            horizon,
                    "predicted_at":       bar_ts,
                    "signal":             _classify(blended),
                    "probability_up":     blended,
                    "confidence":         abs(blended - 0.5) * 2,
                    "model_weight":       model_weight,
                    "hype_weight":        hype_weight,
                    "model_prediction_id": None,
                    "hype_signal_id":     None,
                })
        except Exception as exc:
            print(f"  [xgb] inference error for {horizon}: {exc}")

    print(f"  [xgb] {len(predictions)} predictions across {len(_HORIZONS)} horizons")
    return predictions


def compute_sentiment_summary(mentions: list[dict]) -> list[dict]:
    """Pre-aggregate sentiment per ticker from this batch (24h window)."""
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for m in mentions:
        by_ticker[m["ticker"]].append(m)

    summaries: list[dict] = []
    now = time.time()

    for ticker, rows in by_ticker.items():
        sentiments = [r["vader_compound"] for r in rows if r.get("vader_compound") is not None]
        if not sentiments:
            continue
        scores  = [r["score"] for r in rows]
        ratios  = [r["upvote_ratio"] for r in rows if r.get("upvote_ratio")]

        # Upvote-weighted sentiment: weight each post by log(1 + score)
        import math
        weights  = [math.log1p(max(s, 0)) for s in scores]
        total_w  = sum(weights) or 1.0
        wt_sent  = sum(s * w for s, w in zip(sentiments, weights)) / total_w

        top_post = max(rows, key=lambda r: r["score"])

        summaries.append({
            "ticker":                    ticker,
            "window_hours":              24,
            "computed_at":               now,
            "mention_count":             len(rows),
            "avg_sentiment":             sum(sentiments) / len(sentiments),
            "upvote_weighted_sentiment": wt_sent,
            "avg_upvote_ratio":          sum(ratios) / len(ratios) if ratios else None,
            "source_count":              len({r["subreddit"] for r in rows}),
            "top_title":                 top_post.get("title", "")[:200],
        })

    return summaries


# ── Market data (yFinance) ─────────────────────────────────────────────────────

def _compute_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI(14), MACD, Bollinger Bands(20), volume ratio from OHLCV history."""
    try:
        from ta.momentum import RSIIndicator
        from ta.trend import MACD as MACDIndicator
        from ta.volatility import BollingerBands

        close  = hist["Close"]
        volume = hist["Volume"]

        rsi    = RSIIndicator(close=close, window=14).rsi()
        macd_i = MACDIndicator(close=close)
        bb     = BollingerBands(close=close, window=20)

        hist = hist.copy()
        hist["rsi_14"]           = rsi
        hist["macd"]             = macd_i.macd()
        hist["macd_signal"]      = macd_i.macd_signal()
        hist["macd_histogram"]   = macd_i.macd_diff()
        hist["bb_upper"]         = bb.bollinger_hband()
        hist["bb_lower"]         = bb.bollinger_lband()
        hist["bb_position"]      = bb.bollinger_pband()   # 0–1 within band

        vol_ma20 = volume.rolling(20).mean()
        hist["volume_ratio_20d"] = volume / vol_ma20

        hist["price_momentum_1d"] = close.pct_change(1)
        hist["price_momentum_5d"] = close.pct_change(5)

        atr = (hist["High"] - hist["Low"]).rolling(14).mean()
        hist["atr_normalized"] = atr / close
    except Exception as exc:
        print(f"  [ta] indicator computation error: {exc}")
    return hist


def fetch_market_data(
    tickers: list[str],
    pipeline_started_at: float,
    existing_price_meta: dict[str, dict] | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    Returns (price_snapshot_rows, institutional_data_rows, scraper_event_rows, metadata_rows).
    Fetches 3 months of history so technical indicators have enough periods.
    
    If existing_price_meta is provided:
      - New tickers (not in DB or <15 bars): writes full historical backfill (~70 bars).
      - Existing tickers (>=15 bars in DB): writes only newest candles (ts >= latest_ts),
        cutting price snapshot writes by ~98% while keeping indicators 100% accurate.
    """
    price_rows:    list[dict] = []
    inst_rows:     list[dict] = []
    event_rows:    list[dict] = []
    metadata_rows: list[dict] = []
    today = time.strftime("%Y-%m-%d")
    now   = time.time()
    
    existing_meta = existing_price_meta or {}
    new_tickers_count = 0
    existing_tickers_count = 0

    for ticker in tickers:
        try:
            t    = yf.Ticker(ticker)
            start_date = (date.today() - timedelta(days=100)).strftime("%Y-%m-%d")
            end_date   = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
            hist = t.history(start=start_date, end=end_date, interval="1d", auto_adjust=True)

            if hist.empty:
                event_rows.append({
                    "event_type":          "ticker_not_found",
                    "ticker":              ticker,
                    "detail":              "yfinance returned empty history (possibly delisted)",
                    "pipeline_started_at": pipeline_started_at,
                    "occurred_at":         now,
                })
                continue

            hist = _compute_indicators(hist)

            def _safe(row, col):
                v = row.get(col)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    return None
                return float(v)

            # Check if we already have sufficient history for this ticker in D1
            existing = existing_meta.get(ticker)
            is_new = (
                existing is None
                or existing.get("latest_ts") is None
                or existing.get("bar_count", 0) < 15
            )

            if is_new:
                new_tickers_count += 1
                ticker_bars = hist
            else:
                existing_tickers_count += 1
                latest_ts = float(existing["latest_ts"])
                # Filter to bars at or newer than our latest DB timestamp (or at least the latest candle)
                filtered = hist[hist.index.map(lambda ts: ts.timestamp() if hasattr(ts, 'timestamp') else float(ts)) >= latest_ts]
                ticker_bars = filtered if not filtered.empty else hist.iloc[-1:]

            for ts, row in ticker_bars.iterrows():
                ts_val = ts.timestamp() if hasattr(ts, 'timestamp') else float(ts)
                price_rows.append({
                    "ticker":            ticker,
                    "interval":          "1d",
                    "ts":                ts_val,
                    "open":              _safe(row, "Open"),
                    "high":              _safe(row, "High"),
                    "low":               _safe(row, "Low"),
                    "close":             _safe(row, "Close"),
                    "volume":            int(row["Volume"]) if row.get("Volume") else None,
                    "rsi_14":            _safe(row, "rsi_14"),
                    "macd":              _safe(row, "macd"),
                    "macd_signal":       _safe(row, "macd_signal"),
                    "macd_histogram":    _safe(row, "macd_histogram"),
                    "bb_upper":          _safe(row, "bb_upper"),
                    "bb_lower":          _safe(row, "bb_lower"),
                    "bb_position":       _safe(row, "bb_position"),
                    "volume_ratio_20d":  _safe(row, "volume_ratio_20d"),
                    "price_momentum_1d": _safe(row, "price_momentum_1d"),
                    "price_momentum_5d": _safe(row, "price_momentum_5d"),
                    "atr_normalized":    _safe(row, "atr_normalized"),
                })

            inst_row: dict = {"ticker": ticker, "report_date": today}
            try:
                full_info = t.info
                inst_row.update({
                    "short_interest_pct":          full_info.get("shortPercentOfFloat"),
                    "short_ratio":                 full_info.get("shortRatio"),
                    "institutional_ownership_pct": full_info.get("institutionsPercentHeld"),
                })
                company_name = full_info.get("longName") or full_info.get("shortName") or ""
                if company_name:
                    metadata_rows.append({
                        "ticker":       ticker,
                        "company_name": company_name,
                        "sector":       full_info.get("sector") or "",
                        "industry":     full_info.get("industry") or "",
                        "exchange":     full_info.get("exchange") or "",
                        "quote_type":   full_info.get("quoteType") or "",
                        "updated_at":   int(now),
                    })
            except Exception:
                pass
            inst_rows.append(inst_row)

            time.sleep(0.3)

        except Exception as exc:
            detail = str(exc)[:300]
            event_type = "delisted" if "delisted" in detail.lower() else "parse_error"
            event_rows.append({
                "event_type":          event_type,
                "ticker":              ticker,
                "detail":              detail,
                "pipeline_started_at": pipeline_started_at,
                "occurred_at":         now,
            })

    print(f"  [yfinance] {len(price_rows)} price rows ({new_tickers_count} backfilled, {existing_tickers_count} incremental), {len(inst_rows)} ok, {len(event_rows)} skipped, {len(metadata_rows)} metadata")
    if event_rows:
        print(f"  [yfinance] skipped: {[e['ticker'] for e in event_rows]}")

    return price_rows, inst_rows, event_rows, metadata_rows


def fetch_yfinance_news(tickers: list[str]) -> list[dict]:
    """Fetch per-ticker news articles from yfinance (10 articles per tracked ticker)."""
    rows: list[dict] = []
    now  = time.time()
    seen: set[str] = set()

    for ticker in tickers:
        try:
            news_items = yf.Ticker(ticker).news or []
            for item in news_items:
                content    = item.get("content", {})
                article_id = content.get("id") or item.get("id", "")
                if not article_id or article_id in seen:
                    continue
                seen.add(article_id)
                title      = content.get("title", "")
                url        = (content.get("canonicalUrl") or {}).get("url", "")
                pub_str    = content.get("pubDate", "")
                published  = _parse_iso(pub_str, now) if pub_str else now
                tickers_in = [ticker] + [
                    (s.get("symbol") or "") for s in (content.get("relatedTickers") or [])
                ]
                tickers_in = [t for t in tickers_in if t]
                rows.append({
                    "source":        "yfinance_news",
                    "article_id":    str(article_id)[:200],
                    "ticker":        ticker,
                    "title":         title[:400],
                    "summary":       content.get("summary", "")[:500],
                    "url":           url[:500],
                    "published_utc": published,
                    "scraped_utc":   now,
                })
            time.sleep(0.1)
        except Exception as exc:
            print(f"  [yfinance news] {ticker}: {exc}")

    print(f"  [yfinance news] {len(rows)} articles across {len(tickers)} tickers")
    return rows


# ── News RSS (CNBC + Seeking Alpha) ───────────────────────────────────────────

def fetch_news() -> list[dict]:
    """Fetch articles from financial RSS feeds."""
    news_rows: list[dict] = []
    now = time.time()

    sources = [
        ("cnbc",          "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839135"),
        ("cnbc_finance",  "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664"),
        ("seeking_alpha", "https://seekingalpha.com/market_currents.xml"),
        ("seeking_alpha_analysis", "https://seekingalpha.com/feed.xml"),
        ("reuters",       "https://feeds.reuters.com/reuters/businessNews"),
        ("reuters_tech",  "https://feeds.reuters.com/reuters/technologyNews"),
        ("yahoo_finance", "https://finance.yahoo.com/news/rssindex"),
        ("marketwatch",   "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
        ("benzinga",      "https://www.benzinga.com/feed"),
        ("motley_fool",   "https://www.fool.com/feeds/index.aspx"),
    ]

    for source, url in sources:
        try:
            feed = feedparser.parse(url, agent=_BOT_UA, request_headers={"User-Agent": _BOT_UA})
            if not feed.entries:
                continue
            for entry in feed.entries:
                article_id  = getattr(entry, "id", getattr(entry, "link", ""))
                title       = getattr(entry, "title", "")
                tickers     = extract_tickers(title)
                pub         = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
                published   = calendar.timegm(pub) if pub else now
                news_rows.append({
                    "source":        source,
                    "article_id":    article_id[:200],
                    "ticker":        tickers[0] if tickers else None,
                    "title":         title[:400],
                    "summary":       getattr(entry, "summary", "")[:500],
                    "url":           getattr(entry, "link", "")[:500],
                    "published_utc": published,
                    "scraped_utc":   now,
                })
            print(f"  [news] {source}: {len(feed.entries)} articles")
        except Exception as exc:
            print(f"  [news] {source}: {exc}")

    return news_rows


# ── SEC EDGAR Form 4 (insider trades) ─────────────────────────────────────────

def fetch_insider_trades() -> list[dict]:
    """Fetch recent Form 4 filings from SEC EDGAR RSS."""
    rows: list[dict] = []
    now  = time.time()
    url  = (
        "https://www.sec.gov/cgi-bin/browse-edgar"
        "?action=getcurrent&type=4&dateb=&owner=include&count=40&search_text=&output=atom"
    )
    try:
        feed = feedparser.parse(url, agent=_EDGAR_UA, request_headers={"User-Agent": _EDGAR_UA})
        for entry in feed.entries:
            title      = getattr(entry, "title", "")
            summary    = getattr(entry, "summary", "")
            link       = getattr(entry, "link", "")
            filing_id  = getattr(entry, "id", link)

            # Extract filing date from summary ("Filed: YYYY-MM-DD")
            date_match = re.search(r"Filed:</b>\s*(\d{4}-\d{2}-\d{2})", summary)
            filed_date = date_match.group(1) if date_match else None

            rows.append({
                "filing_id":   filing_id[:200],
                "ticker":      None,           # Form 4 RSS doesn't include ticker; enriched later
                "company_name": title[:200],
                "insider_name": None,
                "transaction_type": None,
                "shares":      None,
                "price":       None,
                "filed_date":  filed_date,
                "scraped_utc": now,
            })

        print(f"  [edgar] {len(rows)} Form 4 filings")
    except Exception as exc:
        print(f"  [edgar] {exc}")

    return rows


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run() -> None:
    started_at = time.time()
    stats: dict[str, int] = {k: 0 for k in [
        "subreddits_scraped", "mentions_scraped", "new_mentions",
        "vader_scored", "prices_updated", "predictions_written",
    ]}
    error_msg: str | None = None

    try:
        invalid_tickers = d1.fetch_invalid_tickers()
        historical      = d1.fetch_tracked_tickers()
        seed_tickers    = sorted(historical or config.WATCHLIST)

        print("\n── Scraping social sources ──")
        bsky_mentions  = fetch_bluesky(seed_tickers)
        masto_mentions = fetch_mastodon()
        social_mentions = [
            m for m in (bsky_mentions + masto_mentions)
            if m["ticker"] not in invalid_tickers
        ]

        print("\n── Fetching RSS news (for ticker coverage) ──")
        rss_rows     = fetch_news()
        rss_mentions = [m for m in news_to_mentions(rss_rows) if m["ticker"] not in invalid_tickers]
        print(f"  → {len(rss_mentions)} mention rows from {len(rss_rows)} RSS articles")

        # Build ticker universe: current mentions + historical + WATCHLIST as guaranteed floor
        initial_mentions = social_mentions + rss_mentions
        ticker_counts    = Counter(m["ticker"] for m in initial_mentions)
        current_top      = [t for t, _ in ticker_counts.most_common(config.TOP_TICKERS_FOR_MARKET_DATA)]
        current_set      = set(current_top)
        hist_extra       = [t for t in historical      if t not in current_set]
        watch_extra      = [t for t in sorted(config.WATCHLIST) if t not in current_set and t not in set(hist_extra)]
        top_tickers      = (current_top + hist_extra + watch_extra)[:config.MAX_TICKERS_FOR_MARKET_DATA]

        print("\n── Fetching yFinance news (per tracked ticker) ──")
        yf_news      = fetch_yfinance_news(top_tickers[:80])
        yf_mentions  = [m for m in news_to_mentions(yf_news) if m["ticker"] not in invalid_tickers]
        print(f"  → {len(yf_mentions)} mention rows from {len(yf_news)} yFinance articles")

        # All mentions: social + RSS news + yFinance news
        mentions = initial_mentions + yf_mentions

        print("\n── Ingesting user tips ──")
        ingest_user_tips(mentions)

        stats["subreddits_scraped"] = 0
        stats["mentions_scraped"]   = len(mentions)
        stats["vader_scored"]       = len(mentions)

        result = d1.ingest("raw_mentions", mentions)
        stats["new_mentions"] = result["inserted"]
        print(f"\n  → {result['inserted']} new, {result['skipped']} dupes "
              f"({len(social_mentions)} social + {len(rss_mentions)} RSS + {len(yf_mentions)} yf_news)")

        print("\n── Computing ticker sentiment summary + hype signals ──")
        summaries = compute_sentiment_summary(mentions)
        d1.ingest("ticker_sentiment_summary", summaries, mode="replace")
        print(f"  → {len(summaries)} ticker summaries")

        hype_signals = compute_hype_signals(mentions)
        d1.ingest("hype_signals", hype_signals, mode="replace")
        print(f"  → {len(hype_signals)} hype signals")

        print("\n── Fetching market data (yFinance) ──")
        print(f"  tracking {len(top_tickers)} tickers ({len(current_top)} current + {len(hist_extra)} historical + {len(watch_extra)} watchlist)")
        existing_price_meta = d1.fetch_price_latest_timestamps("1d")
        price_rows, inst_rows, event_rows, metadata_rows = fetch_market_data(
            top_tickers, started_at, existing_price_meta=existing_price_meta
        )

        d1.ingest("price_snapshots", price_rows, mode="replace")
        d1.ingest("institutional_data", [
            r for r in inst_rows
            if any(v for k, v in r.items() if k not in ("ticker", "report_date") and v is not None)
        ], mode="replace")
        if metadata_rows:
            d1.ingest("ticker_metadata", metadata_rows, mode="replace")
        if event_rows:
            d1.ingest("scraper_events", event_rows)
        stats["prices_updated"] = len(price_rows)

        print("\n── Running XGBoost predictions (blended model + hype) ──")
        combined_preds = run_predictions(price_rows, inst_rows, hype_signals)
        if combined_preds:
            d1.ingest("combined_predictions", combined_preds)
            stats["predictions_written"] = len(combined_preds)

        print("\n── Storing news articles ──")
        d1.ingest("news_articles", rss_rows + yf_news)

        print("\n── Fetching insider trades (SEC EDGAR Form 4) ──")
        insider_rows = fetch_insider_trades()
        d1.ingest("insider_trades", insider_rows)

        status = "completed"
        print(f"\n── Run complete in {time.time() - started_at:.1f}s ──")
        print(f"   mentions={stats['mentions_scraped']} new={stats['new_mentions']} prices={stats['prices_updated']}")

    except Exception as exc:
        status    = "failed"
        error_msg = str(exc)
        print(f"\n[PIPELINE ERROR] {exc}", file=sys.stderr)
        raise

    finally:
        d1.ingest("pipeline_runs", [{
            "started_at":          started_at,
            "finished_at":         time.time(),
            "status":              status,
            "triggered_by":        "schedule",
            "subreddits_scraped":  stats["subreddits_scraped"],
            "mentions_scraped":    stats["mentions_scraped"],
            "new_mentions":        stats["new_mentions"],
            "vader_scored":        stats["vader_scored"],
            "prices_updated":      stats["prices_updated"],
            "predictions_written": stats["predictions_written"],
            "error_message":       error_msg,
        }])


if __name__ == "__main__":
    run()
