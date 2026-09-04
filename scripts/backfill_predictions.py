"""
Backfill historical model predictions for tickers already tracked by the pipeline.

Uses yfinance + real XGBoost models locally, then writes directly to D1 via
wrangler (no WRITE_TOKEN needed — uses your Cloudflare API token from wrangler).

Usage:
    python scripts/backfill_predictions.py
    python scripts/backfill_predictions.py TSLA NVDA HOOD
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf

CLOUDFLARE_DIR = ROOT / "cloudflare"
MODELS_DIR     = ROOT / "models"

FEATURE_COLS = [
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
HORIZONS = {
    "intraday": "xgb_intraday.json",
    "short":    "xgb_short.json",
    "medium":   "xgb_medium.json",
}


def classify(prob_up: float) -> str:
    if prob_up >= 0.75: return "strong_buy"
    if prob_up >= 0.60: return "buy"
    if prob_up <= 0.25: return "strong_sell"
    if prob_up <= 0.40: return "sell"
    return "neutral"


def compute_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    from ta.momentum import RSIIndicator
    from ta.trend import MACD as MACDIndicator
    from ta.volatility import BollingerBands
    close, volume = hist["Close"], hist["Volume"]
    hist = hist.copy()
    hist["rsi_14"]           = RSIIndicator(close=close, window=14).rsi()
    macd_i                   = MACDIndicator(close=close)
    hist["macd_histogram"]   = macd_i.macd_diff()
    bb                       = BollingerBands(close=close, window=20)
    hist["bb_position"]      = bb.bollinger_pband()
    hist["volume_ratio_20d"] = volume / volume.rolling(20).mean()
    hist["price_momentum_1d"] = close.pct_change(1)
    hist["price_momentum_5d"] = close.pct_change(5)
    hist["atr_normalized"]   = (hist["High"] - hist["Low"]).rolling(14).mean() / close
    return hist


def d1_insert(predictions: list[dict]) -> None:
    import tempfile, os
    batch_size = 20
    for i in range(0, len(predictions), batch_size):
        batch = predictions[i:i + batch_size]
        rows  = ",\n  ".join(
            f"('{p['ticker']}','{p['horizon']}',{p['predicted_at']},"
            f"'{p['signal']}',{p['probability_up']},{p['probability_down']},"
            f"{p['confidence']},{p['feature_ts']})"
            for p in batch
        )
        sql = (
            "INSERT OR IGNORE INTO model_predictions "
            "(ticker,horizon,predicted_at,signal,probability_up,probability_down,confidence,feature_ts) "
            f"VALUES\n  {rows};"
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
            f.write(sql)
            tmp = f.name
        try:
            result = subprocess.run(
                f'npx wrangler d1 execute retail-predict --remote --file "{tmp}"',
                cwd=str(CLOUDFLARE_DIR), capture_output=True, text=True, shell=True,
            )
            if result.returncode != 0:
                print(f"  D1 error: {result.stderr[:200]}")
            else:
                print(f"  batch {i // batch_size + 1}/{-(-len(predictions)//batch_size)} ({len(batch)} rows)", end="\r")
        finally:
            os.unlink(tmp)


def get_tickers_from_db() -> list[str]:
    result = subprocess.run(
        'npx wrangler d1 execute retail-predict --remote --json '
        '--command "SELECT DISTINCT ticker FROM model_predictions ORDER BY ticker"',
        cwd=str(CLOUDFLARE_DIR), capture_output=True, text=True, shell=True,
    )
    try:
        data = json.loads(result.stdout)
        if isinstance(data, list):
            return [row["ticker"] for row in data[0]["results"]]
        return [row["ticker"] for row in data["results"]]
    except Exception as exc:
        print(f"Could not parse D1 response ({exc}), using fallback ticker list")
        return [
            "AAPL", "AMZN", "AMD", "GOOGL", "GME", "HOOD", "INTC",
            "MSTR", "NFLX", "NET", "NVDA", "NCNA", "ONFO", "PACB",
            "PLTR", "QBTS", "SOFI", "SPY", "TSLA", "WWR",
        ]


def backfill(tickers: list[str]) -> None:
    start_date = (date.today() - timedelta(days=100)).strftime("%Y-%m-%d")
    end_date   = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    cutoff_ts  = time.time() - 3 * 86400

    models: dict[str, xgb.XGBClassifier] = {}
    for horizon, fname in HORIZONS.items():
        path = MODELS_DIR / fname
        if path.exists():
            m = xgb.XGBClassifier()
            m.load_model(str(path))
            models[horizon] = m
            print(f"  loaded {horizon}")
        else:
            print(f"  missing {path}")

    if not models:
        sys.exit("No models found.")

    all_predictions: list[dict] = []

    for ticker in tickers:
        print(f"\n{ticker} ...", end=" ", flush=True)
        try:
            t    = yf.Ticker(ticker)
            hist = t.history(start=start_date, end=end_date, interval="1d", auto_adjust=True)
        except Exception as exc:
            print(f"yfinance error: {exc}")
            continue

        if hist.empty:
            print("no data")
            continue

        hist  = compute_indicators(hist)
        count = 0

        for ts, row in hist.iterrows():
            predicted_at = ts.timestamp() + 72000
            if predicted_at >= cutoff_ts:
                continue

            fv = {col: np.nan for col in FEATURE_COLS}
            for col in ["rsi_14", "macd_histogram", "bb_position", "volume_ratio_20d",
                        "price_momentum_1d", "price_momentum_5d", "atr_normalized"]:
                v = row.get(col)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    fv[col] = float(v)

            X = pd.DataFrame([fv], columns=FEATURE_COLS).astype(float)

            for horizon, model in models.items():
                try:
                    prob_up = float(model.predict_proba(X)[:, 1][0])
                    all_predictions.append({
                        "ticker":           ticker,
                        "horizon":          horizon,
                        "predicted_at":     round(predicted_at, 3),
                        "signal":           classify(prob_up),
                        "probability_up":   round(prob_up, 6),
                        "probability_down": round(1.0 - prob_up, 6),
                        "confidence":       round(abs(prob_up - 0.5) * 2, 6),
                        "feature_ts":       round(ts.timestamp(), 3),
                    })
                    count += 1
                except Exception as exc:
                    print(f"\n  error {ticker}/{horizon}: {exc}")

        print(f"{count // max(len(models), 1)} days scored")
        time.sleep(0.3)

    if not all_predictions:
        print("\nNothing to insert.")
        return

    print(f"\nWriting {len(all_predictions)} predictions to D1 ...")
    d1_insert(all_predictions)
    print(f"\nDone — {len(all_predictions)} rows attempted (INSERT OR IGNORE)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        tickers = [t.upper() for t in sys.argv[1:]]
        print(f"Backfilling {len(tickers)} tickers: {tickers}")
    else:
        print("Fetching tracked tickers from D1 ...")
        tickers = get_tickers_from_db()
        print(f"Found {len(tickers)} tickers: {tickers}")

    backfill(tickers)
