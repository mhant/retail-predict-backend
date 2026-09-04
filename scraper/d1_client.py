"""HTTP client for the Cloudflare Worker write proxy."""
from __future__ import annotations

import time
import requests
from scraper.config import WORKER_URL, WRITE_TOKEN

_SESSION = requests.Session()
_SESSION.headers.update({
    "Authorization": f"Bearer {WRITE_TOKEN}",
    "Content-Type": "application/json",
})

_CHUNK_SIZE = 500   # rows per Worker request (well under 5000 limit)
_MAX_RETRIES = 3


def fetch_tracked_tickers() -> frozenset[str]:
    try:
        resp = _SESSION.get(f"{WORKER_URL}/api/tracked-tickers", timeout=10)
        if resp.ok:
            return frozenset(resp.json().get("tickers", []))
    except Exception:
        pass
    return frozenset()


def fetch_user_tips() -> list[dict]:
    try:
        resp = _SESSION.get(f"{WORKER_URL}/api/tips/recent", timeout=10)
        if resp.ok:
            return resp.json().get("data", [])
    except Exception:
        pass
    return []


def fetch_invalid_tickers() -> frozenset[str]:
    try:
        resp = _SESSION.get(
            f"{WORKER_URL}/api/invalid-tickers",
            timeout=10,
        )
        if resp.ok:
            return frozenset(resp.json().get("tickers", []))
    except Exception:
        pass
    return frozenset()


def fetch_price_latest_timestamps(interval: str = "1d") -> dict[str, dict]:
    """
    Fetch latest price timestamp and bar count per ticker from D1.
    Returns: { "GME": {"latest_ts": 1725321600.0, "bar_count": 70}, ... }
    """
    try:
        resp = _SESSION.get(
            f"{WORKER_URL}/api/prices/latest-timestamps?interval={interval}",
            timeout=10,
        )
        if resp.ok:
            return resp.json().get("data", {})
    except Exception as exc:
        print(f"  [d1] warning: failed to fetch price timestamps ({exc}) — defaulting to full backfill")
    return {}


def ingest(table: str, rows: list[dict], mode: str = "ignore") -> dict:
    """
    Bulk-insert rows into a D1 table via the Worker proxy.
    Chunks automatically. Returns aggregate inserted/skipped counts.
    """
    if not rows:
        return {"inserted": 0, "skipped": 0, "attempted": 0}

    total = {"inserted": 0, "skipped": 0, "attempted": 0}

    for i in range(0, len(rows), _CHUNK_SIZE):
        chunk = rows[i : i + _CHUNK_SIZE]
        result = _post_with_retry(table, chunk, mode)
        total["inserted"]  += result.get("inserted", 0)
        total["skipped"]   += result.get("skipped", 0)
        total["attempted"] += result.get("attempted", len(chunk))

    return total


def _post_with_retry(table: str, rows: list[dict], mode: str) -> dict:
    for attempt in range(_MAX_RETRIES):
        try:
            resp = _SESSION.post(
                f"{WORKER_URL}/ingest",
                json={"table": table, "rows": rows, "mode": mode},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if attempt == _MAX_RETRIES - 1:
                raise RuntimeError(f"D1 ingest failed for '{table}' after {_MAX_RETRIES} attempts: {exc}") from exc
            time.sleep(2 ** attempt)   # exponential backoff: 1s, 2s
    return {}
