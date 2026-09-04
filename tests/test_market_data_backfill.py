"""
Unit tests for price snapshots write optimization and new ticker backfill preservation.
"""
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import time

# Set dummy env vars for scraper.config import
import os
os.environ.setdefault("WORKER_URL", "https://dummy-worker.workers.dev")
os.environ.setdefault("WRITE_TOKEN", "dummy-write-token")

from scraper.pipeline import fetch_market_data, run_predictions
from scraper import d1_client


def _make_mock_history(num_bars: int = 70) -> pd.DataFrame:
    """Generate a realistic synthetic daily price dataframe."""
    now = time.time()
    # 70 daily timestamps spaced by 86400 seconds
    timestamps = [pd.Timestamp.fromtimestamp(now - (num_bars - i) * 86400) for i in range(num_bars)]
    
    data = {
        "Open":   np.linspace(100.0, 150.0, num_bars),
        "High":   np.linspace(105.0, 155.0, num_bars),
        "Low":    np.linspace(98.0, 148.0, num_bars),
        "Close":  np.linspace(102.0, 152.0, num_bars),
        "Volume": np.random.randint(1000000, 5000000, size=num_bars),
    }
    df = pd.DataFrame(data, index=timestamps)
    return df


class TestMarketDataBackfill(unittest.TestCase):

    @patch("scraper.pipeline.yf.Ticker")
    def test_new_ticker_receives_full_backfill(self, mock_ticker_cls):
        """New ticker with no record in existing_price_meta gets all ~70 historical bars."""
        mock_df = _make_mock_history(70)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_ticker.info = {
            "shortPercentOfFloat": 0.05,
            "shortRatio": 2.5,
            "institutionsPercentHeld": 0.65,
            "longName": "New Test Company Inc.",
        }
        mock_ticker_cls.return_value = mock_ticker

        # Existing metadata is empty (completely new ticker)
        existing_meta = {}

        price_rows, inst_rows, event_rows, meta_rows = fetch_market_data(
            ["NEWCO"], pipeline_started_at=time.time(), existing_price_meta=existing_meta
        )

        self.assertEqual(len(price_rows), 70, "New ticker must retain all 70 historical price bars for backfill")
        self.assertEqual(price_rows[0]["ticker"], "NEWCO")
        self.assertIsNotNone(price_rows[-1]["rsi_14"], "Indicators should be populated")
        self.assertEqual(len(inst_rows), 1)
        self.assertEqual(len(meta_rows), 1)
        self.assertEqual(meta_rows[0]["company_name"], "New Test Company Inc.")

    @patch("scraper.pipeline.yf.Ticker")
    def test_low_bar_count_ticker_receives_full_backfill(self, mock_ticker_cls):
        """Ticker with <15 bars in D1 is treated as incomplete and gets full 70-bar backfill."""
        mock_df = _make_mock_history(70)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_ticker.info = {}
        mock_ticker_cls.return_value = mock_ticker

        # Only 5 bars previously recorded in DB
        existing_meta = {
            "PARTIAL": {"latest_ts": mock_df.index[-1].timestamp(), "bar_count": 5}
        }

        price_rows, _, _, _ = fetch_market_data(
            ["PARTIAL"], pipeline_started_at=time.time(), existing_price_meta=existing_meta
        )

        self.assertEqual(len(price_rows), 70, "Incompletely backfilled ticker (<15 bars) must get all 70 bars")

    @patch("scraper.pipeline.yf.Ticker")
    def test_existing_ticker_selective_write(self, mock_ticker_cls):
        """Existing ticker with 70 bars in DB only writes newest candles (ts >= latest_ts)."""
        mock_df = _make_mock_history(70)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_ticker.info = {}
        mock_ticker_cls.return_value = mock_ticker

        # DB has data up to the 2nd to last bar (index -2)
        latest_db_ts = mock_df.index[-2].timestamp()
        existing_meta = {
            "AAPL": {"latest_ts": latest_db_ts, "bar_count": 70}
        }

        price_rows, _, _, _ = fetch_market_data(
            ["AAPL"], pipeline_started_at=time.time(), existing_price_meta=existing_meta
        )

        # Only index -2 and index -1 should be returned (2 rows instead of 70)
        self.assertEqual(len(price_rows), 2, "Existing ticker should only write candles at or after latest_ts")
        self.assertGreaterEqual(price_rows[0]["ts"], latest_db_ts)
        self.assertGreaterEqual(price_rows[1]["ts"], latest_db_ts)

    @patch("scraper.pipeline.yf.Ticker")
    def test_none_existing_meta_defaults_to_full_backfill(self, mock_ticker_cls):
        """When existing_price_meta is None (e.g. network failure), safely default to full backfill."""
        mock_df = _make_mock_history(70)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_ticker.info = {}
        mock_ticker_cls.return_value = mock_ticker

        price_rows, _, _, _ = fetch_market_data(
            ["GME"], pipeline_started_at=time.time(), existing_price_meta=None
        )

        self.assertEqual(len(price_rows), 70, "None metadata must safely default to full backfill without crashing")

    @patch("scraper.pipeline.yf.Ticker")
    def test_mixed_new_and_existing_tickers(self, mock_ticker_cls):
        """A batch containing both a new ticker and an existing ticker writes 70 rows for new and 1-2 for existing."""
        mock_df = _make_mock_history(70)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_ticker.info = {}
        mock_ticker_cls.return_value = mock_ticker

        latest_db_ts = mock_df.index[-1].timestamp()
        existing_meta = {
            "EXISTING": {"latest_ts": latest_db_ts, "bar_count": 70}
            # "BRANDNEW" is not in existing_meta
        }

        price_rows, _, _, _ = fetch_market_data(
            ["EXISTING", "BRANDNEW"], pipeline_started_at=time.time(), existing_price_meta=existing_meta
        )

        existing_rows = [r for r in price_rows if r["ticker"] == "EXISTING"]
        new_rows = [r for r in price_rows if r["ticker"] == "BRANDNEW"]

        self.assertEqual(len(existing_rows), 1, "Existing ticker should have only 1 row")
        self.assertEqual(len(new_rows), 70, "New ticker should have all 70 rows")
        self.assertEqual(len(price_rows), 71, "Total rows written should be 71 (not 140)")


class TestD1ClientTimestamps(unittest.TestCase):

    @patch("scraper.d1_client._SESSION.get")
    def test_fetch_price_latest_timestamps_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "ok": True,
            "data": {
                "AAPL": {"latest_ts": 1725321600.0, "bar_count": 70},
                "NVDA": {"latest_ts": 1725321600.0, "bar_count": 68},
            }
        }
        mock_get.return_value = mock_resp

        result = d1_client.fetch_price_latest_timestamps("1d")
        self.assertIn("AAPL", result)
        self.assertEqual(result["AAPL"]["bar_count"], 70)
        self.assertEqual(result["NVDA"]["latest_ts"], 1725321600.0)

    @patch("scraper.d1_client._SESSION.get")
    def test_fetch_price_latest_timestamps_error_handling(self, mock_get):
        mock_get.side_effect = Exception("Connection timed out")
        result = d1_client.fetch_price_latest_timestamps("1d")
        self.assertEqual(result, {}, "Should return empty dict on failure without raising exception")


if __name__ == "__main__":
    unittest.main()
