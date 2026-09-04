-- Retail Predict — D1 Read Performance Indexes
-- Eliminates table scans and optimizes time-window queries on raw_mentions and price_snapshots

CREATE INDEX IF NOT EXISTS ix_mentions_scraped_compound ON raw_mentions (scraped_utc DESC, vader_compound);
CREATE INDEX IF NOT EXISTS ix_mentions_ticker_scraped_score ON raw_mentions (ticker, scraped_utc DESC, score DESC);
CREATE INDEX IF NOT EXISTS ix_price_interval_ticker_ts ON price_snapshots (interval, ticker, ts DESC);
CREATE INDEX IF NOT EXISTS ix_events_type_ticker ON scraper_events (event_type, ticker);
