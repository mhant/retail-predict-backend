/**
 * Retail Predict — Cloudflare Worker (write proxy + read API)
 *
 * Write (requires Authorization: Bearer <WRITE_TOKEN>):
 *   POST /ingest                — bulk insert rows into a D1 table
 *
 * Read (public, CORS-enabled for React frontend):
 *   GET  /api/sentiment         — ticker sentiment summaries (?window=168)
 *   GET  /api/mentions          — mentions for a ticker (?ticker=GME&window=168)
 *   GET  /api/prices            — OHLCV for a ticker (?ticker=GME&interval=1d)
 *   GET  /api/pipeline          — recent pipeline run history
 *   GET  /api/events            — recent scraper events (invalid tickers etc.)
 *   GET  /health                — DB connectivity check
 */

const CORS = {
  'Access-Control-Allow-Origin':  '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

function apiJson(data, status = 200, cacheTtl = 0) {
  const headers = { 'Content-Type': 'application/json', ...CORS };
  if (cacheTtl > 0) {
    headers['Cache-Control'] = `public, max-age=60, s-maxage=${cacheTtl}`;
  } else {
    headers['Cache-Control'] = 'no-store';
  }
  return new Response(JSON.stringify(data), {
    status,
    headers,
  });
}

// Whitelisted tables and their accepted column names.
// Any table or column not listed here is rejected — prevents SQL injection.
const SCHEMA = {
  pipeline_runs: [
    'started_at', 'finished_at', 'status', 'triggered_by',
    'subreddits_scraped', 'mentions_scraped', 'new_mentions',
    'vader_scored', 'prices_updated', 'predictions_written', 'error_message',
  ],
  raw_mentions: [
    'source', 'source_id', 'subreddit', 'ticker', 'title', 'selftext',
    'author', 'score', 'ups', 'upvote_ratio', 'num_comments',
    'url', 'created_utc', 'scraped_utc',
    'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral',
  ],
  sentiment_scores: [
    'mention_id', 'analyzer', 'compound', 'positive', 'negative',
    'neutral', 'processed_utc',
  ],
  ticker_sentiment_summary: [
    'ticker', 'window_hours', 'computed_at', 'mention_count',
    'avg_sentiment', 'upvote_weighted_sentiment', 'avg_upvote_ratio',
    'source_count', 'top_title',
  ],
  news_articles: [
    'source', 'article_id', 'ticker', 'title', 'summary',
    'url', 'published_utc', 'scraped_utc',
  ],
  price_snapshots: [
    'ticker', 'interval', 'ts', 'open', 'high', 'low', 'close', 'volume',
    'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
    'bb_upper', 'bb_lower', 'bb_position',
    'sma_20', 'sma_50', 'atr_14', 'atr_normalized',
    'volume_ratio_20d', 'price_momentum_1d', 'price_momentum_5d',
  ],
  institutional_data: [
    'ticker', 'report_date', 'institutional_ownership_pct',
    'short_interest_pct', 'short_ratio', 'put_call_ratio',
    'insider_buy_count_30d', 'insider_sell_count_30d',
  ],
  insider_trades: [
    'filing_id', 'ticker', 'company_name', 'insider_name',
    'transaction_type', 'shares', 'price', 'filed_date', 'scraped_utc',
  ],
  model_predictions: [
    'ticker', 'horizon', 'predicted_at', 'signal',
    'probability_up', 'probability_down', 'confidence', 'feature_ts',
  ],
  hype_signals: [
    'ticker', 'computed_at', 'window_hours', 'signal',
    'avg_sentiment', 'upvote_weighted_sentiment',
    'mention_count', 'mention_velocity', 'source_count',
  ],
  combined_predictions: [
    'ticker', 'horizon', 'predicted_at', 'signal',
    'probability_up', 'confidence', 'model_weight', 'hype_weight',
    'model_prediction_id', 'hype_signal_id',
  ],
  prediction_outcomes: [
    'ticker', 'horizon', 'predicted_at', 'prediction_type',
    'prediction_id', 'predicted_signal', 'predicted_prob_up',
    'actual_return', 'evaluated_at', 'was_correct',
  ],
  scraper_events: [
    'event_type', 'ticker', 'detail', 'pipeline_started_at', 'occurred_at',
  ],
  ticker_metadata: [
    'ticker', 'company_name', 'sector', 'industry', 'exchange', 'quote_type', 'updated_at',
  ],
};

// Tables where we REPLACE (upsert) instead of IGNORE on conflict
const UPSERT_TABLES = new Set([
  'ticker_sentiment_summary',
  'institutional_data',
  'pipeline_runs',
  'ticker_metadata',
]);

const D1_BATCH_SIZE = 100; // D1 batch limit per call


function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', ...CORS },
  });
}

function unauthorized() {
  return json({ ok: false, error: 'Unauthorized' }, 401);
}

function badRequest(msg) {
  return json({ ok: false, error: msg }, 400);
}

function isAuthed(request, env) {
  const header = request.headers.get('Authorization') || '';
  const token = header.startsWith('Bearer ') ? header.slice(7) : '';
  return token.length > 0 && token === env.WRITE_TOKEN;
}

/**
 * Build and execute batch INSERT statements for one table.
 * Chunks into groups of D1_BATCH_SIZE to stay within D1 limits.
 */
async function batchInsert(db, table, rows, mode) {
  if (!rows.length) return { inserted: 0, skipped: 0 };

  const allowedCols = SCHEMA[table];
  // Use only columns that exist in this table's schema
  const cols = Object.keys(rows[0]).filter(c => allowedCols.includes(c));
  if (!cols.length) throw new Error(`No valid columns found for table '${table}'`);

  const verb = UPSERT_TABLES.has(table) || mode === 'replace'
    ? 'INSERT OR REPLACE'
    : 'INSERT OR IGNORE';

  const placeholders = cols.map((_, i) => `?${i + 1}`).join(', ');
  const sql = `${verb} INTO ${table} (${cols.join(', ')}) VALUES (${placeholders})`;

  let totalInserted = 0;

  // Process in chunks
  for (let i = 0; i < rows.length; i += D1_BATCH_SIZE) {
    const chunk = rows.slice(i, i + D1_BATCH_SIZE);
    const stmts = chunk.map(row => {
      const values = cols.map(c => row[c] ?? null);
      return db.prepare(sql).bind(...values);
    });
    const results = await db.batch(stmts);
    totalInserted += results.reduce((sum, r) => sum + (r.meta?.changes ?? 0), 0);
  }

  return { inserted: totalInserted, skipped: rows.length - totalInserted };
}


function containsUrl(text) {
  if (/https?:\/\/|ftp:\/\//i.test(text)) return true
  if (/\bwww[\s.]/i.test(text)) return true
  if (/\b[\w-]+\.(com|net|org|io|co|gov|edu|xyz|app|dev|ai|info|biz|me|tv|us|uk|ca|au|de|fr|finance|money|trade)\b/i.test(text)) return true
  if (/\b\w+\s*[\[(]\.?dot\.?[\])]\s*\w+/i.test(text)) return true
  if (/\b\w+\s*[\[(]\s*\.\s*[\])]\s*\w+/i.test(text)) return true
  if (/\b\w+\s+dot\s+\w+/i.test(text)) return true
  return false
}

async function hashIp(ip) {
  const data = new TextEncoder().encode(ip || 'unknown')
  const buf  = await crypto.subtle.digest('SHA-256', data)
  return Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2, '0')).join('').slice(0, 16)
}

// Worker isolate in-memory cache (shared across requests in the same isolate instance)
const ISOLATE_CACHE = new Map();

function getIsolateCached(key, ttlSec = 300) {
  const item = ISOLATE_CACHE.get(key);
  if (item && (Date.now() - item.ts) < ttlSec * 1000) {
    return item.data;
  }
  return null;
}

function setIsolateCache(key, data) {
  if (ISOLATE_CACHE.size > 500) {
    const oldestKey = ISOLATE_CACHE.keys().next().value;
    if (oldestKey) ISOLATE_CACHE.delete(oldestKey);
  }
  ISOLATE_CACHE.set(key, { ts: Date.now(), data });
}

export default {
  async fetch(request, env, ctx) {
    const url    = new URL(request.url);
    const path   = url.pathname;
    const params = url.searchParams;

    // CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS });
    }

    // Public read endpoints that can be safely served from memory / Cloudflare edge cache
    const isCacheableRead = request.method === 'GET' &&
      path.startsWith('/api/') &&
      !path.startsWith('/api/debug') &&
      !path.startsWith('/api/tips/all') &&
      !path.startsWith('/api/tips/reported') &&
      !path.startsWith('/api/tips/recent');

    const cacheKey = request.url;

    if (isCacheableRead) {
      // 1. Check ultra-fast worker isolate memory cache (0.001ms)
      const memData = getIsolateCached(cacheKey);
      if (memData) {
        return apiJson(memData, 200, 300);
      }

      // 2. Check Cloudflare CDN edge cache
      try {
        const cache = caches.default;
        const cached = await cache.match(request);
        if (cached) {
          try {
            const dataClone = await cached.clone().json();
            setIsolateCache(cacheKey, dataClone);
          } catch (_) {}
          return cached;
        }
      } catch (_) {}
    }

    function respond(data, status = 200, cacheTtl = 300) {
      if (isCacheableRead && cacheTtl > 0) {
        setIsolateCache(cacheKey, data);
      }
      const res = apiJson(data, status, cacheTtl);
      if (isCacheableRead && cacheTtl > 0 && ctx && ctx.waitUntil) {
        try {
          ctx.waitUntil(caches.default.put(request, res.clone()));
        } catch (_) {}
      }
      return res;
    }

    // ── Read API (public, no auth) ────────────────────────────────────────────

    // GET /api/sentiment?window=168
    // Reads directly from pre-aggregated ticker_sentiment_summary (reads ~100 rows instead of 50,000)
    if (request.method === 'GET' && path === '/api/sentiment') {
      const windowHours = Math.min(720, Math.max(1, parseInt(params.get('window') || '168', 10) || 168));
      
      // 1. First attempt: read from pre-aggregated ticker_sentiment_summary
      let { results } = await env.DB.prepare(
        `SELECT
           tss.ticker,
           tss.mention_count,
           tss.avg_sentiment,
           tss.upvote_weighted_sentiment,
           tss.avg_upvote_ratio,
           tss.source_count,
           tss.computed_at                             AS latest_mention_utc,
           tss.computed_at                             AS first_mention_utc,
           tm.company_name,
           tm.sector,
           tm.industry,
           tm.exchange,
           tm.quote_type,
           tss.top_title
         FROM ticker_sentiment_summary tss
         LEFT JOIN ticker_metadata tm ON tm.ticker = tss.ticker
         WHERE tss.ticker NOT IN (
           SELECT ticker FROM scraper_events
           WHERE event_type IN ('ticker_not_found', 'delisted')
           GROUP BY ticker HAVING COUNT(*) >= 2
         )
         ORDER BY tss.computed_at DESC, tss.mention_count DESC
         LIMIT 200`
      ).all();

      // 2. Safe fallback if summary table hasn't been populated yet: clean 1-pass query without correlated subquery
      if (!results || results.length === 0) {
        const cutoff = Date.now() / 1000 - windowHours * 3600;
        const fallback = await env.DB.prepare(
          `SELECT
             rm.ticker,
             COUNT(*)                                    AS mention_count,
             AVG(vader_compound)                         AS avg_sentiment,
             AVG(vader_compound)                         AS upvote_weighted_sentiment,
             AVG(upvote_ratio)                           AS avg_upvote_ratio,
             COUNT(DISTINCT subreddit)                   AS source_count,
             MAX(scraped_utc)                            AS latest_mention_utc,
             MIN(scraped_utc)                            AS first_mention_utc,
             tm.company_name,
             tm.sector,
             tm.industry,
             tm.exchange,
             tm.quote_type,
             ''                                          AS top_title
           FROM raw_mentions rm
           LEFT JOIN ticker_metadata tm ON tm.ticker = rm.ticker
           WHERE scraped_utc >= ?1 AND vader_compound IS NOT NULL
           GROUP BY rm.ticker
           ORDER BY MAX(scraped_utc) DESC
           LIMIT 200`
        ).bind(cutoff).all();
        results = fallback.results || [];
      }

      return respond({ ok: true, data: results }, 200, 300);
    }

    // GET /api/tracked-tickers — all active tracked tickers
    if (request.method === 'GET' && path === '/api/tracked-tickers') {
      const { results } = await env.DB.prepare(
        `SELECT DISTINCT ticker FROM ticker_sentiment_summary
         WHERE ticker IS NOT NULL
           AND ticker NOT IN (
             SELECT ticker FROM scraper_events
             WHERE event_type IN ('ticker_not_found', 'delisted')
             GROUP BY ticker HAVING COUNT(*) >= 2
           )
         UNION
         SELECT DISTINCT ticker FROM hype_signals
         WHERE ticker IS NOT NULL
           AND ticker NOT IN (
             SELECT ticker FROM scraper_events
             WHERE event_type IN ('ticker_not_found', 'delisted')
             GROUP BY ticker HAVING COUNT(*) >= 2
           )
         ORDER BY ticker`
      ).all();
      return respond({ ok: true, tickers: results.map(r => r.ticker) }, 200, 300);
    }

    // GET /api/invalid-tickers — tickers with 2+ yfinance failures (used by pipeline)
    if (request.method === 'GET' && path === '/api/invalid-tickers') {
      const { results } = await env.DB.prepare(
        `SELECT ticker FROM scraper_events
         WHERE event_type IN ('ticker_not_found', 'delisted')
         GROUP BY ticker HAVING COUNT(*) >= 2`
      ).all();
      return respond({ ok: true, tickers: results.map(r => r.ticker) }, 200, 300);
    }

    // GET /api/mentions?ticker=GME
    // No time filter — top-scored PullPush posts can be months old, a window
    // filter would exclude all real data. Returns top 50 by score instead.
    if (request.method === 'GET' && path === '/api/mentions') {
      const ticker = params.get('ticker');
      if (!ticker) return apiJson({ ok: false, error: 'ticker param required' }, 400);
      const { results } = await env.DB.prepare(
        `SELECT title, selftext, author, score, ups, upvote_ratio, num_comments,
                created_utc, vader_compound, subreddit, url
         FROM raw_mentions
         WHERE ticker = ?1
         ORDER BY score DESC LIMIT 50`
      ).bind(ticker).all();
      return respond({ ok: true, ticker, data: results }, 200, 300);
    }

    // GET /api/prices?ticker=GME&interval=1d
    if (request.method === 'GET' && path === '/api/prices') {
      const ticker   = params.get('ticker');
      const interval = params.get('interval') || '1d';
      if (!ticker) return apiJson({ ok: false, error: 'ticker param required' }, 400);
      const { results } = await env.DB.prepare(
        `SELECT ts, open, high, low, close, volume,
                rsi_14, macd_histogram, bb_upper, bb_lower,
                sma_20, sma_50, volume_ratio_20d
         FROM price_snapshots
         WHERE ticker = ?1 AND interval = ?2
         ORDER BY ts ASC`
      ).bind(ticker, interval).all();
      return respond({ ok: true, ticker, interval, data: results }, 200, 300);
    }

    // GET /api/prices/latest-timestamps?interval=1d
    if (request.method === 'GET' && path === '/api/prices/latest-timestamps') {
      const interval = params.get('interval') || '1d';
      const { results } = await env.DB.prepare(
        `SELECT ticker, MAX(ts) AS latest_ts, COUNT(*) AS bar_count
         FROM price_snapshots
         WHERE interval = ?1
         GROUP BY ticker`
      ).bind(interval).all();
      const map = {};
      for (const r of (results || [])) {
        map[r.ticker] = { latest_ts: r.latest_ts, bar_count: r.bar_count };
      }
      return respond({ ok: true, data: map }, 200, 60);
    }

    // GET /api/pipeline
    if (request.method === 'GET' && path === '/api/pipeline') {
      const { results } = await env.DB.prepare(
        `SELECT id, started_at, finished_at, status, triggered_by,
                mentions_scraped, new_mentions, prices_updated,
                predictions_written, error_message
         FROM pipeline_runs ORDER BY started_at DESC LIMIT 20`
      ).all();
      return respond({ ok: true, data: results }, 200, 60);
    }

    // GET /api/predictions?ticker=GME
    if (request.method === 'GET' && path === '/api/predictions') {
      const ticker = params.get('ticker')
      if (!ticker) return apiJson({ ok: false, error: 'ticker param required' }, 400)
      const [modelRows, hyp, histRows] = await Promise.all([
        env.DB.prepare(
          `SELECT horizon, signal, probability_up, confidence, predicted_at, model_weight, hype_weight
           FROM combined_predictions WHERE ticker = ?1
           ORDER BY predicted_at DESC LIMIT 3`
        ).bind(ticker).all(),
        env.DB.prepare(
          `SELECT signal, avg_sentiment, upvote_weighted_sentiment, mention_count,
                  mention_velocity, source_count, computed_at
           FROM hype_signals WHERE ticker = ?1
           ORDER BY computed_at DESC LIMIT 1`
        ).bind(ticker).first(),
        env.DB.prepare(
          `SELECT horizon, signal, probability_up, confidence, predicted_at, model_weight, hype_weight
           FROM combined_predictions WHERE ticker = ?1
           ORDER BY predicted_at DESC LIMIT 100`
        ).bind(ticker).all(),
      ])
      return respond({ ok: true, ticker, model: modelRows.results, hype: hyp, history: histRows.results }, 200, 300);
    }

    // GET /api/events
    if (request.method === 'GET' && path === '/api/events') {
      const { results } = await env.DB.prepare(
        `SELECT event_type, ticker, detail, occurred_at
         FROM scraper_events ORDER BY occurred_at DESC LIMIT 100`
      ).all();
      return respond({ ok: true, data: results }, 200, 300);
    }

    // GET /api/debug/ip-hash — returns your hashed IP (for setting up the rate-limit allowlist)
    if (request.method === 'GET' && path === '/api/debug/ip-hash') {
      const ipHash = await hashIp(request.headers.get('CF-Connecting-IP') || '')
      return json({ ip_hash: ipHash })
    }

    // ── Tipline ───────────────────────────────────────────────────────────────

    // POST /api/tips — submit a user tip (public, Turnstile + moderation guarded)
    if (request.method === 'POST' && path === '/api/tips') {
      if (!env.TURNSTILE_SECRET) {
        return json({ ok: false, error: 'Tip submission not configured' }, 503)
      }
      let body
      try { body = await request.json() } catch { return badRequest('Invalid JSON') }

      const { ticker, tip, turnstileToken } = body
      if (!ticker || typeof ticker !== 'string') return badRequest('ticker required')
      if (!tip || typeof tip !== 'string') return badRequest('tip required')
      const trimmed = tip.trim()
      if (trimmed.length < 10)  return badRequest('Tip must be at least 10 characters')
      if (trimmed.length > 500) return badRequest('Tip must be under 500 characters')
      if (containsUrl(trimmed)) return json({ ok: false, error: 'contains_link' }, 422)
      if (!turnstileToken)      return badRequest('Turnstile token required')

      // Verify Turnstile
      const tsForm = new FormData()
      tsForm.append('secret',   env.TURNSTILE_SECRET)
      tsForm.append('response', turnstileToken)
      tsForm.append('remoteip', request.headers.get('CF-Connecting-IP') || '')
      const tsRes  = await fetch('https://challenges.cloudflare.com/turnstile/v0/siteverify', { method: 'POST', body: tsForm })
      const tsData = await tsRes.json()
      if (!tsData.success) return json({ ok: false, error: 'bot_check_failed' }, 422)

      // Rate limit: max 5 tips per IP per hour (allowlisted IPs bypass)
      const ipHash     = await hashIp(request.headers.get('CF-Connecting-IP') || '')
      const allowlist  = (env.ALLOWLISTED_IP_HASHES || '').split(',').map(s => s.trim()).filter(Boolean)
      const isAllowed  = allowlist.includes(ipHash)
      if (!isAllowed) {
        const cutoff = Math.floor(Date.now() / 1000) - 3600
        const rateRow = await env.DB.prepare(
          'SELECT COUNT(*) AS cnt FROM tips WHERE ip_hash = ?1 AND submitted_at > ?2'
        ).bind(ipHash, cutoff).first()
        if ((rateRow?.cnt ?? 0) >= 5) return json({ ok: false, error: 'rate_limited' }, 429)
      }

      // Cloudflare Workers AI moderation — fail closed
      try {
        const aiRes = await env.AI.run('@cf/meta/llama-3.1-8b-instruct', {
          messages: [
            {
              role: 'system',
              content: `You are a content safety classifier for a stock market discussion site. Users post opinions about publicly traded companies and market trends — e.g. "NVDA looks bullish", "Apple poised to dominate AI", "OpenAI and Anthropic prepping for IPOs". Financial discussion, stock tips, company analysis, and investment opinions are ALL safe, even if you disagree with them.

Classify as UNSAFE only if the message contains: hate speech, racial or ethnic slurs, explicit threats of violence against a person, explicit sexual content, sexual content involving minors, or targeted personal harassment.

Market opinions, company names, predictions, hype, FUD, and any other stock-related discussion are SAFE.

Reply with ONLY the single word "safe" or "unsafe". No punctuation, no explanation.`,
            },
            { role: 'user', content: trimmed },
          ],
          max_tokens: 5,
          temperature: 0,
        })
        const verdict = (aiRes.response || '').trim().toLowerCase()
        if (!verdict.startsWith('safe')) {
          return json({ ok: false, error: 'flagged' }, 422)
        }
      } catch (err) {
        console.error('AI moderation error:', err.message)
        return json({ ok: false, error: 'moderation_unavailable' }, 503)
      }

      // Store approved tip
      const now = Math.floor(Date.now() / 1000)
      await env.DB.prepare(
        'INSERT INTO tips (ticker, body, status, submitted_at, ip_hash) VALUES (?1, ?2, ?3, ?4, ?5)'
      ).bind(ticker.toUpperCase().trim(), trimmed, 'approved', now, ipHash).run()

      return json({ ok: true })
    }

    // GET /api/tips?ticker=GME — public feed of approved tips for a ticker
    if (request.method === 'GET' && path === '/api/tips') {
      const ticker = params.get('ticker')
      if (!ticker) return badRequest('ticker param required')
      const { results } = await env.DB.prepare(
        `SELECT id, ticker, body, submitted_at
         FROM tips WHERE ticker = ?1 AND status = 'approved' AND report_count < 5
         ORDER BY submitted_at DESC LIMIT 20`
      ).bind(ticker.toUpperCase()).all()
      return respond({ ok: true, ticker, data: results }, 200, 60)
    }

    // GET /api/tips/feed — all approved tips, newest first (public)
    if (request.method === 'GET' && path === '/api/tips/feed') {
      const { results } = await env.DB.prepare(
        `SELECT id, ticker, body, submitted_at
         FROM tips WHERE status = 'approved' AND report_count < 5
         ORDER BY submitted_at DESC LIMIT 100`
      ).all()
      return respond({ ok: true, data: results }, 200, 60)
    }

    // POST /api/tips/:id/report — increment report count (public)
    const reportMatch = path.match(/^\/api\/tips\/(\d+)\/report$/)
    if (request.method === 'POST' && reportMatch) {
      const tipId = parseInt(reportMatch[1])
      const { meta } = await env.DB.prepare(
        'UPDATE tips SET report_count = report_count + 1 WHERE id = ?1'
      ).bind(tipId).run()
      if (!meta.changes) return json({ ok: false, error: 'not_found' }, 404)
      return json({ ok: true })
    }

    // DELETE /api/tips/:id — hard delete (auth required)
    const deleteMatch = path.match(/^\/api\/tips\/(\d+)$/)
    if (request.method === 'DELETE' && deleteMatch) {
      if (!isAuthed(request, env)) return unauthorized()
      const tipId = parseInt(deleteMatch[1])
      await env.DB.prepare('DELETE FROM tips WHERE id = ?1').bind(tipId).run()
      return json({ ok: true })
    }

    // GET /api/tips/all — all tips with optional ?search= and ?ticker= filters (auth required)
    if (request.method === 'GET' && path === '/api/tips/all') {
      if (!isAuthed(request, env)) return unauthorized()
      const search = params.get('search') || ''
      const ticker = params.get('ticker') || ''
      const status = params.get('status') || ''
      let query = 'SELECT id, ticker, body, status, report_count, submitted_at FROM tips WHERE 1=1'
      const bindings = []
      if (ticker) { query += ' AND ticker = ?'; bindings.push(ticker.toUpperCase().trim()) }
      if (status === 'reported') { query += ' AND report_count >= 5' }
      else if (status === 'clean') { query += ' AND report_count = 0' }
      if (search) { query += ' AND body LIKE ?'; bindings.push(`%${search}%`) }
      query += ' ORDER BY submitted_at DESC LIMIT 500'
      const { results } = await env.DB.prepare(query).bind(...bindings).all()
      return apiJson({ ok: true, data: results })
    }

    // GET /api/tips/reported — tips flagged for review (auth required)
    if (request.method === 'GET' && path === '/api/tips/reported') {
      if (!isAuthed(request, env)) return unauthorized()
      const { results } = await env.DB.prepare(
        'SELECT id, ticker, body, report_count, submitted_at FROM tips WHERE report_count > 0 ORDER BY report_count DESC, submitted_at DESC'
      ).all()
      return apiJson({ ok: true, data: results })
    }

    // GET /api/tips/recent — all approved tips from last 30h (pipeline, auth required)
    if (request.method === 'GET' && path === '/api/tips/recent') {
      if (!isAuthed(request, env)) return unauthorized()
      const cutoff = Math.floor(Date.now() / 1000) - 108000
      const { results } = await env.DB.prepare(
        `SELECT id, ticker, body, submitted_at
         FROM tips WHERE status = 'approved' AND report_count < 5 AND submitted_at > ?1
         ORDER BY submitted_at DESC`
      ).bind(cutoff).all()
      return apiJson({ ok: true, data: results })
    }

    // ── Health check (no auth) ────────────────────────────────────────────────
    if (request.method === 'GET' && path === '/health') {
      try {
        // Verify DB is reachable
        await env.DB.prepare("SELECT 1").first();
        return json({
          ok: true,
          service: 'retail-predict-worker',
          db: 'retail-predict',
          timestamp: new Date().toISOString(),
        });
      } catch (err) {
        return json({ ok: false, error: 'DB unreachable', detail: err.message }, 503);
      }
    }

    // ── Ingest endpoint ───────────────────────────────────────────────────────
    if (request.method === 'POST' && path === '/ingest') {
      if (!isAuthed(request, env)) return unauthorized();

      let body;
      try {
        body = await request.json();
      } catch {
        return badRequest('Invalid JSON body');
      }

      const { table, rows, mode } = body;

      if (!table || typeof table !== 'string') return badRequest('Missing or invalid "table"');
      if (!SCHEMA[table]) return badRequest(`Unknown table '${table}'. Allowed: ${Object.keys(SCHEMA).join(', ')}`);
      if (!Array.isArray(rows) || rows.length === 0) return badRequest('"rows" must be a non-empty array');
      if (rows.length > 5000) return badRequest('Max 5000 rows per request');

      try {
        const { inserted, skipped } = await batchInsert(env.DB, table, rows, mode);
        ISOLATE_CACHE.clear();
        return json({ ok: true, table, attempted: rows.length, inserted, skipped });
      } catch (err) {
        console.error(`Ingest error for table '${table}':`, err);
        return json({ ok: false, error: err.message }, 500);
      }
    }

    return json({ ok: false, error: 'Not found' }, 404);
  },
};
