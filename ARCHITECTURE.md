# GraphOne / FrontierAtlas Intelligence Graph
## Ingestion Pipeline — Architecture & Production Design

---

### 1. Scale Strategy — collecting 500,000+ records without manual intervention

The pipeline is built around one principle: **the code that scrapes 100 records is the same
code that scrapes 500,000; only the loop bound and the number of parallel workers change.**

- **Pagination as the scaling primitive.** Every source scraper (`scrape_papers_with_stars`,
  `scrape_startups_and_products`, `scrape_all_fresh`) is a bounded loop over pages/cursors.
  Raising `target_count` from 1,000 to 500,000 requires no code change — only infrastructure.
- **Horizontal sharding.** At production scale, a single process is not the unit of
  concurrency — a *partition* is. Arxiv queries shard by category + date range
  (`cat:cs.AI AND submittedDate:[2024-01-01 TO 2024-01-07]`, etc.); directory scrapers shard
  by alphabetical prefix or category ID. Each shard is an independent job pushed onto a task
  queue (e.g. Celery/SQS/Cloud Tasks) and picked up by a pool of stateless worker containers.
  Adding capacity is adding workers, not rewriting logic.
- **Per-domain concurrency policies** (`DomainPolicy` in `src/utils/http.py`) are attached to
  the *domain*, not the run, so a fleet of 200 workers still only issues, say, 2 req/s against
  a given host — scale-out adds throughput across *sources* without overwhelming any one of
  them.
- **Idempotent writes.** Every record's identity key is `(recordType, source.url)` or, for
  content without a stable URL, `content_fingerprint()` (SHA-256 of normalized text). Workers
  can be re-run, retried, or duplicated without producing duplicate records downstream — this
  is what makes "add more workers" safe rather than "add more workers and now reconcile
  duplicates."
- **Backpressure, not backlog.** Each worker's target queue depth is monitored; if the
  LLM-extraction stage falls behind the scraping stage (likely, since scraping fans out much
  faster than a token-budgeted LLM call completes), scrapers throttle rather than pile raw HTML
  into unbounded memory. In practice this is a bounded queue (Redis list / SQS) between
  "raw fetched" and "LLM extracted" stages.

### 2. Handling 413s & 429s across thousands of concurrent extractions

**429 (rate limit) — three layers of defense, applied in order:**

1. *Prevention*: `DomainPolicy` (`AsyncLimiter` + `asyncio.Semaphore`) caps steady-state
   request rate per domain below the host's documented or observed limit, so 429s are the
   exception, not the norm.
2. *Reaction*: `HttpClient._get_with_retry` (src/utils/http.py) catches 429, reads
   `Retry-After` when present, and otherwise backs off with `tenacity`'s
   `wait_exponential_jitter` (full jitter, capped at 60s, 5 attempts) — this avoids the
   thundering-herd problem where every worker retries at the exact same moment.
3. *LLM-tier 429s specifically*: the `LLMOrchestrator` (src/llm/orchestrator.py) treats a 429
   from one provider as a signal to fall to the **next tier in the chain**
   (Gemini Flash → Groq Llama 3 → DeepSeek) rather than only retrying the same provider —
   this converts a rate-limit event into near-zero added latency instead of a stall.

**413 (payload too large) — handled proactively, not reactively**, because discovering a 413
after sending is wasted latency at scale:

- `chunk_for_budget()` computes each provider's byte budget from its documented context window
  *before* the call and truncates to fit, keeping the first ~70% (headline/lede/abstract —
  highest information density) and the last ~30% (often bylines, dates, footer metadata),
  dropping the noisy middle first. This is chosen over naive head-only truncation because
  publication dates and author info are frequently *appended* to article bodies.
- Each provider in the fallback chain declares its own `max_context_tokens`, so a payload that
  fits Gemini Flash's window but not Groq's is automatically re-chunked to a smaller budget
  when the chain falls through, rather than passing the same oversized payload to every tier
  and 413'ing on each one in turn.

**At thousands-of-concurrent-extractions scale**, the LLM stage runs as its own worker pool,
sized independently from the scraping pool, with each worker's per-provider concurrency capped
low enough that a full fleet doesn't collectively exceed the *provider's* org-level rate limit
(distinct from the per-worker limit) — this requires a shared, cross-worker token-bucket
(Redis-backed) rather than the in-process `AsyncLimiter` used for the single-process demo, which
is called out explicitly in the Storage Strategy section below.

### 3. Freshness Tracking — never process the same article/job twice across distributed nodes

Three mechanisms compose to guarantee this:

1. **Deterministic content fingerprinting**: `content_fingerprint()` (src/utils/dates.py)
   hashes whitespace-normalized content, giving every record a stable identity independent of
   which node fetched it or when.
2. **Distributed seen-set**: `SeenStore` in the demo is an in-process `set()`; in production
   this is a **Redis SET with a 30-day TTL** keyed by `content_fingerprint`, shared across all
   crawler nodes. Every node checks-and-sets atomically (`SADD` returns whether the value was
   new), so two nodes racing to fetch the same freshly-published article both get a correct
   fresh/duplicate verdict with no double-processing — this is the cross-node piece that a
   local `set()` cannot provide.
3. **Strict date window as the primary filter, content-identity as the fallback**: a record
   with a parseable date is fresh iff `now - published_date <= 24h`
   (`is_fresh()` in src/utils/dates.py). A record with **no** parseable date (missing meta tag,
   non-standard format) falls back to "is this content new to the seen-set" — this is the
   Phase II "intelligent heuristic," and it degrades gracefully: a source with reliable dates
   is governed by the strict window; a source without one is governed by novelty, which is a
   safe default because re-scraping unchanged content is wasted work but re-processing *new*
   content the moment it appears is exactly the freshness goal.

### 4. Storage Strategy

- **Raw fetched HTML/text → Object storage (S3/GCS)**, keyed by `content_fingerprint`. Cheap,
  durable, write-once; lets us re-run LLM extraction later (new model, bug fix) without
  re-scraping — re-scraping is the expensive, rate-limited step, so it's never repeated once done.
- **Canonical structured records → PostgreSQL**, one table per record type, JSONB `content`
  column plus typed columns for fields we filter/join on (`canonicalName`, `collectedAt`,
  `github_stars`). The schema is semi-structured (many optional fields) but query patterns —
  "papers with >1k stars," "jobs posted today," "products for canonical startup X" — are
  classic relational filters/joins; JSONB gives schema flexibility without losing SQL.
- **Entity relationship graph → Neo4j** (or Postgres + Apache AGE if a second database is
  undesirable). The product is explicitly an "Intelligence *Graph*" — multi-hop queries like
  "papers from authors at startups that raised funding in the last 30 days" are graph
  traversals, not joins-of-joins; a dedicated graph store keeps these O(traversal depth)
  instead of O(join explosion).
- **Semantic search (papers/news/products) → pgvector** on the same Postgres instance
  initially, migrating to a dedicated vector store (Qdrant) only once query volume justifies it.
- **Cross-node dedup / rate-limit token buckets → Redis.** Sub-millisecond atomic ops
  (`SADD`, token-bucket scripts) needed at the request-per-second granularity of thousands of
  concurrent workers — a relational database is the wrong latency class for this.
- **Entity Mapping Log (audit trail) → append-only Postgres table**, never mutated. The
  assignment asks for this as a deliverable tab; append-only means the resolution history for
  any entity can always be reconstructed.

**Why this split**: Postgres is the system of record, queryable by analysts immediately
(satisfies the Sheets-export deliverable via a plain `SELECT`); the graph store is the
product-facing traversal layer; object storage is the cheap archive that makes re-processing
free; Redis is the low-latency coordination layer that makes running hundreds of independent
scraping/extraction workers safe.

---

### Appendix: Anti-Bot Strategy (Phase V)

For standard bot defenses, `StealthBrowserPool` (src/scrapers/anti_bot.py) uses pooled,
persistent Playwright contexts with `navigator.webdriver` patched out, realistic
viewport/locale/UA, human-paced delays, and network-idle waits instead of fixed sleeps.

For **Cloudflare-managed-challenge or Datadome-class** protection, defeating the challenge
in-house is unreliable (challenges change frequently) and a ToS/legal grey area to build
custom. The production strategy for these hardest-blocked, highest-value sources is to route
*only those domains* through a managed unblocking API (Bright Data, ScraperAPI, Zyte) behind
the same `HttpClient.get()` interface used everywhere else — `DomainPolicy` decides per-domain
whether a request goes through the direct client or the managed-unblocking client, so this is
a configuration difference, not an architectural one.
