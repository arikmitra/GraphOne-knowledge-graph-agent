# GraphOne / FrontierAtlas — AI Intelligence Graph Ingestion Pipeline

A working, async, multi-source scraping + LLM-extraction pipeline for building an AI
industry knowledge graph: startups, products, research papers, jobs, and news — with
entity resolution (including an optional web-verification fallback for ambiguous names)
and full audit logging. Built as a take-home deliverable for the GraphOne/FrontierAtlas
AI Engineer trial (see `architecture.md` for the design writeup).

## What's here

```
graphone-pipeline/
├── src/
│   ├── schemas/models.py        # Pydantic output schemas (Startup/Product/Paper/Job/News)
│   ├── utils/
│   │   ├── http.py              # Async HTTP client: per-domain rate limits, retry/backoff
│   │   ├── dates.py             # Relative-date parsing + 24h freshness classification
│   │   ├── writers.py           # JSONL + CSV (Sheets-tab-shaped) output writers
│   │   └── logging_setup.py     # Structured run-scoped logging
│   ├── llm/
│   │   ├── orchestrator.py      # Multi-tier fallback chain + context-budget chunking
│   │   └── providers.py         # Real adapters: Gemini Flash / Groq Llama 3 / DeepSeek
│   ├── resolver/
│   │   ├── entity_resolver.py   # Exact/alias/fuzzy entity resolution + audit log
│   │   ├── web_verifier.py      # Optional web-search + LLM fallback for ambiguous names
│   │   └── search_adapters.py   # Real Brave Search + LLM-chain adapters, wired into pipeline.py
│   ├── scrapers/
│   │   ├── research_papers.py   # Arxiv API + GitHub stars correlation
│   │   ├── startups_products.py # HN "Show HN" bulk scraper (directory pattern)
│   │   ├── news_jobs.py         # RSS/Atom multi-source, 24h freshness filtering
│   │   └── anti_bot.py          # Playwright stealth browser pool (Phase V)
│   └── pipeline.py              # End-to-end CLI entry point
├── tests/                       # 58 unit + integration tests (pytest + pytest-asyncio)
├── demo_offline_run.py          # Full pipeline demo with realistic sample data, no network
├── demo_web_verification.py     # Demo wiring real search + LLM into entity resolution
├── test_stealth.py              # Phase V smoke test against a bot-detection test page
├── architecture.md / .pdf / .docx  # Phase VI writeup (scale, 413/429, freshness, storage)
└── requirements.txt
```

## Setup

**1. Clone/place the repo and create a virtual environment** (recommended so dependencies
don't clash with other Python projects on your machine):
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Install the Playwright browser** (only needed for the anti-bot / Phase V path —
`test_stealth.py` and `src/scrapers/anti_bot.py`; skip this if you're not touching that part):
```bash
playwright install chromium
```

**4. (Optional) Configure API keys.** Every key below is optional and independently
degradable — the pipeline runs with zero keys set, just with reduced capability:

```bash
export GEMINI_API_KEY=...      # LLM extraction tier 1
export GROQ_API_KEY=...        # LLM extraction tier 2 (fallback)
export DEEPSEEK_API_KEY=...    # LLM extraction tier 3 (fallback)
export GITHUB_TOKEN=...        # raises GitHub API limit from 60/hr to 5000/hr
export BRAVE_SEARCH_API_KEY=...  # enables --web-verify-entities (see below)
```
- If **no** `GEMINI_API_KEY` / `GROQ_API_KEY` / `DEEPSEEK_API_KEY` is set, the LLM
  orchestrator's fallback chain has nothing to fall through to, and extraction calls will
  report failure per-record (logged, not crashed) — everything else in the pipeline still runs.
- If **no** `GITHUB_TOKEN` is set, GitHub star-count enrichment for research papers is capped
  at 60 requests/hour (public rate limit) instead of 5000/hour.
- If **no** `BRAVE_SEARCH_API_KEY` is set, `--web-verify-entities` silently has no effect
  (logged as a warning) — entity resolution falls back to deterministic-only matching.

## Running it

### Option A — Offline demo (no network, no API keys, run this first)

The fastest way to confirm the codebase works in your environment at all:
```bash
python demo_offline_run.py
```
This exercises the real schema validation, date normalization, entity resolution (including
the "OpenAI" / "Open AI" / "OpenAI, Inc." canonicalization case from the assignment spec), and
LLM fallback chain logic against realistic sample data baked into the script — no scraping, no
network calls, no API keys required. Expect output like:
```
=== OFFLINE DEMO RUN COMPLETE ===
Startups written:        5
Products written:        3
...
```
If this doesn't run cleanly, something is wrong with the Python environment/dependencies
itself, before you even get to live sources — fix that first.

### Option B — Live pipeline run against real sources

```bash
python -m src.pipeline --startups 50 --products 50 --papers 50 --output-dir data/output
```
Start with small numbers (as above) for your first run — it's faster to debug a network or
credentials issue against 50 records than 1000. This pages through Arxiv, Hacker News
(Show HN), and 5 news + 5 job RSS feeds concurrently, runs entity resolution, validates every
record against the Pydantic schemas, and writes JSONL + CSV outputs to `data/output/`.

Once the small run succeeds, scale up:
```bash
python -m src.pipeline --startups 1000 --products 1000 --papers 1000 --output-dir data/output
```

Flags:
| Flag | Default | Meaning |
|---|---|---|
| `--startups N` | 50 | Number of startup records to collect |
| `--products N` | 50 | Number of product records to collect |
| `--papers N` | 50 | Number of research paper records to collect |
| `--no-news` | (off) | Skip the news-scraping phase entirely |
| `--no-jobs` | (off) | Skip the job-scraping phase entirely |
| `--output-dir PATH` | `data/output` | Where JSONL/CSV outputs are written |
| `--github-token TOKEN` | none | GitHub token for higher-rate star-count lookups |
| `--web-verify-entities` | (off) | Enable the web-verification fallback (see Option C) |

**What to check after a run:**
1. Look at the console log — each phase prints a summary line (e.g. `Phase I done: 42
   startups, 50 products`).
2. Open `data/output/run_stats.json` — a machine-readable summary of every phase's duration
   and record count, plus `rejects` (records that failed schema validation) and
   `entity_resolutions` (total resolver decisions made).
3. Check `data/output/rejects.jsonl` — should be empty or small; each line is a record that
   failed validation with the error attached. A large reject count signals a scraper or schema
   mismatch worth investigating.
4. Spot-check `data/output/entity_mapping_log.jsonl` (or the CSV equivalent under
   `data/output/csv/Entity_Mapping_Log.csv`) — confirms name resolution is behaving sensibly
   (e.g. no two clearly-different companies merged together).

### Option C — Live pipeline run with web-verified entity resolution

```bash
export BRAVE_SEARCH_API_KEY=...
python -m src.pipeline --startups 50 --products 50 --papers 50 --web-verify-entities
```
Same as Option B, but every ambiguous entity name (a fuzzy match too weak to auto-accept, or
no match at all) is additionally checked via a real web search + LLM judgment call before
being minted as a new canonical entity. This is strictly opt-in: omit the flag, or leave
`BRAVE_SEARCH_API_KEY` unset, and behavior is identical to Option B.

When enabled, an extra `data/output/web_verification_log.jsonl` file is written, containing
only the entries that actually went through verification — inspect it to see what the
search/LLM call found and why it did or didn't merge:
```bash
cat data/output/web_verification_log.jsonl | python -m json.tool
```
`run_stats.json` also reports `"web_verification_enabled"` (true/false) and
`"web_verification_calls"` (how many ambiguous names were actually checked).

### Option D — Standalone demo of web verification only

If you just want to see the web-verification mechanism in isolation, without running the full
pipeline:
```bash
export BRAVE_SEARCH_API_KEY=...
export GEMINI_API_KEY=...   # or GROQ_API_KEY / DEEPSEEK_API_KEY
python demo_web_verification.py
```
This resolves a small fixed list of names — including a deliberately ambiguous rebrand case
("Bard AI" → should resolve to an existing canonical entity) — and prints the full resolution
log with evidence, so you can see exactly what the search returned and how the LLM judged it
without needing a full pipeline run.

### Option E — Phase V anti-bot smoke test

```bash
python test_stealth.py
```
Fetches a bot-detection test page (`bot.sannysoft.com`) through the stealth Playwright pool
and saves the rendered HTML to `stealth_check.html`. Open that file in a normal browser and
check that rows like "WebDriver", "Chrome", and "Plugins Length" read green/pass — this
confirms the stealth measures (patched `navigator.webdriver`, realistic headers, human-paced
delays) are actually taking effect against a real detection script, not just running silently.

## Testing

**Run the full suite:**
```bash
pytest tests/ -v
```
This requires no network access and no API keys — every test uses mocked HTTP responses,
mocked LLM/search functions, or pure in-memory logic. Expect all 58 tests to pass in well
under a minute.

**Run a single test file** (useful when iterating on one part of the system):
```bash
pytest tests/test_entity_resolver.py -v          # deterministic entity resolution
pytest tests/test_web_verification.py -v         # ambiguous-name web-verification routing
pytest tests/test_llm_web_verifier.py -v         # the search+LLM verifier implementation
pytest tests/test_llm_orchestrator.py -v         # LLM fallback chain + chunking
pytest tests/test_http_domain_policy.py -v       # per-domain rate limiting
pytest tests/test_dates.py -v                    # date parsing + freshness classification
pytest tests/test_schemas.py -v                  # Pydantic schema validation
pytest tests/test_pipeline_web_verification_integration.py -v   # end-to-end wiring check
```

**Run a single test by name** (useful when debugging one failure):
```bash
pytest tests/test_entity_resolver.py::test_the_openai_example_from_spec -v
```

**What the suite covers, by file:**
| File | What it tests |
|---|---|
| `test_dates.py` | ISO/relative/missing date parsing, 24h freshness window, content-hash dedup |
| `test_entity_resolver.py` | Exact/alias/fuzzy matching, the assignment's "OpenAI" vs "Open AI" example, international legal suffixes |
| `test_http_domain_policy.py` | Per-domain rate limiting, including sub-1-req/s policies (e.g. Arxiv's 0.33 req/s) |
| `test_llm_orchestrator.py` | Fallback chain cascading across providers, context-budget chunking, malformed-JSON recovery |
| `test_llm_web_verifier.py` | The search+LLM verifier: happy path, search/LLM failures, hallucinated-suggestion guard, snippet capping |
| `test_schemas.py` | Pydantic validation — required fields, value constraints (e.g. no negative star counts) |
| `test_web_verification.py` | `EntityResolver.resolve_async`'s ambiguous-band routing, confidence thresholds, raw-name caching |
| `test_pipeline_web_verification_integration.py` | End-to-end: default behavior unchanged, graceful degradation without a key, a real verification call flowing through to output files |

**If a test fails:** the failure message includes the assertion and actual vs. expected
values; re-run just that test with `-v` (shown above) for the full traceback. None of the
tests depend on external services, so a failure indicates either an environment issue (wrong
Python version, missing dependency) or an actual regression — not a flaky network call.

## Design notes worth knowing before you read the code

- **Anti-hallucination by construction**: every record carries `source.url`,
  `rawContentHash`, and `extractionModel` — there is no path from "LLM output" to "written
  record" that skips schema validation, and records that fail validation go to
  `rejects.jsonl` with the error attached rather than being silently dropped or coerced.
- **The LLM fallback chain is provider-agnostic.** `LLMProvider` is a 5-line ABC; swapping in
  a different vendor or adding a 4th tier is adding one adapter class, not touching the
  orchestrator.
- **Entity resolution is auditable, not a black box.** Every resolution decision (exact match,
  curated alias, fuzzy match + score, web-verified merge, or "this is a new canonical entity")
  is logged to `entity_mapping_log.jsonl`, which is also the deliverable "Entity Mapping Log"
  tab. Web-verification decisions additionally carry an `evidence` string explaining what the
  search/LLM call found, retrievable via `resolver.evidence_for(log_index)`.
- **Web verification is a conservative, opt-in fallback, not a default.** It only fires when
  deterministic matching is genuinely ambiguous (a mid-range fuzzy score, or no candidate at
  all), and only auto-merges above an 85% confidence threshold — anything less is logged as
  `web_suggested_low_confidence` for human review rather than silently merged. A verifier
  failure (network error, bad JSON, etc.) degrades to "mint as new + log the failure," and
  never crashes the pipeline. Wired into the live pipeline via `--web-verify-entities` (see
  "Running it" above); see `src/resolver/web_verifier.py` and `src/resolver/search_adapters.py`
  for the full contract and the concrete Brave Search + LLM-chain adapter.
- **Scaling from 1,000 to 500,000 records is a flag change, not a rewrite** — see
  `architecture.md` §1 for how the same pagination-loop code shards horizontally.
