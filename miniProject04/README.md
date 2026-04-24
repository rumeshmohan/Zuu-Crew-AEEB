# Mini Project 04 — Multi-Agent NL2SQL Platform for Nawaloka Hospital

**Course:** AI Engineer Essentials — Module: Multi-Agent Workflows
**Total Points:** 100 (+5 Bonus)

---

## Overview

This project implements a production-grade, multi-agent Natural Language to SQL (NL2SQL) analytics platform for Nawaloka Hospital. Hospital operations staff can ask plain-English questions about patient records, billing, doctor schedules, and departmental performance — and receive live database results, visualisations, and natural language insights — without writing a single line of SQL.

---

## System Architecture

```mermaid
flowchart TD
    User["👤 User / Streamlit UI"]
    Router["🔀 Intent Router Agent<br/>Classifies: sql_generation vs general_chat"]
    Chat["💬 General Chat Response"]
    SQLGen["⚙️ SQL Generator Agent<br/>3-attempt retry loop"]
    Validator["🛡️ SQL Validator<br/>Blocks DROP/DELETE/TRUNCATE<br/>Enforces SELECT-only"]
    DB["🗄️ MediCore PostgreSQL<br/>12-table Hospital CRM"]
    Interpreter["📊 Result Interpreter Agent<br/>Natural language summary<br/>+ chart specification"]
    Fallback["⚠️ Fallback Agent<br/>Graceful error messages"]
    LangFuse["🔭 LangFuse<br/>Observability & Tracing"]
    Output["📈 Result: Summary + Chart + SQL"]

    User -->|"Natural language query"| Router
    Router -->|general_chat| Chat
    Router -->|sql_generation| SQLGen
    SQLGen -->|"Generated SQL"| Validator
    Validator -->|"Validation failed × 3"| Fallback
    Validator -->|"Valid SQL"| DB
    DB -->|"DB execution error"| Fallback
    DB -->|"Raw rows"| Interpreter
    Interpreter --> Output
    Fallback --> Output

    Router -.->|"@observe"| LangFuse
    SQLGen -.->|"@observe"| LangFuse
    Interpreter -.->|"@observe"| LangFuse
    Fallback -.->|"@observe"| LangFuse
```

The pipeline has **four specialist agents** coordinated by a central orchestrator:

| Agent | Role |
|-------|------|
| **Intent Router** | Classifies incoming messages as `sql_generation` or `general_chat` |
| **SQL Generator** | Translates natural language to PostgreSQL with a 3-attempt retry loop |
| **Result Interpreter** | Converts raw DB rows into a plain-English summary and chart spec |
| **Fallback Agent** | Returns user-friendly messages when validation or DB execution fails |

---

## Project Structure

```
nawaloka-nl2sql-platform/
├── config/
│   ├── model.yaml          # LLM provider and model configuration
│   └── params.yaml         # Runtime parameters (temperature, provider, paths)
├── data/
│   └── medicore_data.sql   # Database dump for seeding (ignored in git)
├── report/
│   ├── engineering_report.md # Technical documentation
│   └── Nawaloka-Hospital-NL2SQL-Platform Engineering Report.pdf
├── scripts/
│   ├── download_traces.py  # Downloads LangFuse traces to traces/
│   └── seed_supabase.py    # Seeds the Supabase database from medicore_data.sql
├── src/
│   ├── agents/
│   │   ├── fallback_agent.py     # Graceful error recovery
│   │   ├── interpreter_agent.py  # Result → NL summary + chart spec
│   │   ├── router_agent.py       # Intent classification
│   │   └── sql_agent.py          # NL → SQL generation with retry
│   ├── dashboard/
│   │   └── app.py                # Streamlit chatbot + pre-built dashboard
│   ├── engine/
│   │   ├── db_client.py          # SQLAlchemy DB connection + dynamic schema
│   │   ├── orchestrator.py       # Main pipeline controller
│   │   ├── prompt_builder.py     # Dynamic schema-aware prompt construction
│   │   └── sql_validator.py      # Safety layer — blocks destructive SQL
│   └── utils/
│       ├── config.py             # Config singleton (params.yaml + model.yaml)
│       ├── llm_services.py       # Unified LLM interface (8 providers)
│       └── observability.py      # LangFuse client initialisation
├── tests/
│   └── test_validator.py         # Unit tests for SQL Validator
├── traces/                       # LangFuse trace samples (3 representative queries)
├── .env example                  # Template for environment variables
├── .gitignore                    # Git tracking ignore rules
├── .python-version               # Python version specification
├── docker-compose.yml            # Local PostgreSQL via Docker
├── Makefile                      # CLI commands for project tasks
├── pyproject.toml                # Project metadata and dependencies
└── uv.lock                       # Deterministic dependency lockfile
```

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker Desktop (for local DB)
- A PostgreSQL database (local Docker or Supabase)

### 1. Clone and install dependencies

```bash
git clone https://github.com/rumeshmohan/nawaloka-nl2sql-platform.git
cd nawaloka-nl2sql-platform
uv sync
```

### 2. Configure environment variables

```bash
cp ".env example" .env
```

Edit `.env` and fill in your credentials:

```env
# Required
DB_CONNECTION_STRING=postgresql://user:password@host:5432/medicore

# LLM Provider — only the one you use is needed
GEMINI_API_KEY=your_key_here

# Observability
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

### 3. Start the local database (optional)

If you are not using Supabase, spin up PostgreSQL locally:

```bash
docker compose up -d
```

Then seed the database:

```bash
uv run python scripts/seed_supabase.py
```

### 4. Select your LLM provider

Edit `config/params.yaml`:

```yaml
provider:
  default: gemini   # Options: gemini, openai, groq, anthropic, deepseek, mistral, cohere, ollama
  tier: general
```

### 5. Run the Streamlit dashboard

```bash
uv run streamlit run src/dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Running Tests

```bash
uv run python -m pytest tests/ -v
```

---

## Query Trace Analysis

All three traces were captured from LangFuse during live testing against the MediCore database (50,000 patients, 20 departments). Full JSON files are in the `traces/` directory.

---

### Trace 1 — Simple Query

**Natural Language:** `How many patients are there?`
**Trace ID:** `97dea9e15d4aa74ce2ecb237a24b4140`
**Timestamp:** 2026-04-24T18:54:49Z | **Latency:** 9.071 s | **Agents invoked:** 4

**Generated SQL:**
```sql
SELECT COUNT(patient_id) FROM patients
```

**Validation:** Passed on first attempt. Query begins with `SELECT`, contains no forbidden keywords, and has balanced parentheses.

**Output:** `{ "count": 50000 }`

**Interpretation:** *"There are 50,000 patients in the database."* — Chart type: `metric` (single scalar value).

**Notes:** This is the ideal happy-path trace. The router correctly classified it as `sql_generation`, the SQL generator produced a minimal single-table aggregate query with no joins, and the interpreter chose a `metric` display widget rather than a chart. Total latency of 9 seconds reflects the round trip through four agents and the LLM.

---

### Trace 2 — Complex Query

**Natural Language:** `What is the total billed amount per department?`
**Trace ID:** `9b32ffc4919ef0911819c8b267afffda`
**Timestamp:** 2026-04-24T18:55:58Z | **Latency:** 7.413 s | **Agents invoked:** 4

**Generated SQL:**
```sql
SELECT T2.department_name, SUM(T1.total_amount) AS total_billed_amount
FROM billing_invoices AS T1
INNER JOIN admissions AS T3 ON T1.admission_id = T3.admission_id
INNER JOIN departments AS T2 ON T3.department_id = T2.department_id
GROUP BY T2.department_name
```

**Validation:** Passed on first attempt. The query correctly navigates a three-table join chain (`billing_invoices → admissions → departments`) using foreign keys from the dynamic schema.

**Output:** 20 department rows with `total_billed_amount` values as Decimal.

**Interpretation:** ENT had the highest total billed amount at $279,535,010.34 while Pediatrics had the lowest at $226,406,666.56. Chart type: `bar` (x: `department_name`, y: `total_billed_amount`).

**Notes:** This trace demonstrates the system's core strength — the dynamic schema injected into the prompt included the FK relationships between `billing_invoices.admission_id → admissions.admission_id` and `admissions.department_id → departments.department_id`, which the SQL Generator used to construct the correct two-join query without any human guidance. The lower latency (7.4 s vs 9.1 s for the simple trace) is likely due to LLM response caching and reduced token count on the second query.

---

### Trace 3 — Failed Query (Fallback Triggered)

**Natural Language:** `Show me all the records from the super_secret_imaginary_table`
**Trace ID:** `a566acc86d45100f375966eb1cb6fd3b`
**Timestamp:** 2026-04-24T18:55:17Z | **Latency:** 6.535 s | **Agents invoked:** 2

**Generated SQL:**
```sql
SELECT * FROM super_secret_imaginary_table
```

**Validation:** Passed (syntactically valid SELECT, no forbidden keywords). The validator correctly allows the query through — it is not the validator's responsibility to know which tables exist, only to block destructive commands.

**DB Execution:** Failed with `psycopg2.errors.UndefinedTable: relation "super_secret_imaginary_table" does not exist`.

**Fallback Response:** *"I generated the query, but the database rejected it. This usually happens if the data structure is a bit different than expected."*

**Notes:** This trace confirms the `db_execution` error branch in the orchestrator works correctly. The user receives a friendly, actionable message instead of a raw psycopg2 stack trace. Only 2 agents were invoked (Router + SQL Generator), as the pipeline short-circuits at the DB error before reaching the Interpreter. A production improvement here would be to detect table-not-found errors specifically and respond with *"That table doesn't exist — did you mean one of: patients, doctors, appointments...?"* using the dynamic schema for suggestions.

---

## Reflection & Production Readiness

### What Went Well

The multi-agent decomposition proved highly effective. By separating intent routing, SQL generation, validation, execution, and interpretation into distinct agents, each component became independently testable and debuggable. When the interpreter agent produced malformed JSON in early testing, the issue was isolated immediately to that agent's prompt rather than requiring debugging of the entire pipeline. The dynamic schema injection approach — fetching live table structures and sample rows from the database on each request — eliminated the fragile hardcoded schema strings seen in simpler NL2SQL prototypes and produced noticeably better JOIN accuracy. The LangFuse `@observe` decorators added zero application logic but gave full visibility into latency, token usage, and input/output for every agent call — invaluable during prompt iteration.

### What Was Challenging

The SQL Generator's 3-attempt retry loop handles syntactic validation failures well, but semantic failures (asking about a non-existent table, using a plausible but wrong column name) pass the validator and fail only at execution time. This means the fallback fires *after* a DB round trip rather than before. A schema-aware semantic validator — one that checks generated column names against the inspector output before executing — would catch these earlier and produce better clarification messages.

Latency is the other main challenge. At 7–9 seconds per query across four agent calls, the system is functional but not snappy for a dashboard context. The Gemini Flash model was chosen specifically for its speed-to-quality balance, but each query still requires three or four sequential LLM round trips. Parallelising the Interpreter call with any pre-computation (e.g., pre-fetching chart data) would reduce perceived latency.

### Scaling to 10,000 Queries per Day

The current synchronous architecture will not handle high query volumes without modification. At 10,000 queries per day (~7 queries per minute on average, with peak bursts significantly higher), three changes are essential:

**1. Async pipeline.** Replace the synchronous orchestrator with an async implementation using Python's `asyncio` and an async DB driver (`asyncpg` instead of `psycopg2`). This allows the Streamlit server to handle concurrent requests without blocking.

**2. Query result caching.** Many hospital analytics queries are identical or near-identical (e.g., the same "Revenue by Department" dashboard panel refreshed multiple times per hour). A Redis cache keyed on the normalised natural language query and the current date would serve repeated queries in milliseconds with zero LLM cost.

**3. Connection pooling.** Replace the single SQLAlchemy `create_engine` with a pooled connection manager (`pool_size=20, max_overflow=10`) to handle burst traffic without exhausting the database's connection limit.

### Role-Based Access Control (RBAC)

In its current form, any user of the Streamlit app can query any of the 12 tables including sensitive patient records (`patients`, `diagnoses`, `prescriptions`). Production deployment at a hospital requires RBAC at two levels:

- **Application level:** The router agent's system prompt should be extended with the current user's role and a permitted-tables whitelist. A ward nurse would have access to `patients`, `admissions`, and `appointments` but not `billing_invoices` or `payments`. The SQL Generator is then instructed to refuse queries that touch out-of-scope tables.
- **Database level:** Each role should correspond to a PostgreSQL role with column-level GRANT restrictions. Even if the application layer is bypassed, the DB user cannot return sensitive columns (e.g., `patients.national_id`, `billing_invoices.total_amount` for non-finance roles).

### SQL Injection Mitigation

The current validator blocks destructive DDL/DML commands and enforces `SELECT`-only queries, which covers the most dangerous attack surface. However, a sophisticated prompt injection attack — embedding SQL inside the natural language query itself — could still produce unintended queries if the LLM echoes user input directly into the SQL. Mitigations for production:

- **Parameterised execution:** While dynamic SQL generation inherently cannot use bound parameters for table/column names, the DB client should use `sqlalchemy.text()` with bound parameters for any literal values extracted from the query (e.g., date ranges, names).
- **Query allowlisting:** For high-security deployments, maintain a library of pre-approved query templates and use the LLM only to fill in parameters (dates, department names, numeric thresholds) rather than generating freeform SQL.
- **Input sanitisation:** Strip or escape characters like `--`, `;`, `/*`, and `*/` from user input before passing to the SQL Generator to reduce the surface area for injection attempts.

### Retrospective

If restarting this project, two architectural decisions would change. First, the Streamlit dashboard and the NL2SQL pipeline are tightly coupled — the dashboard imports directly from `src/engine/orchestrator.py`. Exposing the orchestrator as a FastAPI REST endpoint would allow the dashboard, a mobile app, and third-party integrations to all consume the same pipeline independently. Second, the retry logic in `sql_agent.py` feeds the validation error message back to the LLM as a correction prompt, which works well for syntax errors but poorly for semantic errors where the model repeats the same mistake. A self-reflection step — asking the model to critique its own SQL against the schema before submitting — would reduce the number of retries reaching the DB layer.

Overall the project successfully demonstrates that a multi-agent decomposition, dynamic schema injection, and structured observability can produce a reliable NL2SQL system that handles ambiguity, blocks dangerous operations, and degrades gracefully on failure — exactly the production readiness criteria a real hospital deployment would require.