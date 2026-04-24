# 🛍️ Kapruka Gift-Concierge Agent
> **AEE Bootcamp · Mini Project 03** — Solving "Gifting Chaos" through Agentic Design Patterns & Cognitive Memory Systems.

---

## 📖 Overview

The **Kapruka TEAM Agent** (Tiered, Enriched, Agentic Memory) is an intelligent, multi-agent gift concierge system purpose-built for [kapruka.com](https://kapruka.com) — Sri Lanka's largest online gift store.

It eliminates the core failure of traditional chatbots — **statelessness** — by combining a custom three-tier memory stack, a RAG-powered product catalog, specialist agent routing, and a safety reflection loop that strictly prevents allergen-unsafe recommendations from ever reaching the customer.

---

## ✨ Key Features

- **Live Data Pipeline** — A headless Playwright crawler scrapes live product data from kapruka.com, which is then passed through an LLM enrichment step to standardise prices, categories, and inject `SAFETY:` allergen tags into every product description.
- **3-Tier Cognitive Memory Stack:**
  - **Tier 1 · Semantic Memory** — Persistent JSON recipient profiles (`profiles.json`) tracking per-recipient likes, dislikes, allergies, and past orders.
  - **Tier 2 · LT-RAG** — Qdrant vector store ingesting the enriched product catalog for semantic similarity search.
  - **Tier 3 · Short-Term Buffer** — Session-scoped sliding-window conversation buffer that injects recent dialogue turns into every LLM prompt.
- **Specialist Orchestration** — An LLM-based intent router classifies each message into one of four lanes (`[CATALOG]`, `[LOGISTICS]`, `[PREFERENCE]`, `[CHITCHAT]`) and dispatches to the correct specialist, keeping each LLM call focused and cheap.
- **Safety Reflection Loop** — A strict Draft → Reflect → Revise pipeline critiques every catalog recommendation against the recipient's saved allergy profile before a response is returned to the user.
- **Streamlit Concierge UI** — A WhatsApp-inspired chat interface with dark/light mode, occasion and budget filters, quick-reply buttons, semantic memory status indicators, and live allergen alerts.

---

## 📂 Repository Structure

```
Kapruka-Gift-Concierge-Agent/
├── agents/
│   ├── catalog_agent.py          # RAG-powered gift recommendation specialist
│   ├── chitchat_agent.py         # Warm, culturally aware Sri Lankan greeter
│   ├── logistics_agent.py        # Island-wide delivery coverage specialist
│   ├── preference_agent.py       # Extracts and persists recipient facts
│   ├── reflection_loop.py        # Draft → Reflect → Revise safety reviewer
│   └── router.py                 # Intent classifier → [CATALOG|LOGISTICS|PREFERENCE|CHITCHAT]
├── config/
│   ├── model.yaml                # LLM provider routing and model tiers
│   └── params.yaml               # App parameters, RAG thresholds, paths
├── data/
│   ├── catalog.json              # Enriched production product database
│   ├── catalog_pre_patch_backup.json  # Raw scraped data backup
│   ├── logistics_policy.txt      # Shipping rules and island-wide delivery fees
│   └── profiles.json             # Semantic memory (recipient preferences & allergies)
├── memory/
│   ├── session_buffer.py         # Tier 3: sliding-window conversation ring buffer
│   └── vector_db.py              # Tier 2: Qdrant ingestion, LLM tagging, and retrieval
├── notebooks/
│   ├── 01_the_playwright_crawler.ipynb
│   ├── 02_cognitive_memory_lab.ipynb
│   ├── 03_specialist_orchestration_04_the_reflection_loop.ipynb
│   └── 05_performance_and_proposal.ipynb
├── reports/
│   └── Kapruka_TEAM_Agent_Technical_Proposal.pdf # Final technical proposal and architecture design
├── scraper/
│   ├── kapruka_crawler.py        # Playwright headless browser bot
│   └── clean_and_patch.py        # LLM normaliser for raw crawled data
├── tests/
│   ├── test_catalog_agent.py
│   ├── test_data_validation.py
│   ├── test_integration_flow.py
│   ├── test_logistics_agent.py
│   ├── test_preference_agent.py
│   ├── test_reflection_loop.py
│   └── test_router.py
├── ui/
│   └── app.py                    # Streamlit Concierge frontend
├── utils/
│   ├── config.py                 # YAML config reader + API key resolver
│   └── llm_services.py           # Shared LLM client factory
├── .env .example                 # API key template
├── .gitignore
├── pyproject.toml
├── uv.lock                       # Dependency lockfile
└── README.md
```

---

## 🏗️ Architecture

```
 User Input
     │
     ▼
 ┌─────────┐
 │  Router │ — classifies intent: [CATALOG] | [LOGISTICS] | [PREFERENCE] | [CHITCHAT]
 └────┬────┘
      │
      ├──[CATALOG]────► Catalog Agent (Qdrant RAG + Tier 1 Semantic Profile)
      │                        │
      │                        ▼
      │                Reflection Loop ◄── Recipient Allergy Profile
      │                Draft → Reflect → Revise
      │
      ├──[LOGISTICS]──► Logistics Agent (Sri Lankan district delivery rules)
      │
      ├──[PREFERENCE]─► Preference Updater (writes facts to profiles.json)
      │
      └──[CHITCHAT]───► Chit-Chat Agent (culturally warm Sri Lankan greeting)
                               │
                        SessionBuffer ← response appended
```

### Memory Stack

```
┌─────────────────────────────────┐
│   Tier 3 · Short-Term Buffer    │  ← Session ring buffer (last 5 turn-pairs)
├─────────────────────────────────┤
│   Tier 2 · LT-RAG (Qdrant)     │  ← Vectorised product catalog (persistent)
├─────────────────────────────────┤
│   Tier 1 · Semantic Memory      │  ← JSON recipient profiles (persistent)
└─────────────────────────────────┘
```

---

## 📓 Notebooks

| Notebook | Purpose | Marks |
|---|---|---|
| `01_the_playwright_crawler.ipynb` | Scrape → LLM Enrich → Audit `catalog.json`. Verifies integer prices, URL preservation, and `SAFETY:` tag coverage. | Part 1 · 15 |
| `02_cognitive_memory_lab.ipynb` | Validate all three memory tiers: load a Tier 1 recipient profile, ingest the catalog into Qdrant (Tier 2), and test the Tier 3 session ring buffer. | Part 2 · 30 |
| `03_specialist_orchestration_04_the_reflection_loop.ipynb` | Wire router + all specialists + the reflection safety loop into a single `process_user_message()` pipeline and run a 5-turn multi-intent conversation test. | Parts 3 & 4 · 45 |
| `05_performance_and_proposal.ipynb` | Performance benchmarking, deployment architecture, and generation of the final Technical Proposal. | Part 5 & 6 · 10 |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- A running Qdrant instance (local file-based path, no server needed — configured in `config/params.yaml`)
- At least one LLM provider API key (Groq recommended for speed)

### 1 · Clone & Install

```bash
git clone https://github.com/rumeshmohan/Kapruka-Gift-Concierge-Agent.git
cd Kapruka-Gift-Concierge-Agent
pip install .
```

### 2 · Environment Setup

```bash
cp ".env .example" .env
```

Open `.env` and fill in your API keys. Only the keys for your chosen providers are required:

```
GROQ_API_KEY=your_groq_key          # Recommended — fast and free tier available
COHERE_API_KEY=your_cohere_key      # Required for Cohere embeddings (default)
OPENAI_API_KEY=your_openai_key      # Optional
OPENROUTER_API_KEY=your_or_key      # Optional
GEMINI_API_KEY=your_gemini_key      # Optional
```

> ⚠️ Never commit your `.env` file. It is listed in `.gitignore`.

### 3 · Configure Your Provider

Edit `config/params.yaml` to set your preferred LLM provider:

```yaml
provider:
  default: groq   # Options: groq, openai, cohere, gemini, openrouter, ollama
```

### 4 · Data Pipeline (first-time only)

If you want to pull fresh live data, run the crawler and enrichment pipeline:

```bash
python scraper/kapruka_crawler.py
python scraper/clean_and_patch.py
```

This produces `data/catalog.json` — the enriched product database.

### 5 · Ingest into Qdrant

```bash
python memory/vector_db.py
```

This embeds all products, generates LLM metadata tags, and populates the local Qdrant vector store at `./qdrant_db`.

### 6 · Launch the UI

```bash
streamlit run ui/app.py
```

---

## ⚙️ Configuration

### `config/params.yaml` — Runtime Parameters

| Key | Default | Description |
|---|---|---|
| `provider.default` | `groq` | Active LLM provider |
| `embedding.provider` | `cohere` | Provider used for vector embeddings |
| `rag.score_threshold` | `0.35` | Minimum cosine similarity for RAG results |
| `rag.max_k` | `5` | Maximum products returned per query |
| `rag.collection_name` | `kapruka_catalog` | Qdrant collection name |
| `llm.temperature` | `0.2` | Sampling temperature for all LLM calls |
| `llm.max_tokens` | `256` | Max response tokens |

### `config/model.yaml` — Model Routing

Maps every supported provider to `general`, `strong`, and `reason` chat tiers, plus embedding tiers. The `general` tier is used for most agents; the `strong` tier is reserved for the safety reflection loop.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Individual test files cover each agent, the router, data validation, and the full integration flow.

---

## 🔒 Safety Design

The system enforces a two-layer allergen safety net:

1. **Reflection Loop** (`agents/reflection_loop.py`) — A dedicated `strong`-tier LLM reviews every catalog draft against the recipient's saved allergy profile before the response is returned. If a violation is found it outputs a `REVISED:` response explaining the specific blocked ingredient.

2. **UI Allergen Banner** (`ui/app.py`) — Every assistant message is scanned client-side for allergen keywords. If detected outside a safe context (e.g., "nut-free", "allergy-friendly"), a live warning banner is shown.

---

## 🌏 Sri Lankan Delivery Coverage

Delivery zones and fees are governed by `data/logistics_policy.txt`:

| Zone | Fee | Delivery Time |
|---|---|---|
| Colombo City (Col 1–15) | Rs. 200 | Same/Next Day |
| Greater Colombo & Gampaha | Rs. 300 | 1–2 working days |
| Outstation Major Cities (Kandy, Galle…) | Rs. 400 | 2–3 working days |
| Remote Districts (Jaffna, Nuwara Eliya…) | Rs. 500 | 3–5 working days |
| Any order over Rs. 10,000 | **Free** | Per zone above |

> ⚠️ Fresh cakes and flowers can only be delivered to Colombo, Gampaha, and Kandy districts.

---

## 🤝 Supported LLM Providers

| Provider | Chat | Embeddings |
|---|---|---|
| Groq | ✅ | ❌ |
| Cohere | ✅ | ✅ |
| OpenAI | ✅ | ✅ |
| Google Gemini | ✅ | ✅ |
| OpenRouter | ✅ | ✅ |
| Ollama (local) | ✅ | ❌ |
| DeepSeek | ✅ | ❌ |

All providers are accessed through an OpenAI-compatible API interface — switching providers requires only a change to `config/params.yaml`, no code changes.

---

## ⚠️ Disclaimer

This is a pre-development prototype built for the AEE Bootcamp · Mini Project 03. It is not affiliated with or endorsed by Kapruka Holdings Ltd. Product data is scraped for educational purposes only.

---

*Built by Zuu Crew AI · AI Engineer Essentials Bootcamp*