# 🛍️ Kapruka Gift-Concierge Agent
> **AEE Bootcamp · Mini Project 03** — Solving "Gifting Chaos" through Agentic Design Patterns & Cognitive Memory Systems.

---

## 📖 Overview

The **Kapruka TEAM Agent** (Tiered, Enriched, Agentic Memory) is an intelligent, multi-agent gift concierge system purpose-built for [kapruka.com](https://kapruka.com) — Sri Lanka's largest online gift store.

It eliminates the core failure of traditional chatbots — **statelessness** — by combining a custom three-tier memory stack, a CLIP-powered multimodal RAG catalog, specialist agent routing, and a safety reflection loop that strictly prevents allergen-unsafe recommendations from ever reaching the customer.

---

## ✨ Key Features

- **Live Data Pipeline** — A headless Playwright crawler scrapes live product data from kapruka.com, which is then passed through an LLM enrichment step to standardise prices, categories, and inject `SAFETY:` allergen tags into every product description.
- **3-Tier Cognitive Memory Stack:**
  - **Tier 1 · Semantic Memory** — Persistent JSON recipient profiles (`profiles.json`) tracking per-recipient likes, dislikes, allergies, and past orders.
  - **Tier 2 · LT-RAG** — Qdrant vector store ingesting the enriched product catalog using **CLIP dual-vector embeddings (text + image)** for cross-modal semantic search.
  - **Tier 3 · Short-Term Buffer** — Session-scoped sliding-window conversation buffer that injects recent dialogue turns into every LLM prompt.
- **Specialist Orchestration** — An LLM-based intent router classifies each message into one of four lanes (`[CATALOG]`, `[LOGISTICS]`, `[PREFERENCE]`, `[CHITCHAT]`) and dispatches to the correct specialist, keeping each LLM call focused and cheap.
- **Safety Reflection Loop** — A strict Draft → Reflect → Revise pipeline critiques every catalog recommendation against the recipient's saved allergy profile before a response is returned to the user.
- **Multimodal CLIP Embeddings** — Products are embedded in a shared 512-dim vector space using both their text descriptions and product images. A natural-language query like *"something elegant in red"* retrieves visually matching products even when the text description doesn't contain those exact words.
- **Configurable Embedding Backend** — Switch between CLIP (local, no API key) and API providers (Cohere, OpenAI, etc.) via a single line in `params.yaml`. When CLIP is active, choose between `text_only` (fast, no image downloads) or `text_image` (dual-vector cross-modal retrieval).
- **Streamlit Concierge UI** — A WhatsApp-inspired chat interface with dark/light mode, occasion and budget filters, quick-reply buttons, semantic memory status indicators, live allergen alerts, and a **product image grid** that visually renders CLIP-matched results.

---

## 🖼️ Multimodal RAG: CLIP-Powered Shared Vector Space

This project implements the full multimodal retrieval architecture described in the AEE Bootcamp session — a CLIP-powered shared vector space where text queries can match products by visual similarity, not just keyword overlap.

### How it works

```
User query: "red velvet cake"
        │
        ▼
  CLIP Text Encoder  ──►  512-dim query vector
        │
        ├──► Search TEXT collection  (product name + description embeddings)
        │
        └──► Search IMAGE collection (product image embeddings)
                │
                ▼
        Merge + deduplicate results  (text hits take priority)
                │
                ▼
        Top-K products ranked by cosine similarity
                │
                ▼
        LLM generates recommendation + UI renders product image grid
```

### Cross-modal retrieval in action

| Query | How it's matched |
|---|---|
| `"red velvet cake"` | Text vector matches description; image vector matches red-toned visuals |
| `"something elegant in blue packaging"` | Image vector finds visually matching products even if "blue" isn't in the description |
| `"nut-free chocolate birthday cake"` | Text vector matches SAFETY tags; reflection loop verifies against allergy profile |

### Vector storage schema (Qdrant, `text_image` mode)

Each product point is stored with **two named vectors**:

| Vector name | Source | Dimension | Model |
|---|---|---|---|
| `text` | CLIP text encoder on `name + category + description` | 512 | `openai/clip-vit-base-patch32` |
| `image` | CLIP image encoder on `image_url` (downloaded at ingest time) | 512 | `openai/clip-vit-base-patch32` |

Because both encoders project into the same shared embedding space, a text query vector is directly comparable to an image vector — enabling true cross-modal retrieval with no extra bridging model.

### Switching embedding modes

Controlled entirely via `config/params.yaml` — no code changes required:

| `embedding.provider` | `embedding.clip_mode` | Vectors stored | Retrieval strategy |
|---|---|---|---|
| `clip` | `text_image` | `text` (512) + `image` (512) | Dual search — cross-modal |
| `clip` | `text_only` | `text` (512) | Text-only; faster ingest |
| `cohere` / `openai` / … | *(n/a)* | `text` (provider dim) | API embedding, text-only |

---

## 📊 Performance (Notebook 05)

| Metric | Result |
|---|---|
| Products scraped | 10,010 |
| Integer prices | 100% |
| URLs preserved | 100% |
| SAFETY tag coverage | 100% |
| **Crawl Score** | **100% ✅** |
| Preference Alignment Score (allergen safety, 10-case benchmark) | **100% ✅** |

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
│   └── params.yaml               # App parameters, RAG thresholds, embedding mode
├── data/
│   ├── catalog.json              # Enriched production product database (10,010 products)
│   ├── catalog_pre_patch_backup.json  # Raw scraped data backup
│   ├── logistics_policy.txt      # Shipping rules and island-wide delivery fees
│   └── profiles.json             # Semantic memory (recipient preferences & allergies)
├── memory/
│   ├── session_buffer.py         # Tier 3: sliding-window conversation ring buffer
│   └── vector_db.py              # Tier 2: CLIP dual-vector Qdrant ingestion & retrieval
├── notebooks/
│   ├── 01_the_playwright_crawler.ipynb
│   ├── 02_cognitive_memory_lab.ipynb          # ← includes CLIP cross-modal retrieval demo
│   ├── 03_specialist_orchestration_04_the_reflection_loop.ipynb
│   └── 05_performance_and_proposal.ipynb
├── reports/
│   └── Kapruka_TEAM_Agent_Technical_Proposal.pdf
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
├── Makefile                      # Cross-platform task runner (Windows & Linux/macOS)
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
      ├──[CATALOG]────► Catalog Agent (Qdrant CLIP RAG + Tier 1 Semantic Profile)
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
│   Tier 2 · LT-RAG (Qdrant)     │  ← CLIP dual-vector product catalog (persistent)
│                                 │    text collection + image collection (512-dim each)
├─────────────────────────────────┤
│   Tier 1 · Semantic Memory      │  ← JSON recipient profiles (persistent)
└─────────────────────────────────┘
```

---

## 📓 Notebooks

| Notebook | Purpose | Marks |
|---|---|---|
| `01_the_playwright_crawler.ipynb` | Scrape → LLM Enrich → Audit `catalog.json`. Verifies integer prices, URL preservation, and `SAFETY:` tag coverage. | Part 1 · 15 |
| `02_cognitive_memory_lab.ipynb` | Validate all three memory tiers. Includes **CLIP cross-modal retrieval demo**: text query matched against both text and image vector collections, with visual results. | Part 2 · 30 |
| `03_specialist_orchestration_04_the_reflection_loop.ipynb` | Wire router + all specialists + the reflection safety loop into a single `process_user_message()` pipeline and run a 5-turn multi-intent conversation test. | Parts 3 & 4 · 45 |
| `05_performance_and_proposal.ipynb` | Performance benchmarking (crawl score, preference alignment, latency), full unit test report, and generation of the final Technical Proposal. | Part 5 · 10 |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) package manager (`pip install uv`)
- A running Qdrant instance (local file-based — no server needed, configured in `config/params.yaml`)
- At least one LLM provider API key (Groq recommended for speed)
- PyTorch-compatible environment for CLIP embeddings (CPU works; GPU strongly recommended for large ingestion runs)

### 1 · Clone & Install

```bash
git clone https://github.com/rumeshmohan/Kapruka-Gift-Concierge-Agent.git
cd Kapruka-Gift-Concierge-Agent
make install        # runs: uv sync
```

Or without `make`:

```bash
uv sync
```

### 2 · Environment Setup

```bash
make env            # copies .env .example → .env
```

Or manually:

```bash
cp ".env .example" .env
```

Open `.env` and fill in your API keys. Only the keys for your chosen LLM provider are required — CLIP embeddings run **locally** with no API key:

```
GROQ_API_KEY=your_groq_key          # Recommended — fast and free tier available
OPENAI_API_KEY=your_openai_key      # Optional
OPENROUTER_API_KEY=your_or_key      # Optional
GEMINI_API_KEY=your_gemini_key      # Optional
COHERE_API_KEY=your_cohere_key      # Optional
```

> ⚠️ Never commit your `.env` file. It is listed in `.gitignore`.

### 3 · Configure Your Provider

Edit `config/params.yaml` to set your preferred LLM provider and embedding mode:

```yaml
provider:
  default: groq   # Options: groq, openai, cohere, gemini, openrouter, ollama, deepseek

embedding:
  provider: clip         # Use local CLIP (no API key needed)
  clip_mode: text_image  # text_image = dual-vector cross-modal | text_only = faster ingest
```

### 4 · Data Pipeline (first-time only)

Run the full pipeline in one command:

```bash
make pipeline       # crawl + enrich + ingest
```

Or step by step:

```bash
make crawl          # scrape kapruka.com → data/catalog.json (raw)
make enrich         # LLM normalise prices, categories, inject SAFETY: tags
make ingest         # CLIP-embed catalog → qdrant_db/ (resume-safe)
```

> ⚠️ The first `make ingest` run will download the CLIP model (~600 MB). Subsequent runs are resume-safe — already-ingested points are skipped.

### 5 · Launch the UI

```bash
make ui             # runs: streamlit run ui/app.py
```

---

## ⚙️ Configuration

### `config/params.yaml` — Runtime Parameters

| Key | Default | Description |
|---|---|---|
| `provider.default` | `groq` | Active LLM provider for all agents |
| `embedding.provider` | `clip` | Embedding backend: `clip` (local) or an API provider (`cohere`, `openai`, `gemini`, `openrouter`) |
| `embedding.clip_mode` | `text_image` | CLIP only: `text_only` (single text vector) or `text_image` (dual text + image vectors, cross-modal retrieval) |
| `embedding.tier` | `default` | Model tier used when `embedding.provider` is an API provider |
| `rag.score_threshold` | `0.25` | Minimum cosine similarity for RAG results |
| `rag.max_k` | `5` | Maximum products returned per query |
| `rag.collection_name` | `kapruka_catalog` | Qdrant collection name |
| `llm.temperature` | `0.2` | Sampling temperature for all LLM calls |
| `llm.max_tokens` | `1024` | Max response tokens |

> **Note:** When `embedding.provider = clip`, embeddings run entirely locally via HuggingFace transformers — no API key required. When set to an API provider, the corresponding key must be present in `.env`.

### `config/model.yaml` — Model Routing

Maps every supported provider to `general`, `strong`, and `reason` chat tiers. The `general` tier is used for most agents; the `strong` tier is reserved for the safety reflection loop.

---

## 🧪 Running Tests

```bash
make test           # pytest tests/
make test-v         # pytest tests/ -v  (verbose)
```

The test suite covers all agents, the router, data validation, and the full integration flow — with mocked LLM calls for deterministic results.

---

## 🔒 Safety Design

The system enforces a two-layer allergen safety net:

1. **Reflection Loop** (`agents/reflection_loop.py`) — A dedicated `strong`-tier LLM reviews every catalog draft against the recipient's saved allergy profile before the response is returned. If a violation is found it outputs a `REVISED:` response that replaces the unsafe recommendation and explains the blocked ingredient.

2. **UI Allergen Banner** (`ui/app.py`) — Every assistant message is scanned client-side for allergen keywords. If detected outside a safe context (e.g., "nut-free", "allergy-friendly"), a live `⚠️ Allergen alert` warning banner is shown below the response.

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
| OpenAI | ✅ | ✅ |
| Google Gemini | ✅ | ✅ |
| Cohere | ✅ | ✅ |
| OpenRouter | ✅ | ✅ |
| DeepSeek | ✅ | ❌ |
| Ollama (local) | ✅ | ❌ |

All providers are accessed through an OpenAI-compatible API interface — switching providers requires only a change to `config/params.yaml`, no code changes.

> **Embeddings** are handled entirely locally by CLIP (`openai/clip-vit-base-patch32` via HuggingFace) — no embedding provider API key is needed regardless of which LLM provider you choose.

---

## 🛠️ Makefile Reference

```
make install        Install dependencies via uv
make env            Copy .env .example → .env

make crawl          Run the Playwright scraper
make enrich         Run LLM enrichment / clean-and-patch
make patch-images   Patch image_url into catalog.json (one-time)
make ingest         Embed catalog into Qdrant (CLIP dual-vectors)
make pipeline       crawl + enrich + ingest (full first-time setup)

make ui             Launch the Streamlit Concierge UI

make test           Run all tests via pytest
make test-v         Run tests with verbose output

make clean          Remove __pycache__, .ipynb_checkpoints, qdrant_db
make zip            Clean + package project for submission
```

Compatible with Windows (PowerShell) and Linux/macOS.

---

## ⚠️ Disclaimer

This is a pre-development prototype built for the AEE Bootcamp · Mini Project 03. It is not affiliated with or endorsed by Kapruka Holdings Ltd. Product data is scraped for educational purposes only.

---

*Built by Zuu Crew AI · AI Engineer Essentials Bootcamp*