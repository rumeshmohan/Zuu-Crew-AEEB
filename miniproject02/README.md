# 🏡 Prime Lands Intelligence Platform

> A production-ready Retrieval-Augmented Generation (RAG) intelligence layer for querying real estate listings — featuring semantic caching (CAG), corrective retrieval (CRAG), and a comparative chunking evaluation suite.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-LCEL-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white"/>
  <img src="https://img.shields.io/badge/Qdrant-Vector%20DB-DC143C?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/Playwright-Async%20Scraper-45ba4b?style=for-the-badge&logo=playwright&logoColor=white"/>
</p>

---

**Course:** AI Engineer Essentials – Context Engineering
**Organization:** Zuu Crew Machine Learning Academy
**Project:** MiniProject02

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Evaluation Highlights](#-evaluation-highlights)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Running Tests](#-running-tests)
- [Performance Benchmarks](#-performance-benchmarks)
- [Contributing](#-contributing)

---

## 🔍 Overview

The **Prime Lands Intelligence Platform** is built to ingest, process, and intelligently query real estate listings from [primelands.lk](https://primelands.lk). It goes beyond standard RAG by layering **Cache-Augmented Generation (CAG)** for cost-efficient repeated queries and **Corrective RAG (CRAG)** for graceful degradation when retrieval confidence is low.

The platform also serves as a **research testbed** for comparing five distinct text chunking strategies across latency, precision, and LLM answer relevance metrics.

---

## 🏗️ System Architecture

The platform is organized into four core engineering components:

### 1. 🕷️ Property Crawler
An asynchronous **Playwright**-based web scraper that navigates `primelands.lk` using BFS traversal. It extracts comprehensive property metadata and persists it in two formats:

| Output Format | Description |
|---|---|
| `.jsonl` | Structured records with all metadata fields |
| `.md` | Human-readable Markdown summaries per listing |

**Extracted fields:** `property_id`, `title`, `address`, `price`, `bedrooms`, `bathrooms`, `sqft`, `amenities`, `agent`

---

### 2. ✂️ Chunking Lab
A comparative text-splitting pipeline that processes the raw corpus using **five distinct strategies**, each embedded and persisted in a local Qdrant vector database:

| Strategy | Description |
|---|---|
| **Semantic** | Splits based on semantic similarity boundaries |
| **Fixed** | Uniform token/character-length chunks |
| **Sliding Window** | Overlapping windows for context continuity |
| **Parent-Child** | Hierarchical chunks with parent context |
| **Late Chunking** | Deferred chunking after full-document embedding |

---

### 3. 🧠 Intelligence Layers

#### RAG — Baseline Retrieval
Standard retrieval pipeline built with **LangChain Expression Language (LCEL)** for clean, composable chain logic.

#### CAG — Cache-Augmented Generation
A two-tier **semantic cache** that intercepts repeated or near-duplicate queries before they hit the LLM:

- Cache lookup via **cosine similarity** (hit threshold: `> 0.90`)
- Serves cached responses instantly, bypassing generation entirely
- Achieves a **109× latency speedup** on cache hits

#### CRAG — Corrective RAG
An intelligent fallback mechanism that monitors retrieval quality:

- Evaluates retrieval confidence score on every query
- If confidence falls below `0.6`, triggers a **corrective fallback** (e.g., web search, reranking, or query rewriting)
- Prevents hallucination on low-confidence retrievals

---

### 4. 📊 Performance Arena
A comprehensive evaluation suite that benchmarks:

- Chunking strategy **latency** and **answer relevance**
- **CAG hit rates** and latency reduction
- **Token cost projections** for scaled usage

---

## 📈 Evaluation Highlights

| Metric | Value |
|---|---|
| 🏆 Best Chunking Strategy | **Sliding Window** |
| Avg. Latency (Sliding Window) | `6.19s` |
| Answer Relevance Score | `0.639` |
| CAG Cache Hit Rate | **79.0%** |
| Latency (cache miss) | `~1,255ms` |
| Latency (cache hit) | `~11.5ms` |
| Cache Speedup Factor | **109×** |
| Monthly Cost Reduction (CAG) | **68.2%** |
| Estimated Monthly Cost (500 DAU) | **~$2.82** |

---

## ✅ Prerequisites

Ensure the following are installed on your system before proceeding:

- **Python** `3.10+`
- **Docker** & **Docker Compose**
- **make** utility
  - macOS / Linux: available by default
  - Windows: install via `winget install GnuWin32.Make` or `choco install make`

> **Cross-platform note:** The Makefile auto-detects your OS and switches between Unix and Windows shell commands automatically — no manual configuration needed.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/prime-lands-intelligence.git
cd prime-lands-intelligence
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in your API keys:

```env
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here
```

### 3. Install Dependencies

```bash
make setup
```

This creates a virtual environment and installs all required Python packages.

---

## 🚀 Usage

### Full Pipeline (Recommended)

Run the complete pipeline — crawl → ingest → test-all → launch services — in a single command:

```bash
make full
```

### Step-by-Step Execution

Run each stage individually for finer control:

```bash
# 1. Crawl property listings from primelands.lk
make crawl

# 2. Chunk, embed, and index data into Qdrant
make ingest

# 3. Run the full test suite (unit + integration)
make test-all

# 4. Start API services via Docker
make up
```

### All Available Make Targets

#### Development

| Target | Description |
|---|---|
| `make setup` | Install all dependencies |
| `make lint` | Run ruff linter across `src/` and `tests/` |
| `make format` | Auto-format with ruff |

#### Pipeline

| Target | Description |
|---|---|
| `make crawl` | Run the web crawler (notebook 01) |
| `make ingest` | Chunk, embed, and index into Qdrant (notebook 02) |
| `make full` | Full end-to-end pipeline: crawl → ingest → test-all → up |

#### Testing

| Target | Description |
|---|---|
| `make test` | Unit tests only |
| `make test-all` | Unit + integration tests |
| `make test-integration` | Integration tests only |
| `make test-cov` | Tests with HTML coverage report → `data/coverage/index.html` |

#### Docker

| Target | Description |
|---|---|
| `make up` | Start all services in production mode |
| `make up-dev` | Start all services with dev profile |
| `make down` | Stop and remove all containers |
| `make logs` | Tail live API logs |
| `make status` | Show service health |

#### Maintenance

| Target | Description |
|---|---|
| `make clean` | Remove Python bytecode and pytest / coverage artefacts |
| `make clean-utils` | Remove linter and type-checker caches (`.ruff_cache`, `.mypy_cache`) |
| `make clean-submission` | Full clean — removes all generated data including `data/chunks`, `data/evaluation`, `data/vectorstore`, and raw scraped files. Use before packaging for submission. |

> ⚠️ **`make clean-submission` is destructive.** It deletes your crawled raw data and all indexed vector store collections. Re-run `make full` to regenerate everything from scratch.

---

## 📁 Project Structure

```
MiniProject02/
│
├── config/                  # Model configs, FAQ seeds, thresholds
│
├── data/
│   ├── raw/                 # Scraped JSONL & Markdown from primelands.lk
│   ├── chunks/              # Generated chunks for all 5 strategies
│   ├── evaluation/          # Output metrics: CAG stats, CRAG impact, chunking comparison
│   └── vectorstore/         # Persistent Qdrant collections
│
├── src/                     # Application source code (Domain-Driven Design)
│   ├── crawler/             # Playwright scraper & BFS navigator
│   ├── chunking/            # Five chunking strategy implementations
│   ├── rag/                 # RAG, CAG, and CRAG pipeline logic
│   └── evaluation/          # Benchmarking and metrics
│
├── tests/                   # Pytest suite — unit & integration tests
│
├── Dockerfile               # Production image definition
├── docker-compose.yml       # Multi-container orchestration
├── Makefile                 # Unified task runner (cross-platform)
└── README.md
```

---

## 🔧 Configuration

Key parameters can be tuned in the `config/` directory:

| Parameter | Default | Description |
|---|---|---|
| `cag_similarity_threshold` | `0.90` | Cosine similarity cutoff for cache hits |
| `crag_confidence_threshold` | `0.60` | Minimum retrieval confidence before fallback |
| `embedding_model` | configurable | Model used for vector embeddings |
| `llm_model` | configurable | LLM used for generation |

---

## 🧪 Running Tests

```bash
# Unit tests only
make test

# Unit + integration tests
make test-all

# Integration tests only
make test-integration

# With HTML coverage report
make test-cov
```

The test suite covers unit tests for each chunking strategy, integration tests for the RAG/CAG/CRAG pipelines, crawler output validation, and vector store read/write operations. Coverage reports are written to `data/coverage/index.html`.

---

## 📊 Performance Benchmarks

### Chunking Strategy Comparison

| Strategy | Avg. Latency | Answer Relevance |
|---|---|---|
| Sliding Window | `6.19s` | `0.639` ⭐ |
| Semantic | — | — |
| Fixed | — | — |
| Parent-Child | — | — |
| Late Chunking | — | — |

> Full benchmark results are written to `data/evaluation/chunking_comparison.json` after running `make test-all`.

### CAG Cost Projection (500 Daily Active Users)

| Scenario | Monthly Cost |
|---|---|
| Without CAG | ~$8.87 |
| With CAG (79% hit rate) | **~$2.82** |
| **Savings** | **68.2%** |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'feat: add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please ensure all new code passes `make lint` and `make test-all` before submitting.

---

<p align="center">Built with ❤️ by <strong>Zuu Crew Machine Learning Academy</strong></p>