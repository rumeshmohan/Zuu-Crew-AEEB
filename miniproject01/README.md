# 🏦 LedgerMind: Financial RAG vs. Fine-Tuning Benchmark

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **High-Stakes Financial RAG Framework**  
> A production-grade framework evaluating the trade-offs between **Parametric Memory (Fine-Tuning)** and **Non-Parametric Memory (Advanced RAG)** for high-stakes financial data interpretation.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Engineering Insights](#engineering-insights)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Architecture](#technical-architecture)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## 🎯 Overview

**LedgerMind** is an experimental AI architecture designed to solve the "Hallucination vs. Reasoning" dilemma in financial analysis. This project implements and rigorously evaluates two competing architectures for processing 100+ page Annual Reports:

1. **Architecture A ("The Intern"):** A Fine-tuned Llama-3-8B model that prioritizes domain-specific syntax and style (Parametric Memory).
2. **Architecture B ("The Librarian"):** An Advanced RAG system utilizing Hybrid Search (Dense + BM25) and Reciprocal Rank Fusion (Non-Parametric Memory).

### The Engineering Challenge

Financial data requires zero tolerance for hallucination. This project builds a **synthetic evaluation pipeline** (using Gemma 3 and DeepSeek-R1) to conduct a head-to-head showdown on 1,600+ Q&A pairs from Uber's 2024 Annual Report, measuring:

* **Factuality:** Can it cite specific legal codes? (RAG domain)
* **Reasoning:** Can it synthesize trends across chapters? (Fine-Tuning domain)
* **Cost/Latency:** Which architecture scales better in production?

---

## 💡 Engineering Insights

Building this framework revealed a critical production constraint: **Hybrid Routing is Mandatory.**

* **The "Vibe" Trap:** The Fine-Tuned model scored **4.01/5** on judge scores because it sounded confident and professional.
* **The "Fact" Reality:** Despite high judge scores, the Fine-Tuned model had a **24.8% error rate** on specific numbers (e.g., page references).

### Production Strategy

Based on this data, a production deployment should not choose *one* model. It should use a **Router**:

1. **Route "Summarize strategic risks"** → Fine-Tuned Model (Better synthesis/reasoning).
2. **Route "What was the EBITDA in Q3?"** → RAG System (Better precision/retrieval).

---

## 🏆 Key Findings

| Metric | RAG + RRF | Fine-Tuned | Winner |
|--------|-----------|------------|---------|
| **Judge Score** (1-5) | 2.54 | **4.01** | 🥇 Fine-Tuned |
| **ROUGE-L** (0-1) | 0.121 | **0.608** | 🥇 Fine-Tuned |
| **Latency** (ms) | **1,696** | 2,524 | 🥇 RAG |
| **Head-to-Head Wins** | 49 | **346** | 🥇 Fine-Tuned |
| **Win Margin** | - | **61.1%** | 🥇 Fine-Tuned |
| **Monthly Cost** | $454 | **$409** | 🥇 Fine-Tuned |

### 📊 Verdict

**The Fine-Tuned Model wins overall** with superior accuracy and coherence. However, it exhibits significant failures on precision-critical queries, including:

- ❌ Citation confabulation (inventing legal codes)
- ❌ Page number omissions
- ❌ Parametric mixing (blending unrelated sections)

---

## 📁 Project Structure

```
./
├── .env                          # Environment variables (API keys)
├── .gitignore                    # Git ignore patterns
├── .python-version               # Python version specification
├── pyproject.toml                # UV/pip dependency management
├── uv.lock                       # Locked dependencies
├── README.md                     # This file
│
├── artifacts/                    # Generated outputs & results
│   ├── data/
│   │   ├── train.jsonl           # 1,146 training Q&A pairs
│   │   ├── golden_test_set.jsonl # 487 test Q&A pairs
│   │   └── intern_predictions.jsonl # Fine-tuned model predictions
│   └── outputs/
│       ├── final_showdown.csv    # Comprehensive evaluation results
│       └── llama-3-financial-intern.zip # LoRA adapter weights
│
├── data/
│   └── pdfs/
│       └── 2024-Annual-Report.pdf # Source document (Uber)
│
├── notebooks/                    # Jupyter workflows (execute in order)
│   ├── 01_data_factory.ipynb     # Synthetic dataset generation
│   ├── 02_finetuning_intern.ipynb # LoRA fine-tuning pipeline
│   ├── 03_rag_librarian.ipynb    # Hybrid RAG + RRF system
│   └── 04_evaluation_arena.ipynb # Head-to-head evaluation
│
├── src/                          # Core application code
│   ├── config/
│   │   ├── config.yaml           # System configuration
│   │   └── prompts.yaml          # Prompt templates
│   └── services/
│       ├── data_manager.py       # Data loading utilities
│       └── llm_services.py       # LLM API wrappers
│
└── utils/                        # Utility scripts
    ├── hallucination_finder.py   # Error analysis tools
    └── submission_checker.py     # Validation scripts
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.11+**
- **CUDA-capable GPU** (for fine-tuning) or **Google Colab T4** (free tier)
- **[UV package manager](https://github.com/astral-sh/uv)** (recommended) or pip

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/ledger-mind.git
cd ledger-mind
```

### Step 2: Install Dependencies

**Option A: Using UV (Recommended)**

```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

**Option B: Using pip**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt  # Note: Generate from pyproject.toml if needed
```

### Step 3: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# LLM API Keys
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Weaviate Cloud (for RAG system)
WEAVIATE_URL=https://xxxxxxxx.weaviate.network
WEAVIATE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 💻 Usage

### 1️⃣ Data Generation (Synthetic Pipeline)

Generate synthetic Q&A pairs from the annual report using Gemma 3:

```bash
jupyter notebook notebooks/01_data_factory.ipynb
```

**Key Features:**
- Teacher-student architecture (Gemma 3:4B → Llama-3.3-70B)
- Structured prompting (40% hard facts, 30% strategic, 30% stylistic)

### 2️⃣ Fine-Tuning "The Intern"

Train the parametric memory model using Unsloth and LoRA:

```bash
jupyter notebook notebooks/02_finetuning_intern.ipynb
```

**Configuration:**
- Base model: `unsloth/llama-3-8b-Instruct-bnb-4bit`
- LoRA (r=16, alpha=16) on attention layers
- 120 training steps, 4-bit NF4 quantization

### 3️⃣ Building "The Librarian" (RAG)

Deploy the advanced hybrid retrieval system:

```bash
jupyter notebook notebooks/03_rag_librarian.ipynb
```

**Pipeline:**
1. Dense Vector Search (top-20) → `all-MiniLM-L6-v2`
2. BM25 Keyword Search (top-20) → Exact entity matching
3. Reciprocal Rank Fusion (k=60) → Combine rankings
4. Cross-Encoder Reranking (top-10) → `ms-marco-MiniLM-L-6-v2`

### 4️⃣ Evaluation Arena

Run the head-to-head showdown using LLM-as-a-Judge (DeepSeek-R1):

```bash
jupyter notebook notebooks/04_evaluation_arena.ipynb
```

---

## 🏗️ Technical Architecture

### The Intern (Fine-Tuning)

```python
# LoRA Configuration
lora_config = LoraConfig(
    r=16,                    # Low-rank dimension
    lora_alpha=16,           # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.0,
    bias="none"
)
```

### The Librarian (RAG + RRF)

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    """
    RRF(d) = Σ [1 / (k + rank_i(d))]
    Combines dense vector search + BM25 keyword search
    """
    rrf_scores = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, doc_obj) in enumerate(ranked_list):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {'score': 0.0, 'doc': doc_obj}
            rrf_scores[doc_id]['score'] += 1.0 / (k + rank + 1)
    return rrf_scores
```

---

## ⚙️ Configuration

### LLM Providers (`src/config/config.yaml`)

```yaml
providers:
  ollama:
    model: "gemma3:4b"
    judge_model: "llama3.2:latest"
  
  openrouter:
    llm_a_model: "google/gemini-2.0-flash-001"
    llm_b_model: "meta-llama/llama-3.3-70b-instruct"
    judge_model: "deepseek/deepseek-r1:free"
```

### RAG Configuration

```yaml
weaviate:
  vectorizer_model: "sentence-transformers/all-MiniLM-L6-v2"
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

retrieval:
  dense_top_k: 20      # Vector search candidates
  bm25_top_k: 20       # Keyword search candidates
  rrf_k: 60            # RRF constant
  final_top_k: 10      # After reranking
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory (OOM)

**Solutions:**
- Reduce batch size to 1.
- Enable `gradient_checkpointing = True`.
- If using Colab, ensure you are on T4 High-RAM runtime.

#### 2. OpenRouter API Rate Limits

**Solutions:**
- Implement exponential backoff (script provided in `utils/`).
- The pipeline defaults to a 1-second delay between calls.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">

**Built by Builders, for Builders.**

[Report Bug](https://github.com/yourusername/ledger-mind/issues) · [Request Feature](https://github.com/yourusername/ledger-mind/issues)

</div>
