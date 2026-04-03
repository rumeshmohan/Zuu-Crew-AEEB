<div align="center">

# 🤖 AI Engineer Essentials — Mini Project 00

**A Production-Ready Multi-Provider LLM Pipeline**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![Mistral AI](https://img.shields.io/badge/Mistral%20AI-FF7000?style=for-the-badge&logo=mistral&logoColor=white)](https://mistral.ai)
[![Cohere](https://img.shields.io/badge/Cohere-39594D?style=for-the-badge&logo=cohere&logoColor=white)](https://cohere.com)

*Build resilient, provider-agnostic AI systems with advanced prompt engineering patterns*

---

[Overview](#-overview) •
[Features](#-features) •
[Architecture](#-project-architecture) •
[Getting Started](#-getting-started) •
[Notebooks](#-progressive-implementation) •
[Customization](#-extension--customization)

</div>

---

## 📋 Overview

This repository contains the implementation of **Mini Project 00** for the AI Engineer Essentials Bootcamp. The project demonstrates how to build a **modular, production-ready AI pipeline** that seamlessly integrates multiple Large Language Model providers—**OpenAI**, **Google Gemini**, **Groq**, **Mistral AI**, and **Cohere**—into a unified, extensible workflow.

### Why This Matters

In production environments, relying on a single LLM provider creates risk. This project teaches you to:

- **Abstract provider complexity** behind clean interfaces
- **Route tasks intelligently** based on capability tiers
- **Manage token budgets** for cost optimization
- **Extract structured data** reliably from unstructured text

---

## ✨ Features

### 🔧 Modular Configuration System
Centralized YAML-based configuration separates concerns between system behaviors and model definitions, enabling environment-specific overrides without code changes.

### 🔄 Provider Abstraction Layer
A unified client interface allows seamless switching between OpenAI, Gemini, Groq, Mistral, and Cohere with zero code modifications—just update your config.

### 📝 Advanced Prompt Engineering
Structured prompt management using a typed template registry supporting:
- **Few-Shot Learning** — Learn from examples
- **Chain-of-Thought (CoT)** — Step-by-step reasoning
- **Tree-of-Thought (ToT)** — Explore multiple reasoning paths

### 📊 Token Economics & Optimization
Precise token counting and context window management using `tiktoken`, with built-in strategies for handling context overflow:
- Intelligent summarization
- Smart truncation
- Context window optimization

### 🎯 Structured Data Extraction
Robust pipelines for extracting JSON and structured data from unstructured text with validation and error handling.

---

## 🏗 Project Architecture

```
MiniProject00/
│
├── 📁 config/                  # Configuration Management
│   ├── config.yaml             # System behaviors, defaults, task overrides
│   └── models.yaml             # Model tier definitions (General/Strong/Reason)
│
├── 📁 data/                    # Input Datasets
│   ├── incidents.txt           # Incident data samples
│   ├── news_feed.txt           # Simulated news feeds
│   ├── sample_messages.txt     # Message classification samples
│   ├── scenarios.txt           # Test scenarios
│   └── synthesized_for_part4.txt # Synthesized data for budget experiments
│
├── 📁 notebooks/               # Interactive Implementation
│   ├── part0_verification.ipynb
│   ├── part1_contract_few_shot.ipynb
│   ├── part2_stability_experiment.ipynb
│   ├── part3_logi_comm.ipynb
│   ├── part4_the_budget_keeper.ipynb
│   └── part5_news_feed_ext.ipynb
│
├── 📁 output/                  # Generated Artifacts
│   ├── classified_messages.xlsx # Message classification results
│   └── flood_report.xlsx       # Flood analysis report
│
├── 📁 utils/                   # Core Engineering Utilities
│   ├── config_loader.py        # Singleton configuration manager
│   ├── llm_client.py           # Unified LLM provider client
│   ├── prompts.py              # Centralized prompt template registry
│   ├── router.py               # Intelligent task routing logic
│   └── token_utils.py          # Token counting & context optimization
│
├── .env.example                # Environment variable template
├── pyproject.toml              # Project dependencies (uv managed)
├── uv.lock                     # Dependency lock file
└── README.md                   # You are here
```

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.11+ | Runtime environment |
| uv | Latest | Fast dependency management |
| API Keys | — | OpenAI, Gemini, Groq, Mistral, Cohere access |

### Installation

**1. Clone the Repository**

```bash
git clone <repository_url>
cd MiniProject00
```

**2. Install Dependencies**

This project uses [uv](https://github.com/astral-sh/uv) for blazing-fast dependency management:

```bash
uv sync
```

> 💡 **Don't have uv?** Install it with: `curl -LsSf https://astral.sh/uv/install.sh | sh`

**3. Configure Environment Variables**

```bash
cp .env.example .env
```

Edit `.env` with your API credentials:

```env
# Required API Keys
OPENAI_API_KEY=sk-your-openai-key-here
GEMINI_API_KEY=your-google-gemini-key-here
GROQ_API_KEY=gsk_your-groq-key-here
MISTRAL_API_KEY=your-mistral-key-here
COHERE_API_KEY=your-cohere-key-here
```

### Running the Project

**1. Activate the Virtual Environment**

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

**2. Launch Jupyter**

```bash
jupyter lab
```

Or open the `notebooks/` folder in VS Code with the Jupyter extension (ensure the kernel points to your virtual environment).

**3. Execute Sequentially**

Start with `part0_verification.ipynb` to validate your setup, then proceed through Parts 1–5.

---

## 📓 Progressive Implementation

The project is structured as a learning journey across six notebooks:

| Part | Notebook | Focus Area | Key Concepts |
|------|----------|------------|--------------|
| **0** | `part0_verification.ipynb` | Environment Setup | Provider connectivity, API validation |
| **1** | `part1_contract_few_shot.ipynb` | Document Intelligence | Few-shot prompting, entity extraction |
| **2** | `part2_stability_experiment.ipynb` | Model Behavior | Temperature effects, determinism testing |
| **3** | `part3_logi_comm.ipynb` | Advanced Reasoning | Chain-of-Thought, Tree-of-Thought patterns |
| **4** | `part4_the_budget_keeper.ipynb` | Cost Optimization | Token budgets, context management |
| **5** | `part5_news_feed_ext.ipynb` | Data Pipelines | JSON extraction, structured outputs |

---

## 🧩 Extension & Customization

### Switching Models

Edit `config/models.yaml` to swap underlying models without touching code:

```yaml
# config/models.yaml
providers:
  openai:
    general: gpt-4o-mini
    strong: gpt-4o
    reason: o1-preview
  
  gemini:
    general: gemini-1.5-flash
    strong: gemini-1.5-pro
    reason: gemini-1.5-pro
  
  mistral:
    general: mistral-small-latest
    strong: mistral-large-latest
    reason: mistral-large-latest
  
  cohere:
    general: command-r
    strong: command-r-plus
    reason: command-r-plus
```

### Adjusting Parameters

Tune defaults in `config/config.yaml`:

```yaml
# config/config.yaml
defaults:
  temperature: 0.7
  max_tokens: 2048
  
tasks:
  contract_analysis:
    temperature: 0.2  # Lower for consistency
  creative_writing:
    temperature: 0.9  # Higher for creativity
```

### Adding New Prompts

Register templates in `utils/prompts.py`:

```python
# utils/prompts.py
PROMPTS = {
    "my_new_task": """
    You are an expert at {domain}.
    
    Given the following input:
    {input}
    
    Please provide:
    1. Analysis
    2. Recommendations
    3. Next steps
    """,
}
```

---

## 🛠 Utility Modules

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `config_loader.py` | Singleton config manager | `get_config()`, `get_model()` |
| `llm_client.py` | Unified provider interface | `complete()`, `stream()` |
| `prompts.py` | Template registry | `get_prompt()`, `format_prompt()` |
| `router.py` | Task routing | `route_to_tier()`, `select_provider()` |
| `token_utils.py` | Token management | `count_tokens()`, `truncate()`, `summarize()` |

---

## 📄 License

This project is part of the AI Engineer Essentials Bootcamp curriculum.

---

<div align="center">

**Built with ❤️ for the AI Engineer Essentials Bootcamp**

</div>
