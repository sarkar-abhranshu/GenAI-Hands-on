# Unit 4: Multimodal Models, Agentic Workflows & LLM Evaluation

Welcome to Unit 4 — the final unit. This unit covers the frontier of applied AI: multimodal perception, autonomous agent systems, and the rigorous evaluation and responsible deployment of LLMs.

---

## Notebooks

### 1. [Multimodal Models](./1_Multimodal_Models.ipynb)
- **Why Multimodal?** Shared embedding spaces vs. unimodal pipelines
- **CLIP**: Contrastive learning, zero-shot image classification, image-text similarity scoring
- **BLIP**: Image captioning, Visual Question Answering (VQA)
- **Whisper**: Speech-to-text transcription, language detection
- **Pipeline**: Audio → Whisper → CLIP image search → BLIP caption (end-to-end demo)

### 2. [Agentic Workflows](./2_Agentic_Workflows.ipynb)
- **ReAct Loop**: Think → Act → Observe → repeat
- **Design Patterns**: Reflection, Tool Use, Planning, Multi-Agent Collaboration
- **AutoGen**: `ConversableAgent`, function registration, Researcher → Writer → Translator pipeline
- **CrewAI**: `Agent`, `Task`, `Crew`, `@tool` decorator, same pipeline rebuilt with roles
- **Devyan**: 4-agent software development pipeline (Architect → Programmer → Tester → Reviewer)

### 3. [Evaluation, Data, Ethics & Trends](./3_Evaluation_Data_Ethics.ipynb)
- **DeepEval**: `LLMTestCase`, `AnswerRelevancyMetric`, `FaithfulnessMetric`, threshold-based testing
- **TruLens**: RAG Triad (Context Relevance, Groundedness, Answer Relevance), leaderboard
- **Data Behind LLMs**: Pre-training data sources, quality pipeline, instruction datasets, deduplication
- **LLM Ethics**: Bias demonstration, fairness frameworks, EU AI Act tiers
- **LLM Security**: Prompt injection demo, defense strategies, responsible AI principles
- **Recent Developments**: Reasoning models (DeepSeek-R1 via Groq), SLMs, frontier trends

---

## Assignment

- **[Assignment: Evaluated Agentic RAG System](./Assignment_Unit4.md)**: Build a multi-agent system that retrieves information, generates answers, evaluates them with DeepEval, and retries if quality is below threshold.

---

## Reading (Deep Dives)

| File | Content |
|---|---|
| [01_DeepDive_Multimodal.md](./reading/01_DeepDive_Multimodal.md) | CLIP contrastive training (InfoNCE loss), BLIP bootstrapping, Whisper architecture, GPT-4o and the multimodal frontier |
| [02_DeepDive_Agentic_Workflows.md](./reading/02_DeepDive_Agentic_Workflows.md) | ReAct pattern, all 4 design patterns in depth, AutoGen vs CrewAI vs LangGraph, production failure modes |
| [03_DeepDive_Evaluation_Data_Ethics.md](./reading/03_DeepDive_Evaluation_Data_Ethics.md) | BLEU/ROUGE failure analysis, DeepEval metrics math, TruLens RAG Triad, data quality pipeline, RLHF/DPO, EU AI Act, prompt injection taxonomy |

---

## Setup

1. Create a `.env` file in this directory:
   ```bash
   GROQ_API_KEY=your_groq_key_here
   TAVILY_API_KEY=your_tavily_key_here
   ```

2. **Groq** (free): [console.groq.com](https://console.groq.com) — used in all 3 notebooks
3. **Tavily** (free tier): [tavily.com](https://tavily.com) — used in Notebook 2 for web search
4. **Notebook 1** requires no API keys — all models run locally via Hugging Face

> **Colab**: Runtime → Change runtime type → T4 GPU (recommended for Notebook 1)

---

## Key Libraries

| Library | Notebook | Purpose |
|---|---|---|
| `transformers` | 1 | CLIP, BLIP, Whisper models |
| `torch` | 1 | Tensor operations, GPU inference |
| `Pillow` | 1 | Image loading and processing |
| `soundfile` | 1 | Audio file I/O |
| `datasets` | 1, 3 | HuggingFace datasets (LibriSpeech, Alpaca) |
| `pyautogen` | 2 | AutoGen agent framework |
| `crewai` | 2 | CrewAI agent framework |
| `tavily-python` | 2 | Web search tool |
| `langchain-groq` | 2, 3 | Groq LLM integration |
| `deepeval` | 3 | LLM evaluation framework |
| `trulens` | 3 | RAG Triad evaluation and monitoring |
| `faiss-cpu` | 3 | Vector store for RAG |
| `sentence-transformers` | 3 | HuggingFace embeddings |
| `pandas`, `matplotlib` | 3 | Data analysis and visualization |

---

## Prerequisites

- **Unit 3** — Advanced RAG (Hybrid Retrieval, Cross-Encoder, HyDE): Notebook 3 builds a RAG system from scratch and evaluates it. Familiarity with FAISS, embeddings, and LangChain LCEL is assumed.
- **Python environment**: Google Colab recommended. Local setup requires Python 3.10+.
