# Unit 3: Advanced RAG & Generation Enhancement

Welcome to Unit 3. This unit takes you from the Naïve RAG pipeline built in Unit 2 to a full production-grade Advanced RAG system, and then covers how to enhance the generation side with fine-tuning, quantization, and Mixture of Experts.

---

## Notebooks

### 1. [Advanced RAG & Retrieval](./1_Advanced_RAG_and_Retrieval.ipynb)
- **Why Naïve RAG fails**: vocabulary mismatch and top-K noise (live demo)
- **Sparse Retrieval (BM25)**: TF-IDF evolution, BM25 formula, `rank_bm25` library
- **Dense Retrieval (SBERT)**: sentence embeddings, bi-encoder architecture
- **ColBERT**: token-level late interaction, MaxSim scoring (built from scratch)
- **Hybrid Retrieval (BM25 + SBERT + RRF)**: Reciprocal Rank Fusion with worked numericals

### 2. [Re-Ranking & Query Expansion](./2_Reranking_and_Query_Expansion.ipynb)
- **Cross-Encoder Re-Ranking**: bi-encoder vs cross-encoder, 2-stage retrieval strategy
- **HyDE (Hypothetical Document Embedding)**: LLM-generated hypothetical answers as queries
- **Multi-Query Retrieval**: LangChain's `MultiQueryRetriever` with Gemini
- **Full Pipeline**: Query Expansion → Hybrid Retrieval → Re-Ranking → LLM Generation (LCEL)

### 3. [Generation Enhancement](./3_Generation_Enhancement.ipynb)
- **Fine-Tuning Decision Tree**: Prompt vs RAG vs Fine-Tune vs Pre-Train
- **LoRA / PEFT**: Low-rank adaptation math, `peft` library, trainable parameter count demo
- **Data Precision**: FP32 vs FP16 vs BF16 vs INT8 — memory and accuracy trade-offs
- **Quantization**: Linear quantization math, INT8 worked example, model size comparison table
- **Mixture of Experts (MoE)**: Architecture, Mixtral explained, software-level MoE with LangChain + Groq

---

## Assignment

- **[Assignment: Advanced RAG System](./Assignment_AdvancedRAG.md)**: Build a complete Advanced RAG pipeline over a custom corpus — Hybrid Retrieval, Cross-Encoder Re-Ranking, and Query Expansion, with a before/after comparison experiment.

---

## Reading (Deep Dives)

| File | Content |
|---|---|
| [01_DeepDive_Advanced_RAG.md](./reading/01_DeepDive_Advanced_RAG.md) | Naïve RAG failure modes, Advanced RAG taxonomy, full pipeline diagram, evaluation metrics (RAGAS) |
| [02_DeepDive_Retrieval_Techniques.md](./reading/02_DeepDive_Retrieval_Techniques.md) | BM25 full derivation with worked example, ColBERT MaxSim, RRF math and analysis, cross-encoder architecture |
| [03_DeepDive_Generation_Enhancement.md](./reading/03_DeepDive_Generation_Enhancement.md) | LoRA math depth (rank, scaling, QLoRA), IEEE 754 formats, quantization algorithms (PTQ/GPTQ), MoE load balancing |

---

## Setup

1. Ensure you have a `.env` file in this directory with your API keys:
   ```bash
   GOOGLE_API_KEY=your_gemini_key_here
   GROQ_API_KEY=your_groq_key_here
   ```
2. Each notebook installs its own dependencies via `%pip install` in the first cell.

### Key Libraries

| Library | Used In | Purpose |
|---|---|---|
| `rank-bm25` | Notebook 1, 2 | BM25 sparse retrieval |
| `sentence-transformers` | Notebook 1, 2 | SBERT bi-encoder + CrossEncoder re-ranker |
| `peft` | Notebook 3 | LoRA adapter definition |
| `transformers` | Notebook 3 | Base model loading, PEFT integration |
| `langchain-google-genai` | Notebook 2, 3 | Gemini (HyDE, generation) |
| `langchain-groq` | Notebook 3 | Groq/Llama (MoE experts, router) |
| `langchain-community` | Notebook 2 | FAISS vector store, MultiQueryRetriever |
| `numpy` | All | Vector math, quantization calculations |

---

## Prerequisites

- Unit 2, Notebook 4 (RAG & Vector Stores) — you should understand basic RAG, FAISS, and cosine similarity before starting this unit.
