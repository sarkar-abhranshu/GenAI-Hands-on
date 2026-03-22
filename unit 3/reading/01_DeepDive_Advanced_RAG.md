# Deep Dive: Unit 3 — Part 1: Advanced RAG

## 1. Why Naïve RAG is Not Enough

Naïve RAG (Retrieve → Augment → Generate) has two fundamental limitations:

### 1.1 The Vocabulary Mismatch Problem

Dense embedding models (SBERT, OpenAI Ada) compress an entire sentence into a single vector. This works for semantic similarity but fails for **exact lexical matching**.

- Query: `"cardiac arrest treatment"`
- Document: `"heart attack intervention protocol"`

These are semantically identical, but if the model has not been fine-tuned on medical text, it may fail to place them close in vector space.

Conversely, dense models struggle with rare proper nouns, product codes, and technical abbreviations (e.g., `"SQL-92 standard"`, `"XR-7700-B part number"`).

### 1.2 The Top-K Quality Problem

Even if retrieval is perfect, top-K selection has no guarantee that all K documents are equally relevant. In a corpus of 10,000 documents, the top-5 may include:
- 3 genuinely relevant documents
- 1 topically related but off-target document
- 1 document that matches keywords but contradicts the others

Sending all 5 to the LLM introduces noise that degrades generation quality.

---

## 2. Advanced RAG Taxonomy

Advanced RAG improvements fall into three stages of the pipeline:

```
Pre-Retrieval          Retrieval               Post-Retrieval
─────────────         ───────────             ────────────────
Query Expansion   →   Hybrid Search      →    Re-Ranking
HyDE                  (BM25 + Dense)          Compression
Multi-Query           Sparse Retrieval         Context Window
                      Dense Retrieval          Management
                      ColBERT
```

### 2.1 Pre-Retrieval Enhancements

**Problem**: The raw user query is often too short and ambiguous.

| Technique | Approach | Best For |
|---|---|---|
| **HyDE** | LLM generates hypothetical ideal answer; embed that instead | Short/vague queries |
| **Multi-Query** | LLM paraphrases query in 3 ways; union of all results | Multi-faceted questions |
| **Step-Back Prompting** | LLM generates a more abstract version of the query | Specific questions needing broader context |

### 2.2 Retrieval Enhancements

**Problem**: Dense-only retrieval misses exact-match cases.

| Method | Algorithm | Strength |
|---|---|---|
| **BM25** | TF-IDF based, statistical | Keywords, proper nouns, abbreviations |
| **Dense (SBERT)** | Neural, single-vector bi-encoder | Semantic similarity, paraphrases |
| **ColBERT** | Neural, multi-vector (per-token) | Token-level semantic matching |
| **Hybrid (RRF)** | Rank fusion of BM25 + Dense | Best of both |

### 2.3 Post-Retrieval Enhancements

**Problem**: Top-K candidates include irrelevant noise.

| Technique | Approach |
|---|---|
| **Cross-Encoder Re-Ranking** | Score each (query, doc) pair with a full BERT model |
| **LLM Re-Ranking** | Ask an LLM to rank candidates by relevance |
| **Contextual Compression** | Use an LLM to extract only the relevant sentences from each retrieved doc |

---

## 3. The Advanced RAG Pipeline (Full)

```
User Query
    │
    ▼ [Pre-Retrieval]
Query Expansion
 ├── HyDE: LLM → Hypothetical Doc → Embed
 └── Multi-Query: LLM → 3 Variants → Retrieve Each
    │
    ▼ [Retrieval]
Hybrid Retrieval
 ├── BM25 index: tokenized corpus → BM25 scores
 ├── Dense index: FAISS / vector store → cosine scores
 └── RRF fusion: rank-based combination
    │
    ▼ [Post-Retrieval]
Re-Ranking
 └── Cross-Encoder: (query, doc) → single relevance score
    │
    ▼ [Generation]
LLM (Gemini / Llama)
 └── System: "Answer using ONLY the provided context"
    │
    ▼
Answer
```

---

## 4. Key Performance Metrics for RAG

When evaluating a RAG system, the key metrics are:

| Metric | What It Measures | How |
|---|---|---|
| **Context Precision** | Are the retrieved docs relevant? | % of retrieved docs that are relevant |
| **Context Recall** | Did we miss any important docs? | % of relevant docs that were retrieved |
| **Answer Faithfulness** | Does the answer match the context? | NLI check: does context entail the answer? |
| **Answer Relevance** | Does the answer address the question? | Cosine similarity of question ↔ answer |

Tools: **RAGAS** library automates all four metrics using an LLM judge.

---

## Teacher's Key: Expected Learning Outcomes

- **Concept**: Explain the two fundamental limitations of Naïve RAG (vocabulary mismatch, top-K noise).
- **Architecture**: Draw the full Advanced RAG pipeline with all three stages.
- **Trade-off**: When would you use HyDE vs Multi-Query for query expansion?
- **Evaluation**: Name two metrics for measuring RAG pipeline quality.
