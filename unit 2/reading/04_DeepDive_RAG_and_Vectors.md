# 📘 Unit 2 Deep Dive: Part 4 - RAG & The Geometry of Meaning

## 1. RAG: The "Open-Book" Architect
**Retrieval Augmented Generation (RAG)** is the bridge between a model's brain and your private data.

### 1.1 Parametric vs. Non-Parametric Memory
- **Parametric:** The weights inside the model (learned during training). It is frozen in time.
- **Non-Parametric:** The documents you provide during runtime. It can be updated instantly.

---

## 2. The Science of Embeddings
Computers don't understand words; they understand **Vectors** (coordinates in space).

### 2.1 Vector Space
An embedding model converts a sentence into a list of numbers (e.g., 384 or 1536 dimensions).
- **Semantic Proximity:** In this 1536-dimensional space, the vector for "King" is mathematically closer to "Queen" than it is to "Refrigerator".
- **Cosine Similarity:** We calculate the "Angle" between vectors. Small angle = High similarity.

---

## 3. Vector Stores & FAISS
A **Vector Store** is a database designed to index and search these coordinates.

### 3.1 Why not a standard SQL database?
SQL is good at exact matches (`name="Piyush"`). It is terrible at "Search for something *nearly* like this". Vector stores use **Approximate Nearest Neighbor (ANN)** algorithms to search billions of vectors in milliseconds.

### 3.2 Indexing Algorithms (The Hard Math)
In the notebook, we saw several FAISS indexes:
- **Flat Index:** Brute-force. Compares your query to *every* vector. 100% accurate but slow.
- **IVF (Inverted File):** Clusters vectors into "buckets". You only search the bucket most similar to your query. Fast, but might miss the exact best match.
- **HNSW (Hierarchical Navigable Small World):** Builds a "Friendship Graph" between vectors. It's like "Six Degrees of Separation" for data. Extremely fast.
- **PQ (Product Quantization):** Compresses vectors by rounding the numbers. Saves 90% of RAM.

---

## 4. The RAG Pipeline Mechanics
In LangChain, the pipeline looks like this:
1.  **Ingestion:** PDF -> Text Chunks -> Embeddings -> Vector Store.
2.  **Retrieval:** User Question -> Embedding -> Semantic Search in Vector Store -> Top-K Context.
3.  **Augmentation:** Inject the Top-K Context into the System Prompt.
4.  **Generation:** LLM reads the context and answers the question.

---

## 5. Challenges in RAG
- **Lost in the Middle:** Models often pay more attention to the beginning and end of the context, ignoring the middle.
- **Chunking Strategy:** If you cut a sentence in half during chunking, you lose meaning.
- **Noise:** If you retrieve irrelevant documents, the model will get confused (Garbage In, Garbage Out).

---

## 🔍 Teacher's Key: Expected Learning Outcomes
- **Concept:** Explain the difference between search in a SQL database vs. a Vector database.
- **Code:** Can the student initialize a FAISS index and add documents to it?
- **Theory:** What is the trade-off between a Flat Index (Search everything) and an IVF Index (Search clusters)?
