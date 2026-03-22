# Deep Dive: Unit 3 — Part 2: Retrieval Techniques

## 1. Sparse Retrieval: BM25

### 1.1 The Evolution from TF-IDF to BM25

**TF-IDF** (Term Frequency × Inverse Document Frequency) was the dominant retrieval algorithm from the 1970s to the 2000s.

$$\text{TF-IDF}(t, d) = \underbrace{\frac{f(t,d)}{|d|}}_{\text{TF}} \times \underbrace{\log\frac{N}{n_t}}_{\text{IDF}}$$

**Problems with TF-IDF**:
1. **No saturation**: If a term appears 100 times, it scores 100× a document where it appears once. In reality, after ~3-4 occurrences, additional occurrences add diminishing value.
2. **Length bias**: A 10,000-word document will almost always score higher than a 100-word document, even if the short document is more relevant.

### 1.2 The Full BM25 Formula

$$\text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

**Breaking down the components**:

**IDF component**:
$$\text{IDF}(q_i) = \log\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1\right)$$

- $N$ = total number of documents in corpus
- $n(q_i)$ = number of documents containing term $q_i$
- Rare terms get high IDF; common terms (e.g., "the", "is") get low IDF

**TF Saturation component**:

The denominator term $f(q_i, D) + k_1 \cdot (\ldots)$ acts as a normalizer. As $f(q_i, D) \to \infty$, the fraction approaches:
$$\frac{f \cdot (k_1+1)}{f + k_1} \to (k_1 + 1)$$

So score is **bounded by** $(k_1 + 1) \cdot \text{IDF}$ — infinite repetitions still have a ceiling.

**Length Normalization component**:
$$1 - b + b \cdot \frac{|D|}{\text{avgdl}}$$

- When $b=0$: no length normalization (pure TF)
- When $b=1$: full length normalization (TF divided by relative length)
- Default: $b=0.75$ — partial normalization

### 1.3 BM25 Worked Example

**Corpus** (3 documents):
- $D_1$: "neural network learns weights" (4 words)
- $D_2$: "neural network backpropagation weights weights" (5 words)
- $D_3$: "decision tree classifier features" (4 words)

**Query**: "neural weights"  
**Parameters**: $k_1 = 1.5$, $b = 0.75$, $\text{avgdl} = 4.33$

**Step 1: IDF values**

| Term | $n(q_i)$ | IDF |
|---|---|---|
| "neural" | 2 (D1, D2) | $\log\frac{3-2+0.5}{2+0.5}+1 = \log(0.6+1) = 0.47$ |
| "weights" | 2 (D1, D2) | $\log\frac{3-2+0.5}{2+0.5}+1 = 0.47$ |

**Step 2: TF for D2** ("neural network backpropagation weights weights")

For "weights" in $D_2$ ($f=2$, $|D_2|=5$):

$$\frac{2 \times (1.5+1)}{2 + 1.5 \times (1 - 0.75 + 0.75 \times \frac{5}{4.33})} = \frac{5.0}{2 + 1.5 \times 1.116} = \frac{5.0}{3.674} = 1.361$$

**Result**: $\text{BM25}(D_2, Q) = 0.47 \times 1.21 + 0.47 \times 1.361 = 0.568 + 0.640 = 1.21$

---

## 2. Dense Retrieval

### 2.1 Bi-Encoder (SBERT)

SBERT fine-tunes a BERT model using **contrastive learning** on sentence pairs labeled as similar/dissimilar (NLI or STS datasets).

**Training objective** (simplified):
$$\mathcal{L} = \max(0, \epsilon - \cos(\vec{s}_1, \vec{s}_2)^+ + \cos(\vec{s}_1, \vec{s}_3)^-)$$

Where $s_2$ is a positive (similar) sentence and $s_3$ is a negative (dissimilar) one.

**Inference**:
1. Pre-compute $\vec{d}_i = \text{SBERT}(D_i)$ for all documents (done once, stored in FAISS)
2. At query time: $\vec{q} = \text{SBERT}(Q)$, then $\text{score}(Q, D_i) = \cos(\vec{q}, \vec{d}_i)$

**Speed**: Sub-millisecond retrieval from millions of documents (FAISS ANN search).

### 2.2 ColBERT: Late Interaction

ColBERT addresses the fundamental limitation of bi-encoders: by compressing the entire document into one vector, you lose token-level semantic detail.

**ColBERT encoding**:
- Query $Q$ → $\{\vec{q}_1, \vec{q}_2, \ldots, \vec{q}_{|Q|}\}$ (one vector per query token)
- Document $D$ → $\{\vec{d}_1, \vec{d}_2, \ldots, \vec{d}_{|D|}\}$ (one vector per document token)

**MaxSim scoring**:
$$\text{score}(Q, D) = \sum_{i=1}^{|Q|} \max_{j=1}^{|D|} \vec{q}_i^\top \vec{d}_j$$

**Intuition**: The word "attack" in the query finds the best match across all document tokens — which might be "assault", "strike", or "hit" if they are contextually similar in the embedding space. No information is lost to pooling.

**Trade-off**:
- Stores $|D| \times d$ vectors per document instead of $d$ vectors (more storage)
- Must compute MaxSim at query time (slower than dot product of 2 vectors)
- Used in production systems like Vespa, Milvus with dedicated ColBERT indexes

---

## 3. Hybrid Retrieval and Reciprocal Rank Fusion

### 3.1 The Scale Incompatibility Problem

| Retriever | Score Example | Range |
|---|---|---|
| BM25 | 4.72 | [0, ∞) |
| SBERT cosine | 0.82 | [-1, 1] |
| ColBERT MaxSim | 6.41 | (-∞, ∞) |

Adding these directly would be dominated by BM25's unbounded scale.

### 3.2 Reciprocal Rank Fusion (RRF)

Instead of combining scores, combine **rank positions**:

$$\text{RRF}(d, R_1, \ldots, R_k) = \sum_{i=1}^{k} \frac{1}{K + r_i(d)}$$

Where:
- $r_i(d)$ = rank of document $d$ in ranked list $i$ (1-indexed)
- $K = 60$ (empirically optimal constant from Cormack et al., 2009)

**Properties of RRF**:
1. **Scale-agnostic**: Only ranks matter, not raw scores
2. **Robust to outliers**: A document ranked #1 by one system but #50 by another still gets solid combined score
3. **No hyperparameter tuning**: $K=60$ works well across almost all datasets
4. **Easy to extend**: Add a third retriever (ColBERT) simply by adding another term to the sum

### 3.3 RRF Numerical Analysis

For $K=60$ and a 5-document corpus:

| Rank | RRF Score | Marginal gain vs prev rank |
|---|---|---|
| 1 | 1/(60+1) = 0.01639 | — |
| 2 | 1/(60+2) = 0.01613 | 0.00026 |
| 3 | 1/(60+3) = 0.01587 | 0.00026 |
| 5 | 1/(60+5) = 0.01538 | — |

Note how small the differences are — this is intentional. RRF is **democratic**: rank 1 is only slightly better than rank 2. This prevents a mediocre but consistently-ranked document from being beaten by a document ranked #1 in one system and #50 in another.

---

## 4. Re-Ranking: Cross-Encoders

### 4.1 Why Bi-Encoders Can't Do Everything

The key architectural difference:

| Architecture | How Query Meets Document | Cost |
|---|---|---|
| **Bi-Encoder** | They never meet — encoded independently | $O(1)$ per query (docs pre-encoded) |
| **Cross-Encoder** | Concatenated as `[CLS] query [SEP] document [SEP]` | $O(n)$ per query (must process each pair) |

Cross-encoders have full **cross-attention** between query and document tokens. Every query token attends to every document token and vice versa — this produces much more precise relevance signals.

### 4.2 The 2-Stage Strategy

```
Stage 1: Bi-Encoder      Stage 2: Cross-Encoder
────────────────────     ────────────────────────
Fast recall              Precise re-ranking
Retrieve top-100         Re-score top-100
~1ms                     ~200ms
Optimize for recall      Optimize for precision
```

This is the architecture used inside Bing, Google, and most enterprise search systems.

### 4.3 Popular Cross-Encoder Models

| Model | Training Data | Use Case |
|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | MS MARCO (web passages) | General purpose, fast |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | MS MARCO | Higher quality, 2× slower |
| `BAAI/bge-reranker-large` | Mixed multilingual | High quality, large |

---

## Teacher's Key: Expected Learning Outcomes

- **Formula**: Write the BM25 formula and explain what $k_1$ and $b$ control.
- **Comparison**: Explain the difference between bi-encoder and cross-encoder architecturally.
- **Math**: Given ranks from two retrievers, compute the RRF score manually.
- **Design decision**: When would you use a cross-encoder in a production system? When would you not?
