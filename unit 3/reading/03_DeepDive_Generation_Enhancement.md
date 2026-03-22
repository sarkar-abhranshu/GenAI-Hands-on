# Deep Dive: Unit 3 — Part 3: Generation Enhancement Techniques

## 1. Fine-Tuning: The Spectrum

There is a continuous spectrum from "use the model as-is" to "retrain it from scratch":

```
Prompt Engineering → RAG → PEFT/LoRA → Full Fine-Tuning → Pre-Training from Scratch
──────────────────   ───   ──────────   ────────────────   ──────────────────────────
No training         No     <1% params   100% params         100% params + data
Cheapest            training  updated   updated             Most expensive
```

### 1.1 When to Choose Each

| Scenario | Recommended Approach |
|---|---|
| Need model to speak in a specific tone/style | Fine-Tuning |
| Need model to know specific facts | RAG |
| Need model to follow a specific output format | Prompt Engineering |
| Need model to perform a new type of reasoning task | Fine-Tuning |
| Facts change frequently (news, stock prices) | RAG |
| Domain uses specialized vocabulary the base model doesn't know | Fine-Tuning |

### 1.2 LoRA: The Math in Depth

**Motivation**: A weight matrix $W \in \mathbb{R}^{d \times d}$ has $d^2$ parameters. For $d=4096$ (LLaMA-2 hidden size), that is 16.7 million parameters *per matrix*. A full LLaMA-2 7B has ~224 such matrices — you cannot update all of them on a consumer GPU.

**LoRA hypothesis**: The weight change $\Delta W$ during fine-tuning has **low intrinsic rank**. That is, $\Delta W$ can be well-approximated by a product of two small matrices:

$$\Delta W \approx BA, \quad B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times d}$$

For $r=8$ and $d=4096$:
- $\Delta W$: 16,777,216 parameters (if trained directly)
- $BA$: 2 × 4096 × 8 = 65,536 parameters (0.39% of $\Delta W$)

**Initialization**:
- $A$ is initialized with random Gaussian values
- $B$ is initialized to **zero** → $\Delta W = BA = 0$ at the start of training
- This means the LoRA model behaves identically to the original model at initialization

**Scaling**:
$$h = Wx + \frac{\alpha}{r} \cdot BAx$$

The $\frac{\alpha}{r}$ scaling factor prevents the learning rate from needing to change when $r$ changes. Common: $\alpha = 2r$.

**LoRA ranks and use cases**:

| Rank $r$ | Trainable params (d=4096) | Use case |
|---|---|---|
| 1 | ~8K | Extremely constrained, minimal adaptation |
| 4 | ~32K | Lightweight style adaptation |
| 8 | ~65K | Standard (most benchmarks use this) |
| 16 | ~131K | Complex task adaptation |
| 64 | ~524K | Near full-rank adaptation |

**QLoRA**: Quantize the base model to INT4 (4-bit), then train LoRA adapters in BF16. This allows fine-tuning a 13B model on a single 24GB GPU.

---

## 2. Data Precision

### 2.1 IEEE 754 Floating-Point Formats

All modern neural networks use IEEE 754 floating-point arithmetic. The bit layout:

```
FP32 (32 bits):
┌─┬────────┬───────────────────────┐
│S│EEEEEEEE│MMMMMMMMMMMMMMMMMMMMMMM│
└─┴────────┴───────────────────────┘
 1    8 bits           23 bits
Sign  Exponent          Mantissa
Value = (-1)^S × 2^(E-127) × (1 + M/2^23)

FP16 (16 bits):
┌─┬─────┬──────────┐
│S│EEEEE│MMMMMMMMMM│
└─┴─────┴──────────┘
 1   5       10
Max value: 65,504

BF16 (16 bits):
┌─┬────────┬───────┐
│S│EEEEEEEE│MMMMMMM│
└─┴────────┴───────┘
 1     8        7
Same exponent as FP32 → same range, less precision
```

### 2.2 Why the Format Matters for LLMs

**Training**:
- FP32: Stable, accurate gradients. Memory-intensive (4 bytes/param).
- FP16: Risk of overflow/underflow in gradient computation. Requires **loss scaling**.
- BF16: Same exponent range as FP32, stable training. Used by Llama 3, Gemma, Mistral.

**Inference**:
- Weights are frozen. Less precision is acceptable — tiny rounding errors average out over millions of multiplications.
- INT8 inference (LLM.int8): quantize weights, dequantize on-the-fly, ~1% accuracy drop.

### 2.3 Mixed Precision Training

Production LLM training uses **mixed precision**:
1. Weights stored in FP32 (the "master copy")
2. Forward + backward pass computed in FP16/BF16
3. FP32 master weights updated at the end of each batch

This gives the speed of FP16 with the numerical stability of FP32.

---

## 3. Quantization

### 3.1 Types of Quantization

| Type | When Applied | Description |
|---|---|---|
| **Post-Training Quantization (PTQ)** | After training | Quantize a pre-trained model. No additional training. |
| **Quantization-Aware Training (QAT)** | During training | Simulate quantization noise during training → model learns to be robust to it |
| **Dynamic Quantization** | At inference time | Quantize activations on-the-fly per batch |

Most LLM quantization is PTQ because LLMs are too expensive to retrain.

### 3.2 Linear (Affine) Quantization Math

To quantize a tensor $X$ from FP32 to INT8:

**Determine range**:
$$x_{\min} = \min(X), \quad x_{\max} = \max(X)$$

**Compute scale and zero-point**:
$$S = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}} = \frac{x_{\max} - x_{\min}}{255}$$
$$Z = \text{round}\left(q_{\min} - \frac{x_{\min}}{S}\right) = \text{round}\left(-128 - \frac{x_{\min}}{S}\right)$$

**Quantize**:
$$X_q = \text{clamp}\left(\text{round}\left(\frac{X}{S}\right) + Z, -128, 127\right)$$

**Dequantize** (reconstruct to FP32):
$$\hat{X} = S \cdot (X_q - Z)$$

**Quantization error** (loss of precision):
$$\epsilon = X - \hat{X} \quad \text{(bounded by } \frac{S}{2}\text{)}$$

### 3.3 Symmetric vs Asymmetric Quantization

**Symmetric**: Zero-point $Z=0$. Simpler math, good for weights (often zero-centered).
$$X_q = \text{round}\left(\frac{X}{S}\right), \quad S = \frac{\max(|X|)}{127}$$

**Asymmetric**: $Z \neq 0$. Better for activations (ReLU outputs are always ≥ 0, so their range is [0, x_max]).

### 3.4 GPTQ Algorithm

GPTQ applies a key insight: when quantizing layer $l$, the quantization error propagates forward. GPTQ compensates for each weight's quantization error by adjusting the remaining unquantized weights in the same row.

This is done using a second-order optimization approach (Hessian-based), resulting in much lower accuracy degradation than naive rounding — enabling INT4 models that perform similarly to FP16.

**Memory requirements for serving a 70B model**:

| Format | Memory | Fits on |
|---|---|---|
| FP32 | 280 GB | 4× A100 80GB |
| FP16 | 140 GB | 2× A100 80GB |
| INT8 | 70 GB | 1× A100 80GB |
| INT4 (GPTQ) | 35 GB | 1× A100 40GB |
| INT4 (GPTQ) | 35 GB | 2× 3090 24GB |

---

## 4. Mixture of Experts (MoE)

### 4.1 The Dense Model Problem

In a standard transformer FFN (which processes each token):
$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x)$$

Both $W_1$ and $W_2$ activate for **every single token**. A 70B model uses 70B parameters for the word "the" just as much as for a complex technical term.

### 4.2 MoE Architecture

MoE replaces the single FFN with $N$ expert FFNs and a router:

$$\text{MoE}(x) = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)$$

Where:
- $G(x) = \text{Softmax}(\text{TopK}(W_g \cdot x))$ — the gating/routing network
- $E_i(x)$ — the $i$-th expert FFN
- Only Top-K (typically K=1 or K=2) experts have non-zero $G(x)_i$

**The computation savings**:
- Mixtral 8x7B: 8 experts, select top-2 per token
- Each expert is a 7B-parameter FFN
- Total params: ~47B, but active params per token: ~13B
- Inference cost ≈ a 13B dense model, but knowledge capacity ≈ a 47B model

### 4.3 Router Collapse and Load Balancing

A naive router will quickly learn to send all tokens to the same 1-2 experts (the ones that got luckily initialized slightly better). This is called **router collapse** — most experts never learn.

**Fix: Auxiliary Load Balancing Loss**

$$\mathcal{L}_{\text{balance}} = \alpha \sum_{i=1}^{N} f_i \cdot P_i$$

Where $f_i$ = fraction of tokens routed to expert $i$, and $P_i$ = average router probability for expert $i$.

Minimizing this loss encourages uniform distribution of tokens across experts.

### 4.4 Software-Level MoE vs Model-Level MoE

| Aspect | Software MoE (LangChain) | Model MoE (Mixtral) |
|---|---|---|
| Where routing happens | Application code | Inside transformer layers |
| Granularity | Per query | Per token |
| Expert differentiation | Different system prompts | Different learned weight matrices |
| Cost | Multiple API calls | One API call |
| Customization | Easy (just prompts) | Requires training |

For production applications, software-level MoE is extremely practical — it's what powers many enterprise chatbot systems.

---

## Teacher's Key: Expected Learning Outcomes

- **LoRA**: Explain the ΔW = BA decomposition. Why does rank matter? What is the effect of initializing B=0?
- **Precision**: What is the difference between FP16 and BF16? Why did LLM training shift to BF16?
- **Quantization**: Given a weight vector, manually compute INT8 quantization (scale, zero-point, quantized values).
- **MoE**: Explain the MoE forward pass formula. What is router collapse and how is it prevented?
- **Trade-offs**: For a startup deploying a 70B model with limited GPU budget, which technique would you apply first and why?
