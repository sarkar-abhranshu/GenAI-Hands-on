# Deep Dive: Multimodal Models — CLIP, BLIP & Whisper

---

## 1. The Multimodal Problem

### Why One Modality Is Not Enough

Early AI systems were strictly unimodal: a vision model processed images; a language model processed text; an audio model processed sound. Connecting them required hand-crafted pipelines where outputs were manually converted between representations.

This brittle architecture had fundamental limits:
- **No semantic alignment** between modalities — you couldn't ask "does this image match this description?" without training a separate classifier
- **Brittle pipelines** — failure in one stage cascades
- **No shared reasoning** — the vision model had no concept of language meaning

### The Shared Embedding Solution

The breakthrough was learning a **joint embedding space** — a vector space where semantically related concepts from different modalities end up close together, regardless of whether they came from text, images, or audio.

```
Shared Embedding Space (512 dimensions):

  "a cat sitting on a sofa" ─────────────────► [0.2, -0.8, 0.5, ...]
                                                        ≈ (cosine similarity ~0.95)
  [image: cat on sofa] ──────────────────────► [0.2, -0.7, 0.6, ...]

  "a dog running outside" ───────────────────► [-0.3, 0.9, -0.2, ...]
                                                        ≈ (cosine similarity ~0.97)
  [image: dog in park] ──────────────────────► [-0.3, 0.8, -0.1, ...]
```

---

## 2. CLIP — Contrastive Language-Image Pretraining

### Architecture

CLIP consists of two independent encoders trained jointly:

```
CLIP Architecture:

  Text:   "a photo of a cat"
              │
         Text Tokenizer
              │
         Transformer Encoder (text)
              │
         Text Embedding  ─────────────────────────┐
                                                   │
                                              Cosine Sim
                                                   │
  Image:  [cat photo]                              │
              │                                    │
         Patch Embedding (16×16 patches)           │
              │                                    │
         Vision Transformer (ViT)                  │
              │                                    │
         Image Embedding ──────────────────────────┘
```

### Contrastive Training Objective

CLIP is trained on (image, text) pairs. For a batch of N pairs, the training objective creates an N×N similarity matrix and uses **InfoNCE loss**:

```
For a batch of N = 4 pairs:

  Text embeddings:  T₁ T₂ T₃ T₄
  Image embeddings: I₁ I₂ I₃ I₄

  Similarity matrix:
           I₁   I₂   I₃   I₄
  T₁  [ 0.95  0.12  0.08  0.15 ]   ← T₁ should match I₁ (high sim)
  T₂  [ 0.10  0.92  0.11  0.09 ]   ← T₂ should match I₂
  T₃  [ 0.07  0.14  0.89  0.11 ]   ← T₃ should match I₃
  T₄  [ 0.09  0.11  0.13  0.93 ]   ← T₄ should match I₄

  Loss = cross-entropy(rows) + cross-entropy(columns)
         ↑ text-to-image           ↑ image-to-text
```

The diagonal should be maximized; all off-diagonal elements minimized. This is the **InfoNCE (Noise Contrastive Estimation)** loss.

### Zero-Shot Transfer

The key insight: CLIP never explicitly trained for "classify images into 1000 categories". But because it learned from natural language, you can create text prompts for any category and compute similarity:

```python
labels = ["cat", "dog", "car"]
prompts = [f"a photo of a {label}" for label in labels]

# CLIP computes similarity between the image and each prompt
# → zero-shot classification with no training data
```

This generalizes to categories that didn't exist during training — CLIP can classify images of things described in natural language it has never seen paired with images.

### Prompt Engineering for CLIP

The exact prompt template significantly affects CLIP's zero-shot accuracy:

| Template | ImageNet Accuracy |
|---|---|
| `{class_name}` | 60.3% |
| `A photo of a {class_name}` | 63.4% |
| `A photo of a {class_name}, a type of pet` | 65.1% |
| Ensemble of 80 templates | 76.2% |

Ensembling multiple prompt templates and averaging the resulting embeddings is a simple but effective technique.

### Limitations of CLIP

1. **No text generation** — CLIP cannot describe an image; it can only score similarity
2. **Compositionality failures** — "a red cube on top of a blue sphere" vs "a blue cube on top of a red sphere" — CLIP often scores these similarly (bag-of-words problem)
3. **Bias** — trained on web data, inherits representational biases
4. **Short text** — best with short descriptions; degrades with long paragraphs

---

## 3. BLIP — Bootstrapped Language-Image Pretraining

### Architecture

BLIP extends CLIP's alignment capability with a full **generative** component:

```
BLIP Architecture (three components):

  1. IMAGE ENCODER (ViT)
     Processes the image → image features

  2. TEXT ENCODER (BERT-like, causal)
     Processes text with image cross-attention
     Used for: Image-Text Matching (ITM) and Image-Text Contrastive (ITC)

  3. TEXT DECODER (autoregressive)
     Generates text conditioned on image features
     Used for: Image Captioning and VQA
```

The three components share parameters but serve different tasks depending on which attention masks are applied.

### The Bootstrapping Innovation

A key challenge in multimodal training: web image-text pairs are **noisy**. The image and its alt-text often don't actually match (e.g., an image of a dog with alt-text "Buy cheap dog food here").

BLIP solves this with a two-stage bootstrapping approach:

```
Stage 1: Pre-train on noisy web data

Stage 2: Bootstrap
  ├─ Captioner (BLIP fine-tuned on clean COCO data)
  │     generates synthetic captions for web images
  │
  └─ Filter (ITM module)
        scores each (image, caption) pair
        removes low-quality pairs
        → CapFilt dataset

Stage 3: Re-train on CapFilt + original clean data
  → significantly better model
```

This self-improvement loop is what gives BLIP its name — the model bootstraps its own training data quality.

### BLIP-2 Improvements

BLIP-2 (2023) introduced the **Q-Former** (Querying Transformer):
- A lightweight module that extracts relevant visual features from a frozen image encoder
- Bridges the visual encoder with any large language model (OPT, FlanT5)
- Only the Q-Former is trained — the image encoder and LLM stay frozen → extremely efficient

```
BLIP-2 Architecture:

  [Image] → [Frozen ViT] → [Q-Former] → [Frozen LLM] → [Text Output]
                              ↑
                        Only this is trained
                        (188M parameters vs billions in ViT/LLM)
```

---

## 4. Whisper — Robust Speech Recognition

### Architecture

Whisper is a sequence-to-sequence Transformer trained directly on (audio spectrogram → text transcript) pairs:

```
Whisper Architecture:

  Raw Audio Waveform (16kHz)
         ↓
  Log-Mel Spectrogram
  (80 mel bins, 30-second window, 25ms frame stride)
         ↓
  2× Convolutional layers (local feature extraction)
         ↓
  Transformer Encoder (processes audio representations)
         ↓
  Transformer Decoder (autoregressively generates tokens)
         ↓
  Token sequence → Transcript
```

### Multi-task Training

Whisper was trained simultaneously on multiple tasks by prepending task-specific tokens:

```
Token sequence:
  [<|startoftranscript|>] [<|en|>] [<|transcribe|>] [<|notimestamps|>]
  ↑ task start             ↑ lang   ↑ task type       ↑ timestamp mode

  For translation:
  [<|startoftranscript|>] [<|fr|>] [<|translate|>] ...
  → transcribes French audio and translates to English
```

This unified multi-task training is why Whisper generalizes so well — it saw the same audio in many contexts during training.

### Training Data Scale

| Whisper data | Detail |
|---|---|
| Total hours | 680,000 hours |
| Languages | 99 |
| Data source | Internet (weakly supervised — audio + transcript pairs) |
| Annotation quality | Variable (automatic and human) |

The massive scale and diversity explain Whisper's robustness to accents, noise, and domain-specific vocabulary. Earlier ASR models (trained on tens of thousands of hours of studio audio) fail catastrophically on spontaneous speech; Whisper handles it gracefully.

### Word Error Rate Benchmarks

| Model | LibriSpeech (clean) | LibriSpeech (other) |
|---|---|---|
| wav2vec 2.0 (fine-tuned) | 1.9% | 3.9% |
| Whisper medium | 3.0% | 5.7% |
| Whisper large-v3 | 2.0% | 4.2% |
| Human performance | ~5.8% | ~12.7% |

Whisper large-v3 approaches human performance on clean audio — remarkable for a zero-shot model with no task-specific fine-tuning.

---

## 5. The Future of Multimodal Models

### GPT-4o and the "Any-to-Any" Vision

The newest frontier models are **natively multimodal** — trained on text, images, and audio together from scratch (not bolted together):

```
GPT-4o (2024) — end-to-end multimodal:

  Input:  [Text] [Image] [Audio] — any combination
               ↓
         Shared Transformer (single model)
               ↓
  Output: [Text] [Image] [Audio] — any combination
```

This differs from chaining CLIP + BLIP + Whisper:
- **Lower latency** — no pipeline overhead
- **Richer cross-modal reasoning** — the model can attend across modalities in every layer
- **Emergent capabilities** — e.g., describing emotions in someone's voice, reading charts and reasoning about them simultaneously

### Key Models to Watch

| Model | Developer | Modalities | Notes |
|---|---|---|---| 
| GPT-4o | OpenAI | Text, Image, Audio, Video | Real-time voice and vision |
| Gemini 1.5 Pro | Google | Text, Image, Audio, Video, Code | 1M token context |
| Claude 3.5 Sonnet | Anthropic | Text, Image | Strong vision reasoning |
| LLaVA | Various | Text, Image | Open-weights visual instruction tuning |
| Qwen-Audio | Alibaba | Text, Audio | Strong open-weights ASR+LLM |
