# Assignment: Evaluated Agentic RAG System

## Overview

In this assignment you will build a **self-evaluating agentic RAG system**: a CrewAI pipeline where one agent retrieves and answers questions using RAG, a second agent evaluates the answer quality using DeepEval, and a third agent revises the answer if quality is below threshold. This combines the core skills from all three Unit 4 notebooks.

---

## Learning Objectives

By completing this assignment you will demonstrate:
1. Building a RAG pipeline with LangChain and FAISS
2. Implementing a multi-agent CrewAI workflow with custom tools
3. Using DeepEval metrics (Faithfulness, Answer Relevancy) programmatically
4. Implementing a **retry loop** — agents recover from low-quality outputs
5. Comparing quality before and after the evaluation-revision cycle

---

## The System You Will Build

```
USER QUESTION
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  AGENT 1: RAG Retriever                             │
│  - Searches the vector store for relevant docs      │
│  - Generates an initial answer using an LLM         │
│  - Outputs: (answer, retrieved_context)             │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  AGENT 2: Quality Evaluator                         │
│  - Runs DeepEval FaithfulnessMetric                 │
│  - Runs DeepEval AnswerRelevancyMetric              │
│  - Outputs: {faithfulness: 0.X, relevancy: 0.X,    │
│              verdict: PASS/FAIL, reasons: [...]}    │
└────────────────────────┬────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
           PASS                   FAIL
              │                     │
              ▼                     ▼
         Final Answer     ┌─────────────────────────┐
                          │  AGENT 3: Revisor        │
                          │  - Reads evaluator       │
                          │    feedback              │
                          │  - Re-generates a        │
                          │    corrected answer      │
                          │  - Grounded in context   │
                          └────────────┬────────────┘
                                       │
                                       ▼
                                 Final Answer (revised)
```

---

## Requirements

### Part 1: Knowledge Base (10 marks)

Build a knowledge base on a topic of your choice (minimum 500 words, 5+ distinct facts). Options:
- A Wikipedia article on a scientific topic
- A set of company policies or FAQ documents
- A research paper abstract + introduction
- A product manual

Load the text, split into chunks (size and overlap of your choice), and build a FAISS vector store with HuggingFace sentence-transformer embeddings.

**Deliverable**: Code that builds the vector store + a brief description of your chosen topic and why you chose it.

### Part 2: RAG Agent (20 marks)

Define a CrewAI `Agent` and `Task` that:
- Has a `@tool`-decorated function that queries your FAISS vector store
- Generates an answer using an LLM (Groq recommended)
- The task output must include **both the answer AND the retrieved context** (needed by the evaluator)

**Deliverable**: Working RAG agent + task definition with sample output for 3 test questions.

### Part 3: Quality Evaluator Agent (25 marks)

Define a CrewAI `Agent` and `Task` that:
- Takes the RAG output (answer + context) as input
- Runs `FaithfulnessMetric` and `AnswerRelevancyMetric` from DeepEval
- Outputs a structured verdict: scores, pass/fail (threshold = 0.7), and specific reasons for any failures

**Hint**: The evaluator's tool should wrap the DeepEval metric calls. The agent reads the RAG output from `context=[rag_task]`.

**Deliverable**: Working evaluator agent with sample evaluation output showing scores and reasons.

### Part 4: Revisor Agent (20 marks)

Define a CrewAI `Agent` and `Task` that:
- Activates only when the evaluator flags a FAIL
- Reads the original question, the failed answer, and the evaluator's specific failure reasons
- Produces a revised answer that addresses each identified issue
- The revision must be grounded in the retrieved context (no new hallucinations)

**Deliverable**: Working revisor agent with a side-by-side comparison: original failed answer vs. revised answer, with evaluator re-scoring of the revised answer.

### Part 5: Full Pipeline (15 marks)

Assemble the full crew:
1. Run the pipeline on **5 test questions** from your knowledge base
2. Run the pipeline on **2 adversarial questions** (questions where the answer is NOT in your knowledge base) — document how the system handles them
3. Track and report: initial pass rate, final pass rate after revision

**Deliverable**: Full pipeline execution with results table:

| Question | Initial Faithfulness | Initial Relevancy | Verdict | Final Faithfulness | Final Relevancy |
|---|---|---|---|---|---|
| Q1 | 0.XX | 0.XX | PASS/FAIL | 0.XX | 0.XX |
| ... | | | | | |

### Part 6: Reflection (10 marks)

Write a 200-300 word reflection answering:
1. What types of questions caused the most failures, and why?
2. How effective was the revision step? Did it consistently improve scores?
3. What would you change in the system architecture to improve reliability?
4. How would you extend this system with TruLens for ongoing monitoring?

---

## Evaluation Rubric

| Criterion | Weight | Description |
|---|---|---|
| Knowledge base quality | 10% | Sufficient content, well-chunked, meaningful topic |
| RAG agent correctness | 20% | Correctly retrieves and answers; context passed to evaluator |
| Evaluator accuracy | 25% | DeepEval integration works; scores make sense; reasons are specific |
| Revisor effectiveness | 20% | Revised answers demonstrably better; grounded in context |
| Full pipeline execution | 15% | All 7 questions run; results table complete |
| Reflection quality | 10% | Thoughtful, specific, references actual observed behavior |

---

## Submission

Submit a single Jupyter notebook (`unit4_assignment_<your_name>.ipynb`) that:
- Runs end-to-end without errors
- Has markdown cells explaining each section
- Includes your reflection at the end

---

## Tips

- **Start with Part 1 and test your RAG chain independently** before adding agents
- **Use `verbose=True`** on all agents to see what they're doing
- **The evaluator agent is tricky** — the DeepEval metrics need to be called inside a `@tool` function that receives the answer and context as string arguments
- **For the revision step**, the most important signal from the evaluator is the `reason` field from each metric — pass this to the revisor in the task description
- **Groq free tier is sufficient** — `llama-3.3-70b-versatile` works well for all steps
