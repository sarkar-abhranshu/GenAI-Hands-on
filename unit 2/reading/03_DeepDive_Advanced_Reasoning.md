# 📘 Unit 2 Deep Dive: Part 3 - Advanced Reasoning (CoT, ToT, GoT)

## 1. The Reasoning Crisis
Simple LLMs take a "Single Pass" at a problem. They start generating tokens immediately. For complex logic, they often go down the wrong path and can't turn back. 

### 1.1 Fast vs. Slow Thinking (System 1 vs. System 2)
- **Standard Prompting:** System 1 (Fast, Intuitive, automatic).
- **Reasoning Frameworks:** System 2 (Slow, Analytical, deliberate).

---

## 2. Chain of Thought (CoT): The "Think Step-by-Step" Magic
In 2022, researchers discovered that adding the phrase **"Let's think step by step"** significantly boosted math and logic performance.

### 2.1 The Latent Space Argument
When a model "thinks", it generates **Intermediate Tokens**. 
- These tokens are appended to the conversation history.
- For the *next* token, the model's **Attention Mechanism** looks back at its own reasoning.
- **Result:** The reasoning acts as a "scratchpad" that helps the model compute more complex functions than it could in a single jump.

---

## 3. Tree of Thoughts (ToT): Exploring Branches
CoT is a single line. **ToT** is a deliberate search through a decision tree.

### 3.1 The Search Process
1.  **Thought Generation:** The model generates 3 distinct "Draft 1s".
2.  **Evaluation:** A "Judge" model (or the same model) evaluates the potential of each draft.
3.  **Selection:** We keep the "Winning" branch and iterate.

### 3.2 When to use it?
Used for "Creative Search" problems (e.g., crossword puzzles, complex coding architecture, high-stakes strategic planning).

---

## 4. Graph of Thoughts (GoT): Nonlinear Intelligence
GoT is the most advanced. It allows thoughts to:
- **Loop:** Re-visit an old idea.
- **Aggregate:** Combine the best parts of Solution A and Solution B into a new Solution C.

### 4.1 Implementation in LangChain
In the notebook, we used `RunnableParallel` to create "Divergent" paths and then piped them into a single "Convergent" prompt. This is a basic GoT pattern.

---

## 5. Self-Correction & Reflection
A key advanced technique is **Self-Reflection**.
1.  **Step 1:** Model generates an answer.
2.  **Step 2:** Model is asked: "Review your answer for any errors or bias."
3.  **Step 3:** Model produces the final, corrected version.

This "Multi-turn" reasoning is significantly more robust than any single prompt.

---

## 🔍 Teacher's Key: Expected Learning Outcomes
- **Concept:** Why does a small model like Llama-3-8b benefit MORE from CoT than a huge model like GPT-4?
- **Code:** Can the student implement a "Writer & Editor" pattern using two different chains?
- **Theory:** What is the difference between a linear Chain of Thought and a parallel Graph of Thoughts?
