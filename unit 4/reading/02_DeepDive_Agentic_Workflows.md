# Deep Dive: Agentic Workflows — Design Patterns, AutoGen & CrewAI

---

## 1. What Makes an AI "Agentic"?

A standard LLM call is **stateless and single-turn**: one prompt in, one response out, no memory of previous interactions, no ability to take actions in the world.

An **agent** is an LLM equipped with:

1. **A loop** — the ability to think, act, observe, and think again
2. **Tools** — Python functions the LLM can call (web search, calculators, APIs, file I/O)
3. **Memory** — storage of previous observations and actions (short-term: conversation history; long-term: vector DB)
4. **Planning** — the ability to break a goal into sub-steps

```
STANDARD LLM:           AGENT:
  Prompt → Response     Goal
                          ↓
                        THINK: What should I do?
                          ↓
                        ACT: Call tool / generate text
                          ↓
                        OBSERVE: What did the tool return?
                          ↓
                        THINK: Am I done? If not, what next?
                          ↓
                        (repeat until goal reached)
```

---

## 2. The ReAct Pattern

**ReAct** (Reasoning + Acting), introduced by Yao et al. (2022), is the foundational agent loop. The LLM interleaves natural language reasoning ("Thought") with tool calls ("Action") and results ("Observation"):

```
Thought: I need to find the current price of Bitcoin.
Action: search_web("Bitcoin price today")
Observation: "Bitcoin is trading at $67,430 as of 2:14 PM EST"

Thought: I have the current price. Now I need to compare it to yesterday's price.
Action: search_web("Bitcoin price yesterday")
Observation: "Bitcoin closed at $65,800 yesterday"

Thought: I can now compute the change.
Action: calculator("(67430 - 65800) / 65800 * 100")
Observation: 2.478...

Thought: I have all the information I need.
Final Answer: Bitcoin rose 2.5% from $65,800 to $67,430.
```

The key insight: by making reasoning **explicit and visible** in the context window, the model can course-correct, handle errors, and make multi-step decisions that a single prompt could not.

---

## 3. Agentic Design Patterns (Agentic AI by Andrew Ng)

### Pattern 1: Reflection

The agent critiques its own outputs and iterates.

```
Generator Agent:    "Here is my answer: ..."
Critic Agent:       "This answer has errors in section 2. Specifically: ..."
Generator Agent:    "You're right. Revised answer: ..."
Critic Agent:       "This looks correct. Approved."
```

**Implementation**: Can be a single agent with a two-step prompt ("generate", then "critique your own output"), or two separate agents (Generator + Critic) in conversation.

**When to use**: Code generation, essay writing, complex analysis where first drafts are typically imperfect.

### Pattern 2: Tool Use

The agent has access to external functions it can call programmatically.

```python
# Tools available to the agent:
tools = [
    web_search,          # searches the internet
    python_repl,         # executes Python code
    file_read,           # reads files from disk
    send_email,          # sends an email
    database_query,      # queries a SQL database
]
```

The LLM decides **when** to use a tool, **which** tool to call, and **what arguments** to pass — just like a human deciding to use a calculator instead of doing mental arithmetic.

### Pattern 3: Planning

The agent explicitly plans before executing.

```
TASK: Write a comprehensive report on climate change impacts on agriculture.

PLAN:
  Step 1: Research recent statistics on crop yield changes due to climate
  Step 2: Research geographical regions most affected
  Step 3: Research adaptation strategies used by farmers
  Step 4: Research economic projections
  Step 5: Synthesize into report with introduction, 4 sections, conclusion, citations

EXECUTE: Run each step in order, using search tools for steps 1-4.
```

**Sub-types**:
- **BabyAGI style**: Creates a task list, executes, adds new tasks discovered during execution
- **Tree of Thoughts**: Generates multiple plan branches, evaluates, prunes, and executes the best path

### Pattern 4: Multi-Agent Collaboration

Multiple specialized agents each handle what they're best at.

```
WHY SPECIALIZE?

  Single generalist agent:
  "Write code, test it, document it, and review it" → mediocre at all four

  Specialized agents:
  Architect → Programmer → Tester → Reviewer
  Each agent has its own:
    - system prompt tailored to its role
    - tools relevant to its job
    - evaluation criteria for its output
  → significantly better results
```

**Key patterns within multi-agent systems**:
- **Pipeline** (sequential): A → B → C (output of A feeds into B)
- **Hierarchical** (orchestrator-worker): Manager agent delegates to specialist agents
- **Debate**: Multiple agents argue for different positions; a judge agent decides
- **Peer review**: All agents review each other's work

---

## 4. AutoGen Architecture

### Core Design Philosophy

AutoGen models agent interaction as **conversations**. Two agents talk to each other via messages. The conversation continues until a termination condition is met (e.g., one agent says "TERMINATE", or `max_turns` is reached).

```
AutoGen Message Flow:

  UserProxy ─────── message ──────────► AssistantAgent
                                              │
                                    LLM generates response
                                    (may include tool_call)
                                              │
  UserProxy ◄────── response ────────────────┘
       │
  (if tool_call: execute tool, return result)
       │
  UserProxy ─────── tool result ──────► AssistantAgent
                                              │
                                    LLM generates next response
                                              │
  UserProxy ◄────── final response ──────────┘
       │
  (if "TERMINATE" in response: stop)
```

### Agent Types

| Agent | LLM | Human Input | Role |
|---|---|---|---|
| `ConversableAgent` | Optional | Optional | Base class, fully configurable |
| `AssistantAgent` | Yes | Never | Generates responses, calls tools |
| `UserProxyAgent` | No | Optional | Executes code/tools, represents human |
| `GroupChatManager` | Yes | Never | Orchestrates group conversations |

### Tool Registration

```python
register_function(
    my_python_function,           # the actual Python function
    caller=assistant_agent,       # which agent decides to call it
    executor=user_proxy,          # which agent executes it
    name="function_name",         # name LLM uses to call it
    description="What it does"    # natural language description for LLM
)
```

The separation of **caller** (decides) and **executor** (runs) is a security feature — the LLM decides to call the tool, but a separate process actually executes it (preventing direct code injection).

---

## 5. CrewAI Architecture

### Core Design Philosophy

CrewAI models agent collaboration as a **professional team**. Each crew member has a defined role, goal, and backstory. Tasks are explicitly declared with dependency chains.

```
CrewAI Object Model:

  Agent(
    role="Senior Data Analyst",          # professional title
    goal="Produce accurate analysis",    # what they optimize for
    backstory="10 years experience...",  # personality/expertise context
    tools=[search, calculator],          # what tools they have
    llm=groq_llm                         # which LLM they use
  )

  Task(
    description="Analyze Q3 sales data",  # what to do
    agent=data_analyst,                   # who does it
    context=[data_collection_task],       # depends on previous tasks
    expected_output="A summary report"    # what success looks like
  )

  Crew(
    agents=[analyst, writer],
    tasks=[analysis_task, writing_task],
    process=Process.sequential           # or hierarchical
  )

  crew.kickoff()  # runs all tasks in declared order
```

### Process Types

**Sequential** (default):
```
Task 1 → Task 2 → Task 3
Output of each task fed as context to the next.
```

**Hierarchical**:
```
Manager Agent
  ├── delegates to Worker 1
  ├── delegates to Worker 2
  └── synthesizes results
Manager controls task assignment dynamically.
```

### Context Injection

CrewAI's `context=[task]` is more powerful than it appears. When `Task B` has `context=[Task A]`, CrewAI automatically injects the full output of Task A into the prompt for Task B:

```
Task B prompt (injected by CrewAI):
  [Task B description]

  Context from previous tasks:
  ---
  [Full output of Task A]
  ---
  Use this context to complete your task.
```

This means agents downstream in the pipeline have full visibility into what earlier agents produced — without any manual orchestration code.

---

## 6. AutoGen vs CrewAI vs LangGraph

| Dimension | AutoGen | CrewAI | LangGraph |
|---|---|---|---|
| **Mental model** | Agents in a chat | Professional crew | State machine / graph |
| **Control flow** | Conversation-driven | Task dependency graph | Explicit graph nodes/edges |
| **State management** | Conversation history | Task outputs | Typed state object |
| **Tool integration** | `register_function` | `@tool` decorator | LangChain tools |
| **Best for** | Flexible back-and-forth reasoning | Structured pipelines with clear roles | Complex state machines, cycles |
| **Debugging** | Chat transcripts | Verbose agent logs | Graph visualization |
| **Learning curve** | Low | Low | Medium |

### When to use each

- **AutoGen**: When agents need to converse iteratively and the conversation structure can't be predetermined (debate, negotiation, collaborative problem solving)
- **CrewAI**: When you have a clear set of roles and a sequential or hierarchical workflow (content pipelines, software development, research pipelines)
- **LangGraph**: When you need explicit state management, cycles (retry loops), branching logic, and fine-grained control over agent behavior

---

## 7. Production Considerations

### Failure Modes

| Issue | Cause | Mitigation |
|---|---|---|
| **Infinite loops** | Agent never reaches termination condition | Set `max_turns`, use explicit "TERMINATE" check |
| **Tool hallucination** | Agent invents tool arguments | Validate arguments before execution; use strict JSON schemas |
| **Context overflow** | Long conversations exceed context window | Summarize older messages; use long-context models |
| **Cost explosion** | Too many LLM calls | Estimate cost before running; use cheaper models for simple steps |
| **Prompt injection** | Malicious content in tool results | Sanitize tool outputs before injecting into prompts |

### Cost Estimation

```python
# Rough cost calculation for a 3-agent pipeline
steps_per_task = 5          # average tool calls + LLM responses
tokens_per_step = 2000      # input + output tokens per step
num_tasks = 3               # number of tasks

total_tokens = steps_per_task * tokens_per_step * num_tasks
# = 30,000 tokens per pipeline run

# Groq (free tier): essentially free for prototyping
# GPT-4o: ~$0.005 per 1K tokens → $0.15 per run
# At 1000 runs/day → $150/day
```

Always prototype with a cheap/free model (Groq, Ollama) before switching to a production model.

### Observability

For production agentic systems, you need visibility into what agents are doing:
- **LangSmith** — traces all LLM calls, token counts, latencies
- **AgentOps** — purpose-built for agent monitoring
- **TruLens** — evaluation-focused (covered in Notebook 3)
- **Logging** — at minimum, log every tool call and response
