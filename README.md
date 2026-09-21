# Softuni — AI Agents and Workflows

Coursework for the SoftUni *AI Agents and Workflows* module. Everything is written as
Google Colab notebooks using **LangChain / LangGraph** with OpenAI models
(`gpt-4.1-mini`) and a **Chroma** vector store.

---

## Main project — Meal Assistant (`exam/exam_multi_agent_system.ipynb`)

A meal-planning multi-agent system. The user describes what they want in plain language
("vegan dinners for 3 days, family of 4, allergic to tree nuts") and the system returns an
**approved meal plan**, a **costed consolidated shopping list** and a **nutrition summary** —
grounded entirely in a recipe knowledge base, never in the model's memory.

It extends the recipe-retrieval project from the previous course into a full LangGraph
workflow with subgraphs, deterministic validation, human-in-the-loop approval and
parallel deliverables.

### Knowledge base

`recipes.md` is split with `MarkdownHeaderTextSplitter` into per-recipe records
(name, country, region, description, ingredients, preparation, fact), validated through a
`Recipe` Pydantic model and stored in a persisted **Chroma** collection (`recipes_db`),
exposed as a retriever with `k=12`.

### Agents and nodes

| Component | Kind | Tools |
|---|---|---|
| **Meal Planner** | agent (tool loop) | `recipes_knowledge_retriever`, `filter_recipes` |
| **Shopping List Builder** | agent (tool loop) | `get_recipe_ingredients`, `estimate_ingredient_prices` |
| **Preferences Parser** | plain structured-output LLM node | — |
| **Nutrition Analyst** | plain structured-output LLM node | — |
| **Plan Validator** | ordinary Python, no LLM | — |

The parser and the analyst take no decisions and need no tools, so they are structured-output
nodes rather than agents.

### Tools

- `recipes_knowledge_retriever` — semantic search over the recipe KB, called multiple times
  with different queries to gather variety for a full plan.
- `filter_recipes` — semantic search restricted by country **or** region/cuisine metadata.
- `get_recipe_ingredients` — exact-name lookup returning parsed `{name, quantity, unit}`
  lines (handles unicode fractions and common units); returns an error instead of
  letting the model invent ingredients.
- `estimate_ingredient_prices` — mock pricing service over a static price table with a
  flagged default fallback.

### Deterministic validation

`plan_validator` is plain Python, not an LLM. It checks day count, repeated recipes,
recipes absent from the knowledge base, **allergen violations** (synonym-expanded,
word-boundary matching so `nut` doesn't fire on `coconut`) and **diet violations**
(vegetarian / vegan / pescatarian / gluten-free / no sugar). The planner gets up to
`MAX_VALIDATION_ATTEMPTS` self-corrections before an imperfect plan is handed to the
human with the problems attached.

### Human in the loop

Two `interrupt()` points:

1. **Allergy check** — raised when `allergies_confirmed` is false. "No allergies mentioned"
   is treated as *unknown*, not as *no allergies*.
2. **Plan approval** — approve, or reject with feedback that is fed back into the planner,
   up to `MAX_REVISIONS` rounds.

### Graph shape

A **Meal Planner subgraph** (`meal_planner → plan_validator → meal_plan_approver`, with
loops back to the planner) is embedded as a single node in the main graph, so "produce an
approved plan" is one step:

```
START → dietary_preferences →(allergies unconfirmed) allergies_clarification
      → meal_planner_subgraph
      → ┬ shopping_list_builder ┬ → final_output → END
        └ nutrition_summary ────┘
```

Once a plan is approved the graph **fans out**: the shopping list and the nutrition summary
are independent, run concurrently, and fan back into `finalize`.

### Memory

- `InMemorySaver` checkpointer — per-thread conversation state and interrupt/resume.
- `InMemoryStore` — dietary preferences remembered per `user_id` across sessions and merged
  with newly parsed ones

### Running it

`execute_workflow(user_request, user_id, human_input=None, thread_id=None)` is the entry
point: it starts the graph on a fresh thread, answers every interrupt (interactively via
`input()`, or from a supplied simulator in the tests) with `Command(resume=...)`, then prints
the plan, the grouped shopping list, tool usage, and returns the final state.

Requires `OPENAI_API_KEY` as a Colab secret, and `recipes.md` at `/content/` (otherwise it
prompts for an upload).

### Test scenarios

- Approval on the first try
- One revision round
- Allergies as a hard safety constraint 
- Repeated rejection hitting the retry cap
- The store-backed memory (same
user, second session)
- An allergy supplied at the clarification prompt
- A dinner-only light plan
- Planning around given ingredients
- Filtering by preparation time

---

## Exercises

### `exercises/langChain_agent_tools.ipynb` — agents, tools and RAG
A customer-support agent over an FAQ knowledge base. Loads `FAQ.md`, splits it by markdown
headers, indexes it in Chroma and wraps the retriever with `create_retriever_tool`. Adds
`@tool` functions against a mock internal order system (`customer_orders`, `lookup_order`)
and demonstrates middleware: `PIIMiddleware` for email redaction plus
`before_agent` / `before_model` / `after_model` / `after_agent` hooks.

### `exercises/langchain_memory_HITL.ipynb` — memory, HITL and evaluation
A luxury travel consultant agent. Covers short-term memory (checkpointer + thread ids),
long-term memory via store-backed `remember_preference` / `recall_preferences` tools scoped
to a `guest_id`, a custom context and state schema, and `HumanInTheLoopMiddleware` gating
the `book_offer` tool behind an approval interrupt. Finishes with **LangSmith** tracing plus
a dataset and an LLM-as-judge evaluation of the agent's behaviour.

### `exercises/langgraph_multi_agent_systems.ipynb` — LangGraph fundamentals
A minimal `StateGraph`: typed state, node functions, conditional routing vs. simultaneous
fan-out edges, graph visualisation with Mermaid, and inspecting checkpoints and state
history from the checkpointer.

---

## Setup

The notebooks are written for Google Colab (each installs its own dependencies with
`!pip install -q ...` and reads secrets via `google.colab.userdata`).

Secrets used: `OPENAI_API_KEY`, and `LANGSMITH_API_KEY` for the memory/HITL exercise.
