# Agent API

A LangGraph-powered orchestration layer that coordinates several specialist agents to answer revision-style questions, fetch fresh facts, run deep MCP research, and draft study mind maps. The current build focuses on the Chainlit playground experience; a production HTTP/React surface is on the roadmap.

## Quick Start

```bash
uv venv --python 3.11
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv sync

# Provide your API keys (OpenAI, Tavily, optional Gemini) in .env
cp .env.example .env && edit .env

# Launch the Chainlit UI
uv run chainlit run interface/chainlit_ui.py
```

Sample interactions and agent traces are available under `examples/`:

```bash
uv run python examples/supervisor_service_demo.py
uv run python examples/test.py          # mind map reference workflow
```

Run the test suite with:

```bash
uv run pytest -q
```

## Minimal Layout

```
agent-api/
├─ src/lastminute_api/
│  ├─ application/agent_service/
│  │  ├─ graph.py         # LangGraph wiring
│  │  ├─ node.py          # Supervisor + worker nodes
│  │  ├─ service.py       # run_revision_agent entry points
│  │  └─ state.py         # Shared conversation state
│  ├─ domain/tools/       # Tavily + mind map tooling
│  └─ infrastructure/     # LLM providers, MCP client, nano banana
├─ interface/chainlit_ui.py
├─ examples/
└─ tests/
```

## Architecture

### LangGraph Supervisor (`graph.py`)

- Builds a `StateGraph` whose entry point is the supervisor node.
- Edges return every worker back to the supervisor and end once a final response is emitted.
- The state is backed by `AgentState`, keeping the chat transcript plus the latest outputs such as `mind_map_url`.

### Decision & Worker Nodes (`node.py`)

1. **Supervisor**
   - Collects the last user query (`_extract_user_query`) and asks an OpenAI model for a structured `SupervisorDecision`.
   - Handoff options: `simple_answer`, `quick_search`, `deep_research`, `image_generation`, `mind_map`.
   - When a worker finishes and the response is staged in `chat_response`, the supervisor finalises and ends the graph.

2. **Quick Search**
   - Calls Tavily (`_tavily_search`), formats numbered snippets, and prompts the LLM with the `quick_search_summary` template to produce a short citation-aware answer.

3. **Deep Research (MCP)**
   - Expands the handoff message into a brief using `deep_research_brief`.
   - Invokes the MCP agent asynchronously and captures the rich answer or error text.

4. **Mind Map**
   - Generates outline JSON (topic + subtopics) from the LLM to keep prompts compact.
   - Calls `create_mindmap` (falls back to `simple_mindmap`) so only a short reference ID (e.g. `mm_132216_5634a1`) hits the model context.
   - The reference is immediately resolved to a base64 PNG via `display_mindmap`/`get_mindmap`, stored in `mind_map_url`, and summarised as bullet points for the user. Chainlit renders the PNG automatically.

5. **Image Generation**
   - Uses another OpenAI call to craft a single-paragraph visual prompt before trying Gemini ⇒ OpenAI image providers via the nano-banana helpers.

Shared helpers at the top of `node.py` cover user query extraction, Tavily formatters, and outline/subtopic derivation. This keeps each worker lean and focused on its external tool surface.

## Chainlit UX

The conversation UI in `interface/chainlit_ui.py`:

- Calls `run_revision_agent` with the message history on every turn.
- Displays the assistant answer, any routing notes, and renders a PNG for either `image_url` (image agent) or `mind_map_url` (mind map worker).

## Configuration

- `.env.example` lists the keys the agents expect: `OPENAI_API_KEY`, `TAVILY_API_KEY`, optional `GEMINI_API_KEY`, and MCP server settings.
- `pyproject.toml` defines the Poetry/uv metadata; `uv.lock` pins dependencies for reproducible installs.

## Tests & Quality

The suite under `tests/` exercises provider prompts, nano banana adapters, and MCP harnesses. Extend with scenario tests when you touch `node.py` or add new workers.

```
uv run pytest -q
```

Lint/format commands are available in the `Makefile`; run them before proposing changes.

## Roadmap

- Expose the LangGraph agent behind a stable FastAPI endpoint.
- Add a React dashboard that reuses the Chainlit flow.
- Expand automated coverage for the supervisor routing edges and the mind map summariser.

Contributions that keep the state lean, prompts clear, and worker outputs reference-based are welcome. EOF
