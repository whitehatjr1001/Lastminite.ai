<div id="top"></div>

# Lastminute.ai

LangGraph supervisor orchestrating multiple specialist agents—quick search, deep MCP research, mind-map synthesis, and image generation—to answer revision-style prompts. Today it ships with a Chainlit playground; a FastAPI + React surface is on the roadmap.

---
## 🎥 Lastminute AI Demo

<div align="center">
  <video width="800" height="450" controls>
    <source src="assets/lastminute-ai-demo.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

*Watch our multi-agent system in action: deep research, quick search, image generation, and mindmap creation with Chainlit UI*

### Feature Demos

#### 🔍 Research Agent
<video width="600" height="400" controls>
  <source src="assets/research-demo.mp4" type="video/mp4">
</video>

#### 🎨 Image Generation  
<video width="600" height="400" controls>
  <source src="assets/image-demo.mp4" type="video/mp4">
</video>


## Tech Stack

<div align="center">

| Layer           | Tools |
|-----------------|-------|
| **Runtime**     | Python 3.11, uv, Docker |
| **LLM & Agents**| LangChain Core, LangGraph, OpenAI, Tavily, MCP (multi-call), Nano Banana image providers |
| **Interface**   | Chainlit (current), FastAPI + React (planned) |
| **Data & Utils**| Pydantic, Requests |
| **Testing**     | Pytest |

</div>

---

## Quick Start

```bash
uv venv --python 3.11
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv sync

cp .env.example .env               # add OPENAI_API_KEY, TAVILY_API_KEY, optional GEMINI

uv run chainlit run interface/chainlit_ui.py
```

Example scripts (logging-heavy):

```bash
uv run python examples/supervisor_service_demo.py
uv run python examples/test.py         # reference-based mind map flow
```

Run tests:

```bash
uv run pytest -q
```

---

## Folder Layout

```
agent-api/
├─ src/lastminute_api/
│  ├─ application/agent_service/
│  │  ├─ graph.py      # Build the LangGraph state machine
│  │  ├─ node.py       # Supervisor + worker nodes
│  │  ├─ service.py    # run_revision_agent sync/async helpers
│  │  └─ state.py      # Shared AgentState definition
│  ├─ domain/tools/    # Tavily client, mind map registry
│  └─ infrastructure/  # LLM providers (OpenAI/Groq), MCP client, nano banana
├─ interface/chainlit_ui.py
├─ examples/
└─ tests/
```

---

## Architecture Snapshot

### LangGraph Supervisor (`graph.py`)

- Entry node: `supervisor` decides the next action via a structured OpenAI router prompt (`SupervisorDecision`).
- Worker nodes (`tavily_agent`, `mcp_agent`, `mind_map_agent`, `image_agent`) update shared state then hand control back to the supervisor.
- Final answers are staged in `chat_response`; once present, the supervisor appends to messages and ends the graph.

### Worker Highlights (`node.py`)

- **Quick Search**: Tavily API + snippet formatter → OpenAI summariser with inline citation markers.
- **Deep Research**: Generates a brief using `deep_research_brief` prompt, invokes MCP agent, records the rich answer.
- **Mind Map**: LLM creates a JSON outline; `create_mindmap` returns a short reference. We immediately resolve the PNG via `display_mindmap/get_mindmap`, summarise as bullet points, and expose both text + image to Chainlit.
- **Image Generation**: Crafts a focused prompt and tries Gemini ⇒ OpenAI image APIs via nano banana helpers.

### Chainlit Frontend (`interface/chainlit_ui.py`)

- Retains message history, calls `run_revision_agent`, renders the assistant reply.
- Displays the PNG from `image_url` (image agent) or `mind_map_url` (mind map worker) automatically.
- Shows routing notes when the supervisor provides rationale.

---

## Configuration

- `.env` expects at least `OPENAI_API_KEY` and `TAVILY_API_KEY`. Optional: `GEMINI_API_KEY`, MCP server host & token.
- `pyproject.toml` + `uv.lock` manage dependencies; `Makefile` contains common lint/format/test tasks.

---

## Roadmap

- Wrap the LangGraph supervisor in a FastAPI HTTP endpoint with streaming responses.
- Build a React dashboard reusing the Chainlit flow.
- Add tests for supervisor routing edges and mind-map outline fallbacks.
- Expand logging/metrics for production observability.

---

## Contributing

1. Fork & clone the repo.
2. Create a feature branch (`git checkout -b feat/my-change`).
3. Run formatting (`make format`) and tests (`uv run pytest`).
4. Open a PR describing the behaviour change and test coverage.

---

## Credits

Thanks to [MichaelisTrofficus](https://github.com/MichaelisTrofficus) for the excellent agent cookie-cutter that inspired this project’s structure.

---

Made with ❤️ by the Lastminute.ai team.

<div align="center">
  <a href="#top">⬆️ Back to top</a>
</div>
