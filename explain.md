# LastMinute Agent Platform Overview

_Last updated: 2025-09-25_

## Vision & Scope
LastMinute is an orchestration layer for intelligent revision and research workflows. It coordinates multiple specialised agents—quick web search, deep tool-based research, and image generation—under a single supervisor that interprets the user’s intent, routes tasks, and assembles the final response. The project aims to provide real-time, tool-augmented answers suitable for education, technical exploration, and visualization requests.

## Architectural Highlights
- **Supervisor Graph (LangGraph):** A structured LangGraph state machine with the supervisor deciding how to handle each query. Decisions are emitted as `SupervisorDecision` objects ensuring deterministic routing.
- **Worker Agents:**
  - *Quick Search Agent* integrates Tavily for fact lookups and distills results via the LLM.
  - *Deep Research Agent* delegates to an MCP agent capable of multi-tool analysis.
  - *Image Agent* leverages the Nano Banana OpenAI client to produce diagrams or illustrations.
- **Shared State (`AgentState`):** Tracks conversation history, routing notes, final answers, and image payloads. Workers update state and hand control back to the supervisor.
- **Service Layer:** `run_revision_agent` wraps the compiled graph, normalises state, and exposes sync/async APIs for consumers.
- **UI/Interface:** A Chainlit front-end (`interface/chainlit_ui.py`) offers conversational interaction, displays rich logs, and renders generated images inline.

## Evolution Timeline
1. **Initial Integration:** Set up MCP and LLM providers with basic agents and skeletal prompts.
2. **Supervisor Overhaul:** Replaced heuristic routing with structured LLM decisions, introduced detailed prompts, and unified the handoff mechanism.
3. **Logging & Observability:** Implemented emoji-coloured logging for clarity, recording each delegation and worker input.
4. **UI Enhancements:** Built Chainlit UI, implemented image rendering support, and ensured session persistence across turns.
5. **Resilience & Safety:** Added Tavily fallbacks, MCP error handling, and graceful degradation for image safety rejections.

## Development Priorities
- **Testing:** Extend automated tests covering supervisor routing and worker failure paths.
- **Credentials & Deployment:** Securely inject API keys (OpenAI, Tavily, MCP) for production-ready usage.
- **User Experience:** Explore streaming responses, richer metadata (citations, routing rationale), and more interactive UI elements.
- **Tooling:** Consider additional MCP servers or retrieval tools to widen research capabilities.

## Getting Started
1. Install dependencies via `uv sync` (ensures `requests` and other runtime deps are present).
2. Supply environment variables (`OPENAI_API_KEY`, `TAVILY_API_KEY`, MCP configs).
3. Run demos:
   - CLI: `uv run python -m examples.supervisor_service_demo`
   - UI: `chainlit run interface/chainlit_ui.py`
4. Inspect logs in the terminal for routing diagnostics and agent handoffs.

## Conclusion
The LastMinute platform is now a structured, multi-agent system capable of deciding when to search, research deeply, or produce imagery—all while keeping users informed through enhanced logging and UI feedback. Continued investment in testing, security, and UX will push it toward production readiness.
