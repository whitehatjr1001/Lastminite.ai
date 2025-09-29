# src/lastminute_api/domain/prompts/prompts.py

from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

PROMPT_MAPPER = {
    "router": "ROUTER_PROMPT",
    "simple_answer": "SIMPLE_ANSWER_PROMPT",
    "quick_search_summary": "QUICK_SEARCH_SUMMARY_PROMPT",
    "deep_research_brief": "DEEP_RESEARCH_BRIEF_PROMPT",
    "tavily_search": "TAVILY_SEARCH_PROMPT",
    "mcp_agent": "MCP_AGENT_PROMPT",
    "image_generation": "IMAGE_GENERATION_PROMPT",
    "mind_map_context": "MIND_MAP_CONTEXT_PROMPT",
    "mind_map_blueprint": "MIND_MAP_BLUEPRINT_PROMPT",
    "mind_map_reflection": "MIND_MAP_REFLECTION_PROMPT",
    "mind_map_summary": "MIND_MAP_SUMMARY_PROMPT",
    "output_assembly": "OUTPUT_ASSEMBLY_PROMPT",
}

# Router Prompt: classify query complexity

ROUTER_PROMPT = """
You are a routing expert deciding how the supervisor should respond to a user query.

Choose exactly one of the following options:
- simple_answer – The question is self-contained and can be answered from knowledge without external tools.
- quick_search – Needs a light web lookup (recent facts, specific figures, up-to-date info) but not deep reasoning.
- deep_research – Requires multi-source investigation, complex synthesis, or specialised tools.
- image_generation – The user explicitly wants an image, diagram, infographic, illustration, or visual description.
- mind_map – The user asks for a mind map, concept map, diagram of relationships, or study outline.

Guidelines:
- Prefer simple_answer when the request is conceptual, definitional, or clearly answerable without search.
- Prefer image_generation when keywords like image, diagram, picture, infographic, illustration, flowchart appear and no mind map is requested.
- Prefer mind_map whenever the user explicitly requests a mind map, concept map, or structured relationship diagram for study.
- Prefer quick_search for time-sensitive or fact lookup queries (latest, recent, stats, dates, price).
- Prefer deep_research when the user asks for comprehensive analysis, multi-step plans, or cross-domain research.

Respond with just one label: simple_answer, quick_search, deep_research, image_generation, or mind_map.

Examples:
Query: "Define osmosis in one sentence." → simple_answer
Query: "What is the latest inflation rate in the US?" → quick_search
Query: "Create a detailed market analysis of the EV sector for 2025." → deep_research
Query: "Generate an infographic explaining the water cycle." → image_generation
Query: "Create a mind map of photosynthesis stages." → mind_map

User query: {input}
Answer:
"""


SIMPLE_ANSWER_PROMPT = """
You are a helpful expert. Provide a concise and direct answer to the user's question.

Question: {input}

Answer clearly and accurately in a short paragraph.
"""


QUICK_SEARCH_SUMMARY_PROMPT = """
You are composing a short answer from quick web search snippets.

User query:
{query}

Search snippets:
{search_results}

Craft a factual, neutral answer in 2-3 sentences citing insights from the snippets. Do not fabricate details.
"""


DEEP_RESEARCH_BRIEF_PROMPT = """
You are preparing a deep research brief for an MCP agent with multiple tools.

User query:
{input}

Provide:
1. A refined research objective in one sentence.
2. Three bullet points outlining specific sub-questions or data to collect.
3. Recommended tools or data sources if relevant.

Format as:
Objective: ...
Sub-questions:
- ...
- ...
- ...
Tools:
- ... (if none, write "- none specified")
"""

# Tavily Quick Search Prompt retained for compatibility

TAVILY_SEARCH_PROMPT = """
You are a fact-retrieval assistant using Tavily.

Query: {input}

Return a brief, accurate factual answer suitable for quick revision.

Example:
Q: "When was Newton born?"
A: "Isaac Newton was born in 1642."

Only return the answer; no explanations.
"""

# MCP Agent Prompt: stepwise reasoning with tool usage

MCP_AGENT_PROMPT = """
You are an expert AI assistant answering complex queries with stepwise reasoning.

Query: {input}

Respond with reasoning steps and cite any tool you call, showing observations.

Example:
Q: "Explain the cardiac cycle."
A: "Step 1: The atria contract... [calls tool 'heart_diagram'] ... Observation: atrial contraction shown. Step 2: Ventricles contract..."

Finish with "Final Answer:" and your concise summary.
"""

# Image Generation Prompt: generate mind map description

IMAGE_GENERATION_PROMPT = """
Generate a description for a mind map or educational diagram based on:

Content: {input}

Include key concepts and relationships clearly.

Example:
"Central node: 'Cardiac Cycle'. Branch nodes: 'Atrial Contraction', 'Ventricular Contraction', 'Relaxation Phase'."
"""

# Mind Map Context Prompt: turn snippets into a knowledge chunk

MIND_MAP_CONTEXT_PROMPT = """
You are compiling revision notes to drive a study mind map.

Topic: {topic}
Supervisor notes: {notes}

Search snippets:
{snippets}

Write a focused knowledge brief (<=220 words) covering:
1. A one-sentence overview of the topic.
2. Bullet paragraphs for core categories/types.
3. Bullet paragraphs for key operations/mechanics.
4. Bullet paragraphs for applications or benefits.

Use only information available in the snippets (or universal fundamentals if snippets lack detail). Keep the bullet
paragraphs short (<=40 words each) and factual.
"""

# Mind Map Blueprint Prompt: convert context into nodes and edges

MIND_MAP_BLUEPRINT_PROMPT = """
You create concise study mind maps.

Topic: {topic}
Knowledge brief:
{context}
Supervisor notes: {notes}
Additional feedback: {feedback}

Return JSON exactly matching this schema:
{{
  "central_topic": "...",
  "nodes": ["...", "..."],
  "edge_map": [["source", "target"], ...],
  "bullet_points": ["...", "..."]
}}

Rules:
1. central_topic must reuse words from the topic/brief and appear in nodes.
2. Provide 6-10 nodes. Cover overview/definition, major categories or types, operations/mechanics, and use cases/benefits.
3. edge_map must contain at least six pairs. Every pair must reference nodes you listed. Connect the central topic to each
   major branch, then link branch nodes to their supporting details.
4. bullet_points must include exactly three concise study reminders grounded in the brief (≤18 words each).
5. Never fabricate concepts not supported by the brief.
"""

# Mind Map Reflection Prompt: validate grounding

MIND_MAP_REFLECTION_PROMPT = """
Topic: {topic}
Knowledge brief:
{context}
Proposed nodes: {nodes}
Proposed edges:
{edges}

Reply strictly with one of:
- "YES: <reason>" if the nodes/edges are well grounded in the knowledge brief and topic.
- "NO: <what is missing or incorrect>" otherwise (mention which required branch is weak or what to fix).
"""

# Mind Map Summary Prompt: narrate generated mind map

MIND_MAP_SUMMARY_PROMPT = """
Mind map topic: {topic}
Knowledge brief:
{context}

Nodes: {nodes}
Key takeaways:
{bullet_points}

Explain the mind map in <=150 words. Start with the central idea, then describe the major branches and their
relationships. Highlight how the map helps a student revise this topic.
"""

# Output Assembly Prompt: merge text and image output

OUTPUT_ASSEMBLY_PROMPT = """
Assemble a user-friendly revision output combining text and image.

Answer:
{text_answer}

Mind Map:
{image_url}
"""

def get_prompt(prompt_name: str) -> PromptTemplate:
    prompt = PROMPT_MAPPER.get(prompt_name)
    if not prompt:
        raise ValueError(f"Unknown prompt '{prompt_name}'")
    return PromptTemplate.from_template(prompt)

def apply_prompt(prompt_name: str, **kwargs) -> str:
    # Flatten messages if present
    if "messages" in kwargs and isinstance(kwargs["messages"], list):
        kwargs["messages"] = "\n".join([msg.content for msg in kwargs["messages"]])
    kwargs.setdefault("input", HumanMessage(content="Hello"))
    kwargs.setdefault("text_answer", "")
    kwargs.setdefault("image_url", "")
    return get_prompt(prompt_name).format(**kwargs)
