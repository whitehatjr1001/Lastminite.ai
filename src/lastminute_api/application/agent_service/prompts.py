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
    "output_assembly": "OUTPUT_ASSEMBLY_PROMPT",
}

# Router Prompt: classify query complexity

ROUTER_PROMPT = """
You are a supervisor deciding how to handle an incoming user request.

Classify the query into exactly one of the following categories:
- simple_answer → the supervisor can answer directly without external tools.
- quick_search → requires a light-weight web lookup (Tavily search) before answering.
- deep_research → requires a deep MCP toolchain workflow.
- image_generation → user wants a diagram, illustration, or other image output.

Return only one of: simple_answer, quick_search, deep_research, image_generation.

Query: {input}
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
