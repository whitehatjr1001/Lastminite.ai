# src/lastminute_api/domain/prompts/prompts.py

from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

PROMPT_MAPPER = {
    "router": "ROUTER_PROMPT",
    "tavily_search": "TAVILY_SEARCH_PROMPT",
    "mcp_agent": "MCP_AGENT_PROMPT",
    "image_generation": "IMAGE_GENERATION_PROMPT",
    "output_assembly": "OUTPUT_ASSEMBLY_PROMPT",
}

# Router Prompt: classify query complexity

ROUTER_PROMPT = """
Classify the user's query as 'quick_search' or 'complex_search'.

Query: {input}

Examples:
Query: "What is photosynthesis?"
Answer: quick_search

Query: "Explain the mechanism of synaptic transmission with steps."
Answer: complex_search

Return only the classification: quick_search or complex_search.
"""

# Tavily Quick Search Prompt: concise factual answer

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
