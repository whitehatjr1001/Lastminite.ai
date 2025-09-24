from lastminute_api.infrastructure.llm_providers.base import get_llm_by_type
from mcp_use import MCPAgent
from lastminute_api.infrastructure.mcp.mcp_client import get_mcp_client

def create_agent():
    llm = get_llm_by_type("openai")
    client = get_mcp_client()
    agent = MCPAgent(
        llm=llm,
        client=client,
        use_server_manager=True,
        max_steps=30,
    )
    return agent
