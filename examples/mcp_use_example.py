import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

async def create_research_agent():
    """Create an agent with both PubMed and ArXiv capabilities"""
    load_dotenv()
    
    # Configuration for both PubMed and ArXiv MCP servers
    config = {
        "mcpServers": {
            # PubMed server (using grll/pubmedmcp)
            "pubmed": {
                "command": "uvx",
                "args": ["pubmedmcp@latest"],
                "env": {
                    "UV_PRERELEASE": "allow",
                    "UV_PYTHON": "3.12"
                }
            },
            # ArXiv server (using blazickjp/arxiv-mcp-server)
            "arxiv": {
                "command": "uv",
                "args": [
                    "tool",
                    "run",
                    "arxiv-mcp-server",
                    "--storage-path", "./papers"
                ]
            }
        }
    }
    
    # Create client and agent
    client = MCPClient.from_dict(config)
    llm = ChatOpenAI(model="gpt-4o")
    agent = MCPAgent(
        llm=llm, 
        client=client, 
        max_steps=30,
        use_server_manager=True  # Intelligently choose servers
    )
    
    return agent, client

async def main():
    agent, client = await create_research_agent()
    
    try:
        # Example research queries
        queries = [
            "Search PubMed for recent studies on dental implant success rates from 2024",
            "Find ArXiv papers about machine learning in medical diagnosis from the last year",
            "Compare recent research on AI applications in dentistry from both PubMed and ArXiv"
        ]
        
        for query in queries:
            print(f"\n🔍 Query: {query}")
            print("=" * 50)
            
            result = await agent.run(query)
            print(f"Result: {result}")
            print("\n")
            
    finally:
        # Clean up
        await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(main())
