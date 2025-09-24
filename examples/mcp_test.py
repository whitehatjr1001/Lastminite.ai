# import asyncio
# import os
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from mcp_use import MCPAgent, MCPClient

# async def main():
#     # Load environment variables
#     load_dotenv()

#     # Create configuration dictionary
#     config ={
#   "mcpServers": {
#     "biomcp": {
#       "command": "uv",
#       "args": ["run", "--with", "biomcp-python", "biomcp", "run"]
#     }
#   }
# }

#     # Create MCPClient from configuration dictionary
#     client = MCPClient.from_dict(config)

#     # Create LLM
#     llm = ChatOpenAI(model="gpt-4o")

#     # Create agent with the client
#     agent = MCPAgent(
#         llm=llm,
#         client=client,
#         max_steps=30,
#         use_server_manager=False,
#     )

#     # Run the query
#     result = await agent.run(
#         "Find recent clinical trials involving dental implants.",
#     )
#     print(f"\nResult: {result}")

# if __name__ == "__main__":
#     asyncio.run(main())

import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient
import os
load_dotenv()

api_key = os.getenv("TALVIY_API_KEY")
async def main():
    # Replace <your-api-key> with your actual Tavily API key
    server_url = f"https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}"
    config = {
        "mcpServers": {
            "tavily": {
                "url": server_url
            }
        }
    }

    client = MCPClient.from_dict(config)
    llm = ChatOpenAI(model="gpt-4o")
    agent = MCPAgent(llm=llm, client=client, max_steps=30)
    result = await agent.run("Perform a Tavily search for top AI news today", max_steps=30)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
