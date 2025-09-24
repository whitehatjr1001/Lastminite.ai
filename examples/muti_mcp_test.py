import asyncio
from lastminute_api.infrastructure.mcp.mcp_agent import create_agent

async def main():
    agent = create_agent()
    result = await agent.run(
        "use tavily mcp to find recent clinical trials involving dental implants.",
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
