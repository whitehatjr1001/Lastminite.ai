"""Example showing in-process MCP tools for ArXiv and PubMed searches.

This script demonstrates how to expose the new ``ArxivMCPClient`` and
``PubMedMCPClient`` classes as Model Context Protocol tools that can be
consumed by the ``mcp_use`` SDK.  Instead of spawning external MCP
servers, we register lightweight in-process connectors that translate
tool calls directly into HTTP requests.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp.types import CallToolResult, ServerCapabilities, TextContent, Tool
from mcp_use import MCPAgent
from mcp_use.connectors.base import BaseConnector

from src.lastminute_api.infrastructure.mcp_clients import ArxivMCPClient, PubMedMCPClient


class InProcessSearchConnector(BaseConnector):
    """Expose a ``BaseMCPClient`` derivative as an MCP tool."""

    def __init__(
        self,
        *,
        name: str,
        title: str,
        description: str,
        client_factory,
    ) -> None:
        super().__init__()
        self._client_factory = client_factory
        self._name = name
        self._tool = Tool(
            name=f"{name}_search",
            title=title,
            description=description,
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search terms sent to the provider.",
                        "minLength": 3,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of items to return (1-25).",
                        "minimum": 1,
                        "maximum": 25,
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        )
        self.capabilities = ServerCapabilities(tools=True)

    @property
    def public_identifier(self) -> str:
        return self._name

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def list_tools(self) -> list[Tool]:
        return [self._tool]

    async def call_tool(  # type: ignore[override]
        self,
        name: str,
        arguments: Dict[str, Any],
        read_timeout_seconds: Optional[Any] = None,
    ) -> CallToolResult:
        if name != self._tool.name:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                isError=True,
            )

        query = str(arguments.get("query", "")).strip()
        limit = int(arguments.get("limit", 5) or 5)
        limit = max(1, min(limit, 25))

        if len(query) < 3:
            return CallToolResult(
                content=[TextContent(type="text", text="Query must be at least 3 characters long.")],
                isError=True,
            )

        try:
            async with self._client_factory() as client:
                results = await client.search_papers(query=query, max_results=limit)
        except Exception as exc:  # pylint: disable=broad-except
            message = f"{self._name} search failed: {exc}"
            return CallToolResult(
                content=[TextContent(type="text", text=message)],
                isError=True,
            )

        payload = json.dumps({"results": results}, indent=2)
        return CallToolResult(
            content=[TextContent(type="text", text=payload)],
            structuredContent={"results": results},
            isError=False,
        )


def build_connectors() -> list[InProcessSearchConnector]:
    """Create connectors for both ArXiv and PubMed."""

    arxiv_connector = InProcessSearchConnector(
        name="arxiv",
        title="ArXiv Search",
        description="Find papers on arXiv using keyword queries.",
        client_factory=lambda: ArxivMCPClient(),
    )

    pubmed_connector = InProcessSearchConnector(
        name="pubmed",
        title="PubMed Search",
        description="Discover medical literature via PubMed.",
        client_factory=lambda: PubMedMCPClient(api_key=os.getenv("NCBI_API_KEY")),
    )

    return [arxiv_connector, pubmed_connector]


async def create_agent() -> MCPAgent:
    """Instantiate an ``MCPAgent`` wired to the in-process connectors."""

    load_dotenv()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    connectors = build_connectors()

    agent = MCPAgent(
        llm=llm,
        connectors=connectors,
        max_steps=6,
        auto_initialize=True,
    )

    for connector in connectors:
        await connector.connect()

    return agent


async def main() -> None:
    """Run a short interactive demo across both providers."""

    agent = await create_agent()

    queries = [
        "Summarize the latest work on quantum error correction (ArXiv).",
        "Find recent clinical trials involving dental implants (PubMed).",
        "Combine ArXiv and PubMed findings on AI-assisted radiology.",
    ]

    try:
        for query in queries:
            print(f"\n🔍 Query: {query}\n{'=' * 60}")
            response = await agent.run(query)
            print(response)
    finally:
        for connector in agent.connectors:
            await connector.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

