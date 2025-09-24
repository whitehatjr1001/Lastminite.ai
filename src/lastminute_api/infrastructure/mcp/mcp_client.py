# src/lastminute_api/infrastructure/mcp/client.py
from mcp_use import MCPClient
from lastminute_api.infrastructure.mcp.config import build_mcp_config

def get_mcp_client():
    config = build_mcp_config()
    return MCPClient.from_dict(config)
