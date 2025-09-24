# src/lastminute_api/config.py
import os
from dotenv import load_dotenv
load_dotenv()

def build_mcp_config():

    api_key = os.getenv("TALVIY_API_KEY")

    config = {
        "mcpServers": {
            "biomcp": {
                "command": "uv",
                "args": ["run", "--with", "biomcp-python", "biomcp", "run"],
            },
            "tavily": {
                "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}"
            },
            # Add new servers here as needed
        }
    }
    return config
