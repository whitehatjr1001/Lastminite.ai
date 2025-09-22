"""ArXiv MCP client."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from .base import BaseMCPClient, MCPClientError

logger = logging.getLogger(__name__)


ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivMCPClient(BaseMCPClient):
    """Thin wrapper around the public ArXiv API for paper discovery."""

    def __init__(
        self,
        *,
        base_url: str = "https://export.arxiv.org/api/query",
        page_size: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._base_url = base_url
        self._page_size = max(1, min(page_size, 100))

    async def search_papers(
        self,
        query: str,
        *,
        start: int = 0,
        max_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        limit = max_results or self._page_size
        params = {
            "search_query": f"all:{query}",
            "start": max(0, start),
            "max_results": max(1, min(limit, 100)),
        }

        response = await self._request("GET", self._base_url, params=params)
        return self._parse_arxiv_response(response.text)

    def _parse_arxiv_response(self, payload: str) -> List[Dict[str, Any]]:
        try:
            root = ET.fromstring(payload)
        except ET.ParseError as exc:
            raise MCPClientError("Failed to parse ArXiv response") from exc

        items: List[Dict[str, Any]] = []
        for entry in root.findall("atom:entry", ARXIV_NS):
            title = (entry.findtext("atom:title", default="", namespaces=ARXIV_NS) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ARXIV_NS) or "").strip()
            url = (entry.findtext("atom:id", default="", namespaces=ARXIV_NS) or "").strip()
            published = (entry.findtext("atom:published", default="", namespaces=ARXIV_NS) or "").strip()
            authors = [
                (author.findtext("atom:name", default="", namespaces=ARXIV_NS) or "").strip()
                for author in entry.findall("atom:author", ARXIV_NS)
                if (author.findtext("atom:name", default="", namespaces=ARXIV_NS) or "").strip()
            ]

            items.append(
                {
                    "title": title,
                    "abstract": summary,
                    "url": url,
                    "published": published,
                    "authors": authors,
                }
            )

        return items


__all__ = ["ArxivMCPClient"]

