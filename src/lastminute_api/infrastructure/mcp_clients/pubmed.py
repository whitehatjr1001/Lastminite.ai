"""PubMed MCP client."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base import BaseMCPClient, MCPClientError, gather_with_concurrency

logger = logging.getLogger(__name__)


class PubMedMCPClient(BaseMCPClient):
    """Wrapper around NCBI's E-utilities for PubMed searches."""

    def __init__(
        self,
        *,
        base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    async def search_papers(
        self,
        query: str,
        *,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        search_params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max(1, min(max_results, 200)),
        }
        if self._api_key:
            search_params["api_key"] = self._api_key

        search_url = f"{self._base_url}/esearch.fcgi"
        response = await self._request("GET", search_url, params=search_params)

        data = response.json()
        id_list = data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return []

        summaries = await self._fetch_summaries(id_list)
        return summaries

    async def _fetch_summaries(self, id_list: List[str]) -> List[Dict[str, Any]]:
        chunks = [id_list[i : i + 50] for i in range(0, len(id_list), 50)]

        tasks = [self._summary_chunk(chunk) for chunk in chunks]
        results = await gather_with_concurrency(3, *tasks)

        merged: List[Dict[str, Any]] = []
        for batch in results:
            merged.extend(batch)
        return merged

    async def _summary_chunk(self, chunk: List[str]) -> List[Dict[str, Any]]:
        summary_params = {
            "db": "pubmed",
            "retmode": "json",
            "id": ",".join(chunk),
        }
        if self._api_key:
            summary_params["api_key"] = self._api_key

        summary_url = f"{self._base_url}/esummary.fcgi"
        response = await self._request("GET", summary_url, params=summary_params)

        payload = response.json()
        result = payload.get("result", {})
        summaries: List[Dict[str, Any]] = []
        for uid in result.get("uids", []):
            item = result.get(uid, {})
            summaries.append(
                {
                    "id": uid,
                    "title": item.get("title", "").strip(),
                    "authors": [auth.get("name", "").strip() for auth in item.get("authors", []) if auth.get("name")],
                    "source": item.get("source", ""),
                    "publication_date": item.get("pubdate", ""),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                    "summary": str(item.get("elocationid", "")).strip(),
                }
            )

        return summaries


__all__ = ["PubMedMCPClient"]
