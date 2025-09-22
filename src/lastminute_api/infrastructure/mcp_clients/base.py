"""Common utilities for MCP (Model Context Protocol) HTTP clients.

This module provides a small abstraction around :mod:`httpx` that allows
specialised MCP clients (e.g. ArXiv, PubMed) to share connection handling
and error semantics.  Each concrete client can focus on translating API
payloads into the domain structures required by the STEM revision system.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Optional

import httpx

logger = logging.getLogger(__name__)


class MCPClientError(RuntimeError):
    """Base exception for MCP client failures."""


class MCPClientTimeoutError(MCPClientError):
    """Raised when an MCP request exceeds the configured timeout."""


class BaseMCPClient:
    """Async context manager wrapping a shared :class:`httpx.AsyncClient`.

    Concrete subclasses should expose public async methods (e.g.
    ``search_papers``) that orchestrate requests using :attr:`client`.
    """

    def __init__(
        self,
        *,
        http_client: Optional[httpx.AsyncClient] = None,
        timeout: float = 30.0,
    ) -> None:
        self._external_client = http_client
        self._client: Optional[httpx.AsyncClient] = http_client
        self._timeout = timeout

    async def __aenter__(self) -> "BaseMCPClient":
        if self._client is None:
            self._client = self._build_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        await self.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise MCPClientError("HTTP client not initialised – use the async context manager")
        return self._client

    @property
    def timeout(self) -> float:
        return self._timeout

    async def aclose(self) -> None:
        if self._client is not None and self._client is not self._external_client:
            await self._client.aclose()
        self._client = self._external_client

    def _build_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self._timeout)

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self.client.request(method, url, timeout=self._timeout, **kwargs)
        except httpx.TimeoutException as exc:
            raise MCPClientTimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise MCPClientError(str(exc)) from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:512]
            logger.debug("MCP request failed: %s", detail)
            raise MCPClientError(f"HTTP {exc.response.status_code}: {detail}") from exc

        return response


async def gather_with_concurrency(limit: int, *tasks: Awaitable[Any]) -> list[Any]:
    """Helper for running multiple coroutines with a concurrency cap."""

    semaphore = asyncio.Semaphore(limit)

    async def _sem_task(coro: Awaitable[Any]) -> Any:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(_sem_task(task) for task in tasks))


__all__ = [
    "BaseMCPClient",
    "MCPClientError",
    "MCPClientTimeoutError",
    "gather_with_concurrency",
]
