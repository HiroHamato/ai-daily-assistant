from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional

import asyncio
import time
import httpx
from config import get_settings


class LlamaClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        *,
        timeout_seconds: float = 60.0,
        min_interval_seconds: float = 1.0,
        max_retries: int = 2,
    ) -> None:
        s = get_settings()
        self.api_key = (api_key or s.SAMBANOVA_API_KEY).strip()
        self.base_url = (base_url or s.SAMBANOVA_BASE_URL).rstrip("/")
        self.model = (model or s.LLM_MODEL).strip()
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_seconds)
        # Simple rate limiter across calls for this instance
        self._rate_lock = asyncio.Lock()
        self._last_call_ts = 0.0
        self._min_interval = float(min_interval_seconds)
        self._max_retries = int(max_retries)

    async def _throttle(self) -> None:
        async with self._rate_lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_call_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call_ts = time.monotonic()

    async def _post_chat(self, json_payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        await self._throttle()
        attempt = 0
        while True:
            resp = await self._client.post("/chat/completions", headers=headers, json=json_payload)
            if resp.status_code == 429 and attempt < self._max_retries:
                retry_after = resp.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 1.5 * (attempt + 1)
                except Exception:
                    delay = 1.5 * (attempt + 1)
                await asyncio.sleep(delay)
                attempt += 1
                continue
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                detail = None
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text
                raise RuntimeError(f"LLM request failed: {e} | detail={detail}") from e
            return resp.json()

    async def acomplete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,  # e.g. {"type": "json_object"}
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> str:
        payload_messages: List[Dict[str, str]] = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        if messages:
            payload_messages.extend(messages)
        else:
            payload_messages.append({"role": "user", "content": prompt})

        headers = {"Authorization": f"Bearer {self.api_key}"}
        if extra_headers:
            headers.update(extra_headers)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if response_format is not None:
            payload["response_format"] = response_format

        data = await self._post_chat(payload, headers)
        return data["choices"][0]["message"]["content"]

    async def astream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> AsyncGenerator[str, None]:
        # For simplicity and to avoid complex rate handling with streams,
        # we call non-streaming under the hood and yield once.
        content = await self.acomplete(
            prompt,
            system=system,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            response_format=response_format,
            extra_headers=extra_headers,
        )
        yield content

    async def aclose(self) -> None:
        await self._client.aclose()
