from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional

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
    ) -> None:
        s = get_settings()
        self.api_key = (api_key or s.SAMBANOVA_API_KEY).strip()
        self.base_url = (base_url or s.SAMBANOVA_BASE_URL).rstrip("/")
        self.model = (model or s.LLM_MODEL).strip()
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_seconds)

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

        resp = await self._client.post("/chat/completions", headers=headers, json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Try to include API error body for easier debugging
            detail = None
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(f"LLM request failed: {e} | detail={detail}") from e

        data = resp.json()
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
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if response_format is not None:
            payload["response_format"] = response_format

        async with self._client.stream(
            "POST", 
            "/chat/completions", 
            headers=headers, 
            json=payload,
        ) as resp:
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                detail = None
                try:
                    detail = await resp.aread()
                except Exception:
                    pass
                raise RuntimeError(f"LLM stream request failed: {e} | detail={detail}") from e

            async for line in resp.aiter_lines():
                if not line:
                    continue
                # OpenAI-style event stream is prefixed with 'data: '
                if not line.startswith("data: "):
                    continue
                chunk = line[len("data: ") :].strip()
                if chunk == "[DONE]":
                    break
                # Parse JSON and extract delta content
                try:
                    data = httpx.Response(200, text=chunk).json()
                except Exception:
                    # Fallback: try direct json
                    try:
                        import json as _json

                        data = _json.loads(chunk)
                    except Exception:
                        continue
                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text_part = delta.get("content")
                if text_part:
                    yield text_part

    async def aclose(self) -> None:
        await self._client.aclose()
