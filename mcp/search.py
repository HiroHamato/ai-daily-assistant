from __future__ import annotations

import asyncio
import re
import time
from typing import Dict, List, Tuple

from duckduckgo_search import DDGS


_CRYPTO_SITES = (
    "coindesk.com",
    "cointelegraph.com",
    "decrypt.co",
    "theblock.co",
    "coinjournal.net",
    "ambcrypto.com",
)


def _is_cyrillic(text: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", text))


_CACHE: Dict[Tuple[str, str, int], Tuple[float, List[Tuple[str, str]]]] = {}
_CACHE_TTL_SEC = 60


async def web_search(query: str, *, max_results: int = 5) -> List[Tuple[str, str]]:
    def _search() -> List[Tuple[str, str]]:
        region = "ru-ru" if _is_cyrillic(query) else "wt-wt"
        key = (query.strip(), region, max_results)
        now = time.time()
        cached = _CACHE.get(key)
        if cached and (now - cached[0] <= _CACHE_TTL_SEC):
            return cached[1]

        out: List[Tuple[str, str]] = []
        with DDGS() as ddgs:
            # Prefer fresh news (last 24h); avoid text fallback to reduce extra requests
            for r in ddgs.news(query, region=region, safesearch="moderate", timelimit="d", max_results=max_results * 2):
                title = (r.get("title") or "").strip()
                url = (r.get("url") or r.get("href") or "").strip()
                if title and url:
                    out.append((title, url))
                if len(out) >= max_results * 2:
                    break

        # Domain-biased sorting post-fetch (no extra network calls)
        def score(item: Tuple[str, str]) -> int:
            url = item[1]
            return 1 if any(domain in url for domain in _CRYPTO_SITES) else 0

        out.sort(key=score, reverse=True)
        out = out[:max_results]
        _CACHE[key] = (now, out)
        return out

    return await asyncio.to_thread(_search)
