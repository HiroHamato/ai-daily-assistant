from __future__ import annotations

import asyncio
from typing import List, Optional

from notion_client import Client

from config import get_settings


class NotionError(Exception):
    pass


_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is None:
        s = get_settings()
        if not s.NOTION_TOKEN:
            raise NotionError("NOTION_TOKEN is not set")
        _client = Client(auth=s.NOTION_TOKEN)
    return _client


async def add_note(title: str, content: Optional[str] = None) -> str:
    s = get_settings()
    if not s.NOTION_DATABASE_ID:
        raise NotionError("NOTION_DATABASE_ID is not set")

    def _create() -> str:
        client = _get_client()
        # Detect actual title property name (e.g., 'Doc name')
        db = client.databases.retrieve(database_id=s.NOTION_DATABASE_ID)
        props = db.get("properties") or {}
        title_prop = None
        for name, meta in props.items():
            if meta and meta.get("type") == "title":
                title_prop = name
                break
        if not title_prop:
            title_prop = "Name"

        children = []
        if content:
            children = [
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": content},
                            }
                        ]
                    },
                }
            ]
        page = client.pages.create(
            parent={"type": "database_id", "database_id": s.NOTION_DATABASE_ID},
            properties={
                title_prop: {
                    "title": [
                        {
                            "type": "text",
                            "text": {"content": title[:200]},
                        }
                    ]
                }
            },
            children=children,
        )
        page_id = page.get("id", "")
        return f"Notion: создана заметка '{title}' ({page_id})"

    return await asyncio.to_thread(_create)


def _extract_title_from_page(page: dict) -> Optional[str]:
    props = page.get("properties") or {}
    for prop in props.values():
        if prop and prop.get("type") == "title":
            pieces = prop.get("title") or []
            text = "".join(p.get("plain_text", "") for p in pieces)
            if text:
                return text
    # Fallbacks
    return page.get("url") or page.get("id")


def _get_db_title_prop(client: Client, database_id: str) -> str:
    db = client.databases.retrieve(database_id=database_id)
    props = db.get("properties") or {}
    for name, meta in props.items():
        if meta and meta.get("type") == "title":
            return name
    return "Name"


async def search_pages(query: str, *, max_results: int = 5) -> List[str]:
    def _search() -> List[str]:
        client = _get_client()
        # 1) Global search across shared pages
        resp = client.search(
            query=query,
            filter={"property": "object", "value": "page"},
            sort={"direction": "descending", "timestamp": "last_edited_time"},
            page_size=max_results,
        )
        results = resp.get("results", [])
        titles: List[str] = []
        for r in results:
            title = _extract_title_from_page(r)
            if title:
                titles.append(title)
        if titles:
            return titles[:max_results]

        # 2) Fallback: query specific database by its title property (if configured)
        s = get_settings()
        if s.NOTION_DATABASE_ID:
            title_prop = _get_db_title_prop(client, s.NOTION_DATABASE_ID)
            if query:
                resp = client.databases.query(
                    database_id=s.NOTION_DATABASE_ID,
                    filter={
                        "property": title_prop,
                        "title": {"contains": query},
                    },
                    page_size=max_results,
                )
            else:
                resp = client.databases.query(
                    database_id=s.NOTION_DATABASE_ID,
                    page_size=max_results,
                )
            titles_db: List[str] = []
            for r in resp.get("results", []):
                props = r.get("properties") or {}
                meta = props.get(title_prop) or {}
                if meta.get("type") == "title":
                    pieces = meta.get("title") or []
                    text = "".join(p.get("plain_text", "") for p in pieces)
                    if text:
                        titles_db.append(text)
                else:
                    t = _extract_title_from_page(r)
                    if t:
                        titles_db.append(t)
            if titles_db:
                return titles_db[:max_results]

        return []

    return await asyncio.to_thread(_search)
