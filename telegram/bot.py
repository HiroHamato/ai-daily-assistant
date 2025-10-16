import asyncio
import re
from typing import List, Tuple

from aiogram import Bot, Dispatcher, Router
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.types import LinkPreviewOptions

from config import get_settings
from llm.llama_client import LlamaClient
from memory.sqlite_store import DialogueMemory
from utils.prompts import build_system_prompt, build_user_prompt
from vector_store.faiss_store import FaissStore
from mcp.weather import get_weather, WeatherError
from mcp.notion import add_note as notion_add_note, search_pages as notion_search_pages, NotionError
from mcp.search import web_search as web_search_ddg

router = Router()

# Globals to keep instances accessible in handlers
_memory: DialogueMemory | None = None
_llm: LlamaClient | None = None
_kb: FaissStore | None = None
_kb_ready: asyncio.Event | None = None

# Reusable option to disable link previews
_LP_OFF = LinkPreviewOptions(is_disabled=True)


@router.message(Command("health"))
async def cmd_health(message: Message) -> None:
    status = "ready" if (_kb_ready and _kb_ready.is_set()) else "initializing"
    await message.answer(f"OK (kb: {status})", link_preview_options=_LP_OFF)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "Я ежедневный ассистент: понимаю контекст, веду заметки, ищу в вебе, показываю погоду.\n"
        "Доступные команды: /health, /help, /kb_add <текст>, /kb_search <запрос>",
        link_preview_options=_LP_OFF,
    )


@router.message(Command("kb_add"))
async def cmd_kb_add(message: Message) -> None:
    assert _kb is not None
    if not (_kb_ready and _kb_ready.is_set()):
        await message.answer("База знаний инициализируется, попробуйте чуть позже.", link_preview_options=_LP_OFF)
        return
    text = (message.text or "").partition(" ")[2].strip()
    if not text:
        await message.answer("Использование: /kb_add <текст для добавления в базу>", link_preview_options=_LP_OFF)
        return
    await _kb.add_texts([text])
    await message.answer("Добавлено в базу знаний.", link_preview_options=_LP_OFF)


@router.message(Command("kb_search"))
async def cmd_kb_search(message: Message) -> None:
    assert _kb is not None
    if not (_kb_ready and _kb_ready.is_set()):
        await message.answer("База знаний инициализируется, попробуйте чуть позже.", link_preview_options=_LP_OFF)
        return
    query = (message.text or "").partition(" ")[2].strip()
    if not query:
        await message.answer("Использование: /kb_search <запрос>", link_preview_options=_LP_OFF)
        return
    results = await _kb.search(query, k=5)
    if not results:
        await message.answer("Ничего не найдено.", link_preview_options=_LP_OFF)
        return
    lines = ["Топ результаты:"]
    for i, (text, score) in enumerate(results, 1):
        lines.append(f"{i}. {text}")
    await message.answer("\n".join(lines), link_preview_options=_LP_OFF)


TOOL_WEATHER_RE = re.compile(r"\[\[\s*WEATHER\s*:\s*([^\]]+?)\s*\]\]", re.IGNORECASE)
TOOL_KB_SEARCH_RE = re.compile(r"\[\[\s*KB_SEARCH\s*:\s*([^\]]+?)\s*\]\]", re.IGNORECASE)
TOOL_KB_ADD_RE = re.compile(r"\[\[\s*KB_ADD\s*:\s*([^\]]+?)\s*\]\]", re.IGNORECASE)
TOOL_NOTION_SEARCH_RE = re.compile(r"\[\[\s*NOTION_SEARCH\s*:\s*(.*?)\s*\]\]", re.IGNORECASE)
TOOL_NOTION_ADD_RE = re.compile(r"\[\[\s*NOTION_ADD\s*:\s*(.*?)\s*\]\]", re.IGNORECASE)
TOOL_WEB_SEARCH_RE = re.compile(r"\[\[\s*WEB_SEARCH\s*:\s*([^\]]+?)\s*\]\]", re.IGNORECASE)


@router.message()
async def handle_message(message: Message) -> None:
    if not message.text:
        return
    if message.text.startswith("/"):
        return  # let command handlers process

    assert _memory is not None
    assert _llm is not None

    chat_id = str(message.chat.id)
    user_text = message.text

    # Save user message
    await _memory.add(chat_id, "user", user_text)

    # Provisional status message
    status_text = "Обрабатываю запрос…"
    status_msg = await message.answer(status_text, link_preview_options=_LP_OFF)

    # Load short history for context (last ~10 turns)
    history: List[Tuple[str, str]] = list(await _memory.history(chat_id, limit=20))
    memory_snippets = [f"{role}: {content}" for role, content in history[-10:]]

    # Accumulated tool outputs
    tool_snippets: List[str] = []

    system_prompt = build_system_prompt()

    # Limit to 2 total LLM calls: first + optional after one tool
    for turn in range(2):
        final_hint = []
        if tool_snippets:
            final_hint.append("Сформируй окончательный ответ, не запрашивай инструменты повторно.")
        composed_user = "\n".join([user_text] + final_hint)
        user_prompt = build_user_prompt(composed_user, memory_snippets=memory_snippets, rag_snippets=tool_snippets)
        try:
            reply = await _llm.acomplete(user_prompt, system=system_prompt, temperature=0.2, max_tokens=384)
        except Exception as e:
            err_text = f"Извините, произошла ошибка LLM: {e}"
            await _memory.add(chat_id, "assistant", err_text)
            await status_msg.edit_text(err_text, link_preview_options=_LP_OFF)
            return

        m_weather = TOOL_WEATHER_RE.search(reply)
        m_kb_search = TOOL_KB_SEARCH_RE.search(reply)
        m_kb_add = TOOL_KB_ADD_RE.search(reply)
        m_notion_search = TOOL_NOTION_SEARCH_RE.search(reply)
        m_notion_add = TOOL_NOTION_ADD_RE.search(reply)
        m_web_search = TOOL_WEB_SEARCH_RE.search(reply)

        if not (m_weather or m_kb_search or m_kb_add or m_notion_search or m_notion_add or m_web_search):
            final_text = reply
            await _memory.add(chat_id, "assistant", final_text)
            await status_msg.edit_text(final_text, link_preview_options=_LP_OFF)
            return

        if turn == 1:
            # We already did one tool; avoid infinite loops
            final_text = "Не удалось сформировать ответ. Попробуйте уточнить запрос."
            await _memory.add(chat_id, "assistant", final_text)
            await status_msg.edit_text(final_text, link_preview_options=_LP_OFF)
            return

        # Execute exactly one tool then loop once more for final answer
        if m_weather:
            city = m_weather.group(1).strip()
            try:
                status_text += "\nполучение погоды…"
                await status_msg.edit_text(status_text, link_preview_options=_LP_OFF)
                weather_text = await get_weather(city)
                tool_snippets.append(f"Погодные данные: {weather_text}")
            except Exception as e:
                tool_snippets.append(f"Погодные данные недоступны ({city}): {e}")
        elif m_kb_search:
            status_text += "\nпоиск в базе знаний…"
            await status_msg.edit_text(status_text, link_preview_options=_LP_OFF)
            if not (_kb and _kb_ready and _kb_ready.is_set()):
                tool_snippets.append("KB: индекс ещё инициализируется")
            else:
                query = m_kb_search.group(1).strip() or user_text
                results = await _kb.search(query, k=5)
                if results:
                    for t, s in results[:5]:
                        tool_snippets.append(f"KB найдено: {t}")
                else:
                    tool_snippets.append("KB: ничего не найдено")
        elif m_kb_add:
            status_text += "\nдобавление в базу знаний…"
            await status_msg.edit_text(status_text, link_preview_options=_LP_OFF)
            if not (_kb and _kb_ready and _kb_ready.is_set()):
                tool_snippets.append("KB: индекс ещё инициализируется, добавление отложено")
            else:
                fact = m_kb_add.group(1).strip() or user_text
                await _kb.add_texts([fact])
                tool_snippets.append(f"KB: факт добавлен: {fact}")
        elif m_notion_search:
            status_text += "\nпоиск в Notion…"
            await status_msg.edit_text(status_text, link_preview_options=_LP_OFF)
            try:
                q = (m_notion_search.group(1) or "").strip() or user_text
                titles = await notion_search_pages(q, max_results=5)
                if titles:
                    for t in titles:
                        tool_snippets.append(f"Notion найдено: {t}")
                else:
                    tool_snippets.append("Notion: ничего не найдено")
            except Exception as e:
                tool_snippets.append(f"Notion: поиск не удался: {e}")
        elif m_notion_add:
            status_text += "\nсоздание заметки в Notion…"
            await status_msg.edit_text(status_text, link_preview_options=_LP_OFF)
            try:
                raw = (m_notion_add.group(1) or "").strip()
                parts = [p.strip() for p in raw.split("|", 1)] if raw else []
                title = parts[0] if parts else (user_text[:200] or "Заметка")
                content = parts[1] if len(parts) > 1 else None
                info = await notion_add_note(title, content)
                tool_snippets.append(info)
            except Exception as e:
                tool_snippets.append(f"Notion: создание не удалось: {e}")
        elif m_web_search:
            status_text += "\nвеб-поиск…"
            await status_msg.edit_text(status_text, link_preview_options=_LP_OFF)
            q = m_web_search.group(1).strip() or user_text
            try:
                results = await web_search_ddg(q, max_results=5)
                if results:
                    for title, url in results:
                        tool_snippets.append(f"Web: {title} — {url}")
                else:
                    tool_snippets.append("Web: ничего не найдено")
            except Exception as e:
                tool_snippets.append(f"Web: ошибка поиска: {e}")
        # Next loop iteration will be the final answer

    fallback = "Не удалось сформировать ответ. Попробуйте переформулировать запрос."
    await _memory.add(chat_id, "assistant", fallback)
    await status_msg.edit_text(fallback, link_preview_options=_LP_OFF)


async def run_bot() -> None:
    s = get_settings()

    token = s.TELEGRAM_BOT_TOKEN
    if not token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in environment")

    # Init memory and LLM
    global _memory, _llm, _kb, _kb_ready
    _memory = DialogueMemory(s.MEMORY_DB_PATH)
    await _memory.init()

    _kb = FaissStore(s.FAISS_INDEX_PATH, s.FAISS_META_PATH, s.EMBEDDING_MODEL)
    _kb_ready = asyncio.Event()

    async def _init_kb_bg() -> None:
        try:
            await _kb.init()
        finally:
            _kb_ready.set()

    asyncio.create_task(_init_kb_bg())

    _llm = LlamaClient()

    bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    dp.include_router(router)

    await dp.start_polling(bot)
