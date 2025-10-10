import os
from typing import List, Tuple

from aiogram import Bot, Dispatcher, Router
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.types import Message

from config import get_settings
from llm.llama_client import LlamaClient
from memory.sqlite_store import DialogueMemory
from utils.prompts import build_system_prompt, build_user_prompt

router = Router()

# Globals to keep instances accessible in handlers
_memory: DialogueMemory | None = None
_llm: LlamaClient | None = None


@router.message(Command("health"))
async def cmd_health(message: Message) -> None:
    await message.answer("OK")


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "Я ежедневный ассистент: понимаю контекст, веду заметки, ищу в вебе, показываю погоду.\n"
        "Доступные команды: /health, /help"
    )


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

    # Load short history for context (last ~10 turns)
    history: List[Tuple[str, str]] = list(await _memory.history(chat_id, limit=20))
    memory_snippets = [f"{role}: {content}" for role, content in history[-10:]]

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(user_text, memory_snippets=memory_snippets)

    try:
        reply_text = await _llm.acomplete(user_prompt, system=system_prompt, temperature=0.2, max_tokens=512)
    except Exception as e:
        reply_text = f"Извините, произошла ошибка LLM: {e}"

    # Save assistant reply
    await _memory.add(chat_id, "assistant", reply_text)

    await message.answer(reply_text)


async def run_bot() -> None:
    s = get_settings()

    token = s.TELEGRAM_BOT_TOKEN
    if not token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in environment")

    # Init memory and LLM
    global _memory, _llm
    _memory = DialogueMemory(s.MEMORY_DB_PATH)
    await _memory.init()

    _llm = LlamaClient()

    bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    dp.include_router(router)

    await dp.start_polling(bot)
