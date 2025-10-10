from __future__ import annotations

from typing import Iterable, Optional


SYSTEM_PROMPT = (
    "Ты — ежедневный ассистент (Daily Assistant). Твоя задача — помогать пользователю в повседневных делах: планирование, заметки, напоминания, ответы на вопросы, быстрый ресёрч и погода. "
    "Всегда отвечай кратко, структурированно и по делу. Учитывай контекст диалога. Если чего-то не знаешь — уточни вопрос.\n\n"
    "Доступные инструменты (MCP):\n"
    "- Weather: получай текущую погоду и прогноз по городу.\n"
    "- Notion: создавай и читай заметки/задачи в рабочей базе.\n"
    "- Web Search: выполняй быстрый веб-поиск для фактов/ссылок.\n\n"
    "Правила:\n"
    "- Если вопрос требует фактов/ссылок — предложи выполнить поиск.\n"
    "- Если пользователь просит сохранить мысль/задачу — предложи записать в Notion.\n"
    "- Если речь о погоде — уточни город/дату, затем запроси Weather.\n"
    "- Пиши вежливо, избегай лишней болтовни.\n"
)


def build_system_prompt(extra_policy: Optional[str] = None) -> str:
    if extra_policy:
        return f"{SYSTEM_PROMPT}\nДополнительные правила:\n{extra_policy.strip()}"
    return SYSTEM_PROMPT


def build_user_prompt(message: str, *, memory_snippets: Optional[Iterable[str]] = None, rag_snippets: Optional[Iterable[str]] = None) -> str:
    parts: list[str] = []
    if memory_snippets:
        parts.append("Контекст прошлых сообщений:")
        for s in memory_snippets:
            parts.append(f"- {s}")
        parts.append("")
    if rag_snippets:
        parts.append("Найденные факты из базы знаний:")
        for s in rag_snippets:
            parts.append(f"- {s}")
        parts.append("")
    parts.append(message.strip())
    return "\n".join(parts)
