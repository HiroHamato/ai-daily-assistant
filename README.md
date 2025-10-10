# Daily Assistant (Python + LLM)

Интеллектуальный ассистент с Telegram-интерфейсом, памятью (SQLite), векторной БЗ (FAISS), MCP-инструментами (Weather, Notion, Web Search) и ядром LLM через SambaNova Cloud API.

## Быстрый старт
1. Создайте и активируйте venv
2. Скопируйте `.env.example` в `.env` и заполните переменные
3. Установите зависимости:
```bash
pip install -r requirements.txt
```
4. Запустите бота:
```bash
python main.py
```

## Переменные окружения
См. `.env.example`. Обязательные: `TELEGRAM_BOT_TOKEN`, `SAMBANOVA_API_KEY` (и при использовании Notion — `NOTION_TOKEN`, `NOTION_DATABASE_ID`).

## Структура проекта
См. ТЗ (`ТЗ.md`), директории создаются по мере реализации.

## Лицензия
MIT или иная на ваш выбор.
