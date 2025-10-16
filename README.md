# Daily Assistant (Python + LLM)

Интеллектуальный ассистент с Telegram-интерфейсом, памятью (SQLite), векторной БЗ (FAISS), MCP-инструментами (Weather, Notion, Web Search) и ядром LLM через SambaNova Cloud API.

## Быстрый старт (локально)
1. Создайте и активируйте venv
2. Скопируйте `env.example` в `.env` и заполните переменные
3. Установите зависимости:
```bash
pip install -r requirements.txt
```
4. Запустите бота:
```bash
python main.py
```

## Деплой (systemd)
1. Скопируйте проект на сервер (Ubuntu/Debian), например `/opt/daily-assistant`
2. Скопируйте `env.example` в `.env` и заполните значения (TOKENы и т.п.)
3. Настройте сервис:
```bash
sudo cp agent.service /etc/systemd/system/agent.service
sudo systemctl daemon-reload
sudo systemctl enable --now agent
```
4. Проверка:
```bash
sudo systemctl status agent
journalctl -u agent -f
```

## Docker (опционально)
```bash
docker compose up --build -d
```

## Используемые MCP-инструменты
- Weather (Open‑Meteo): геокодинг и текущая/дневная погода. Маркер: `[[WEATHER:Город]]`.
- Notion: поиск заметок и создание страниц в базе. Маркеры: `[[NOTION_SEARCH:запрос]]`, `[[NOTION_ADD:Заголовок|Текст]]`.
- Web Search (DuckDuckGo News): новости за 24 часа, приоритет крипто‑домены. Маркер: `[[WEB_SEARCH:запрос]]`.
- KB (FAISS): локальная база знаний — добавление и поиск фактов. Маркеры: `[[KB_ADD:факт]]`, `[[KB_SEARCH:запрос]]`.

## Краткий отчёт
- Назначение: ежедневный ассистент в Telegram с памятью, RAG и инструментами.
- Подключённые MCP: Weather, Notion, Web Search, локальная KB (FAISS).
- Память: SQLite (`memory/dialogue.db`), хранит историю для контекста.
- База знаний: FAISS + SentenceTransformers, асинхронная инициализация, RAG‑вставки.
- LLM: SambaNova Cloud `/chat/completions`, модель задаётся в `.env`, встроенный троттлинг/429‑retry.
- Оркестрация: система tool‑call маркеров, mini‑loop (1 инструмент → финальный ответ), статусы выполнения в чате.

## Папки/файлы
- `telegram/` — Telegram-бот
- `llm/` — клиент SambaNova LLaMA
- `memory/` — SQLite-хранилище
- `vector_store/` — FAISS
- `mcp/` — инструменты Weather/Notion/Web Search
- `utils/` — промпты
- `agent.service` — unit для systemd (в корне)
- `Dockerfile`, `docker-compose.yml` — для контейнерного запуска
- `report.md` — краткий отчёт

## Переменные окружения
См. `env.example`.
