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
