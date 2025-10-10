import aiosqlite
from typing import Optional, Iterable, Tuple


class DialogueMemory:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    async def init(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    ts DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await db.commit()

    async def add(self, chat_id: str, role: str, content: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
                (chat_id, role, content),
            )
            await db.commit()

    async def history(self, chat_id: str, limit: Optional[int] = 50) -> Iterable[Tuple[str, str]]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
                (chat_id, limit or 50),
            )
            rows = await cursor.fetchall()
            # return in chronological order
            return list(reversed([(r[0], r[1]) for r in rows]))
