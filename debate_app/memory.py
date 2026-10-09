import json
import sqlite3
from pathlib import Path
from typing import Any

from .schemas import DebateInput, DebateMessage, JudgeResult


class MemoryStore:
    """SQLite-backed local memory with one transaction per persisted event."""

    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS debates (
                    debate_id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    question TEXT NOT NULL,
                    context TEXT NOT NULL,
                    max_rounds INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    judgment_json TEXT,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    debate_id TEXT NOT NULL,
                    round_number INTEGER NOT NULL,
                    side TEXT NOT NULL,
                    content TEXT NOT NULL,
                    message_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(debate_id, round_number, side),
                    FOREIGN KEY(debate_id) REFERENCES debates(debate_id)
                );
                """
            )

    def create_debate(self, debate_id: str, data: DebateInput) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO debates VALUES (?, ?, ?, ?, ?, ?, datetime('now'), NULL, NULL, NULL)",
                (debate_id, data.topic, data.question, data.context, data.max_rounds, "running"),
            )

    def append_message(self, message: DebateMessage) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO messages
                (debate_id, round_number, side, content, message_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    str(message.debate_id),
                    message.round_number,
                    message.side,
                    message.content,
                    message.model_dump_json(),
                    message.created_at.isoformat(),
                ),
            )

    def complete_debate(self, debate_id: str, judgment: JudgeResult) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE debates SET status = 'completed', completed_at = datetime('now'), judgment_json = ? WHERE debate_id = ?",
                (judgment.model_dump_json(), debate_id),
            )

    def fail_debate(self, debate_id: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE debates SET status = 'failed', error = ? WHERE debate_id = ?",
                (error[:2000], debate_id),
            )

    def get_debate(self, debate_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            debate = connection.execute("SELECT * FROM debates WHERE debate_id = ?", (debate_id,)).fetchone()
            if debate is None:
                return None
            messages = connection.execute(
                "SELECT message_json FROM messages WHERE debate_id = ? ORDER BY round_number, side", (debate_id,)
            ).fetchall()
        result = dict(debate)
        result["messages"] = [json.loads(row["message_json"]) for row in messages]
        return result

