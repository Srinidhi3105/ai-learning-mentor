import sqlite3
import json

from datetime import datetime

from backend.config import DATABASE_FILE
from backend.models import (
    HistoryMessage,
    StoredHistoryMessage
)


def get_connection():

    return sqlite3.connect(
        DATABASE_FILE
    )


def init_database():

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            source_pages TEXT,
            created_at TEXT NOT NULL
        )
    """)


    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_conversations_session
        ON conversations(session_id)
        """
    )


    connection.commit()

    connection.close()


def save_message(
    session_id,
    role,
    content,
    source_pages=None
):

    if source_pages is None:
        source_pages = []


    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        INSERT INTO conversations
        (
            session_id,
            role,
            content,
            source_pages,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            session_id,
            role,
            content,
            json.dumps(source_pages),
            datetime.now().isoformat()
        )
    )


    connection.commit()

    connection.close()


def get_chat_history(
    session_id,
    limit=6
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT role, content
        FROM conversations
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (
            session_id,
            limit
        )
    )


    rows = cursor.fetchall()

    connection.close()

    rows.reverse()


    return [
        HistoryMessage(
            role=role,
            content=content
        )
        for role, content in rows
    ]


def get_all_chat_history(
    session_id
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT
            role,
            content,
            source_pages
        FROM conversations
        WHERE session_id = ?
        ORDER BY id
        """,
        (
            session_id,
        )
    )


    rows = cursor.fetchall()

    connection.close()


    messages = []


    for role, content, source_pages in rows:

        pages = []

        if source_pages:

            try:
                pages = json.loads(
                    source_pages
                )

            except json.JSONDecodeError:
                pass


        messages.append(
            StoredHistoryMessage(
                role=role,
                content=content,
                source_pages=pages
            )
        )


    return messages


def delete_chat_history(
    session_id
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        DELETE FROM conversations
        WHERE session_id = ?
        """,
        (
            session_id,
        )
    )


    connection.commit()

    connection.close()