"""SQLite에 데이터셋 정보와 대화 기록을 저장한다.

- CSV 원본은 파일로 두고, 여기에는 경로·컬럼 매핑 같은 정보만 넣는다.
- 대화는 두 갈래로 저장한다.
  turns:         화면에 보여줄 질문·답변
  history_items: LLM에 다시 보낼 전체 기록 (도구 요청·결과 포함, seq 순서대로)
- 연결은 요청마다 새로 연다. sqlite3 연결은 기본적으로 만든 스레드에서만 쓸 수 있는데,
  FastAPI는 일반 def 엔드포인트를 여러 스레드에서 실행하기 때문이다.
"""

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    path        TEXT NOT NULL,
    column_map  TEXT NOT NULL,  -- JSON: {"date": "Order Date", ...}
    currency    TEXT NOT NULL,
    encoding    TEXT NOT NULL,
    date_format TEXT,
    created_at  TEXT NOT NULL,
    content_hash TEXT  -- 파일 내용 + 읽기 설정의 해시. 같은 업로드면 기존 데이터셋을 재사용한다
);
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(id),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id              INTEGER PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    question        TEXT NOT NULL,
    mode            TEXT NOT NULL,
    answer          TEXT NOT NULL,  -- JSON: {"answer", "suggested_actions"}
    tools_used      TEXT NOT NULL,  -- JSON
    usage           TEXT NOT NULL,  -- JSON
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS history_items (
    id              INTEGER PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    seq             INTEGER NOT NULL,
    item            TEXT NOT NULL,  -- JSON
    UNIQUE (conversation_id, seq)
);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_id() -> str:
    # 순번 대신 무작위 id를 쓴다. URL에 드러나도 다른 대화의 id를 짐작할 수 없다.
    return uuid.uuid4().hex


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row  # row["column"]으로 읽기
    conn.execute("PRAGMA foreign_keys = ON")  # SQLite는 외래 키 검사가 기본으로 꺼져 있다
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    # CREATE TABLE IF NOT EXISTS는 이미 있는 테이블을 바꾸지 않는다.
    # content_hash가 생기기 전에 만든 DB 파일이면 컬럼을 추가한다 (가장 단순한 마이그레이션).
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(datasets)")}
    if "content_hash" not in columns:
        conn.execute("ALTER TABLE datasets ADD COLUMN content_hash TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_content_hash ON datasets(content_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_dataset ON conversations(dataset_id)")
    conn.commit()


# ---------------------------------------------------------------- 데이터셋


def create_dataset(
    conn: sqlite3.Connection,
    filename: str,
    path: str,
    column_map: dict,
    currency: str,
    encoding: str,
    date_format: str | None,
    content_hash: str | None = None,
) -> str:
    dataset_id = new_id()
    with conn:  # 블록이 끝나면 commit, 예외가 나면 rollback
        # 마이그레이션으로 추가한 컬럼은 테이블 끝에 붙으므로, 순서에 기대지 않고 컬럼 이름을 적는다.
        conn.execute(
            "INSERT INTO datasets (id, filename, path, column_map, currency, encoding, date_format, created_at, content_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (dataset_id, filename, path, json.dumps(column_map), currency, encoding, date_format, now(), content_hash),
        )
    return dataset_id


def find_dataset_by_hash(conn: sqlite3.Connection, content_hash: str) -> dict | None:
    row = conn.execute("SELECT id FROM datasets WHERE content_hash = ? LIMIT 1", (content_hash,)).fetchone()
    return get_dataset(conn, row["id"]) if row else None


def get_dataset(conn: sqlite3.Connection, dataset_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
    if row is None:
        return None
    return {**dict(row), "column_map": json.loads(row["column_map"])}


# ---------------------------------------------------------------- 대화


def create_conversation(conn: sqlite3.Connection, dataset_id: str) -> str:
    conversation_id = new_id()
    with conn:
        conn.execute("INSERT INTO conversations VALUES (?, ?, ?)", (conversation_id, dataset_id, now()))
    return conversation_id


def get_conversation(conn: sqlite3.Connection, conversation_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
    return dict(row) if row else None


def list_conversations(conn: sqlite3.Connection, dataset_id: str) -> list[dict]:
    """데이터셋의 대화 목록. 최근에 질문한 대화부터, 첫 질문을 제목처럼 함께 돌려준다."""
    rows = conn.execute(
        """
        SELECT c.id,
               c.created_at,
               MAX(t.created_at) AS updated_at,
               COUNT(t.id)       AS turn_count,
               (SELECT question FROM turns WHERE conversation_id = c.id ORDER BY id LIMIT 1) AS title
        FROM conversations c
        JOIN turns t ON t.conversation_id = c.id
        WHERE c.dataset_id = ?
        GROUP BY c.id
        -- created_at은 초 단위라 같은 초에 질문하면 순서가 갈린다. turn id는 항상 늘어나므로 이것으로 정렬한다.
        ORDER BY MAX(t.id) DESC
        """,
        (dataset_id,),
    ).fetchall()
    return [
        {
            "conversation_id": row["id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "turn_count": row["turn_count"],
            "title": row["title"],
        }
        for row in rows
    ]


def load_history(conn: sqlite3.Connection, conversation_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT item FROM history_items WHERE conversation_id = ? ORDER BY seq", (conversation_id,)
    ).fetchall()
    return [json.loads(row["item"]) for row in rows]


def save_turn(
    conn: sqlite3.Connection,
    conversation_id: str,
    question: str,
    mode: str,
    answer: dict,
    tools_used: list[dict],
    usage: dict,
    new_items: list[dict],
) -> None:
    """질문 하나의 결과를 저장한다. new_items는 이번 질문으로 기록에 새로 붙은 항목(질문 포함)이다.

    화면용 turn과 LLM용 기록을 한 트랜잭션으로 저장해, 중간에 실패해도 둘 중 하나만 남지 않게 한다.
    """
    with conn:
        start = conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 FROM history_items WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()[0]
        conn.executemany(
            "INSERT INTO history_items (conversation_id, seq, item) VALUES (?, ?, ?)",
            [(conversation_id, start + i, json.dumps(item, ensure_ascii=False)) for i, item in enumerate(new_items)],
        )
        conn.execute(
            "INSERT INTO turns (conversation_id, question, mode, answer, tools_used, usage, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                question,
                mode,
                json.dumps(answer, ensure_ascii=False),
                json.dumps(tools_used, ensure_ascii=False),
                json.dumps(usage),
                now(),
            ),
        )


def list_turns(conn: sqlite3.Connection, conversation_id: str) -> list[dict]:
    rows = conn.execute("SELECT * FROM turns WHERE conversation_id = ? ORDER BY id", (conversation_id,)).fetchall()
    return [
        {
            "question": row["question"],
            "mode": row["mode"],
            "answer": json.loads(row["answer"]),
            "tools_used": json.loads(row["tools_used"]),
            "usage": json.loads(row["usage"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]
