import sqlite3

import pytest

import db

ANSWER = {"answer": "답변", "suggested_actions": []}
USAGE = {"mode": "fast", "steps": 1, "input_tokens": 10, "output_tokens": 5, "seconds": 1.0}


@pytest.fixture
def conn(tmp_path) -> sqlite3.Connection:
    conn = db.connect(tmp_path / "test.db")
    db.init_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def conversation_id(conn) -> str:
    dataset_id = db.create_dataset(conn, "a.csv", "/data/a.csv", {"date": "Order Date"}, "USD", "utf-8", None)
    return db.create_conversation(conn, dataset_id)


def test_dataset_round_trip(conn):
    dataset_id = db.create_dataset(conn, "a.csv", "/data/a.csv", {"date": "Order Date"}, "USD", "cp1252", "%m/%d/%Y")

    dataset = db.get_dataset(conn, dataset_id)
    assert dataset["column_map"] == {"date": "Order Date"}  # JSON 문자열이 dict로 돌아온다
    assert dataset["encoding"] == "cp1252"
    assert db.get_dataset(conn, "없는id") is None


def test_conversation_requires_existing_dataset(conn):
    # 외래 키 검사를 켜 두었으므로 없는 데이터셋으로는 대화를 만들 수 없다.
    with pytest.raises(sqlite3.IntegrityError):
        db.create_conversation(conn, "없는id")


def test_history_keeps_order_across_turns(conn, conversation_id):
    first = [{"role": "user", "content": "q1"}, {"type": "message", "text": "a1"}]
    second = [{"role": "user", "content": "q2"}, {"type": "function_call", "call_id": "c1"}, {"type": "message"}]

    db.save_turn(conn, conversation_id, "q1", "fast", ANSWER, [], USAGE, first)
    db.save_turn(conn, conversation_id, "q2", "fast", ANSWER, [{"name": "get_summary"}], USAGE, second)

    assert db.load_history(conn, conversation_id) == first + second
    turns = db.list_turns(conn, conversation_id)
    assert [t["question"] for t in turns] == ["q1", "q2"]
    assert turns[1]["tools_used"] == [{"name": "get_summary"}]
    assert turns[1]["answer"] == ANSWER


def test_conversations_do_not_share_history(conn, conversation_id):
    other = db.create_conversation(conn, db.get_conversation(conn, conversation_id)["dataset_id"])
    db.save_turn(conn, conversation_id, "q1", "fast", ANSWER, [], USAGE, [{"role": "user", "content": "q1"}])

    assert db.load_history(conn, other) == []


def test_failed_save_leaves_nothing(conn, conversation_id):
    # JSON으로 바꿀 수 없는 값이 섞이면 저장이 실패하고, history와 turn 모두 남지 않아야 한다.
    with pytest.raises(TypeError):
        db.save_turn(conn, conversation_id, "q", "fast", {"bad": object()}, [], USAGE, [{"role": "user", "content": "q"}])

    assert db.load_history(conn, conversation_id) == []
    assert db.list_turns(conn, conversation_id) == []


def test_init_db_adds_content_hash_to_old_database(tmp_path):
    # content_hash 컬럼이 생기기 전에 만든 DB 파일을 흉내 낸다.
    conn = db.connect(tmp_path / "old.db")
    conn.execute(
        "CREATE TABLE datasets (id TEXT PRIMARY KEY, filename TEXT NOT NULL, path TEXT NOT NULL, column_map TEXT NOT NULL,"
        " currency TEXT NOT NULL, encoding TEXT NOT NULL, date_format TEXT, created_at TEXT NOT NULL)"
    )
    conn.execute("INSERT INTO datasets VALUES ('old', 'a.csv', '/a.csv', '{}', 'USD', 'utf-8', NULL, 'now')")

    db.init_db(conn)
    dataset_id = db.create_dataset(conn, "b.csv", "/b.csv", {}, "USD", "utf-8", None, "hash-b")

    assert db.get_dataset(conn, "old")["content_hash"] is None  # 기존 행은 그대로 남는다
    assert db.find_dataset_by_hash(conn, "hash-b")["id"] == dataset_id
    conn.close()


def test_list_conversations_skips_empty_and_orders_by_last_question(conn):
    dataset_id = db.create_dataset(conn, "a.csv", "/a.csv", {}, "USD", "utf-8", None)
    first = db.create_conversation(conn, dataset_id)
    second = db.create_conversation(conn, dataset_id)
    db.create_conversation(conn, dataset_id)  # 질문이 없는 대화는 목록에 나오지 않는다
    db.save_turn(conn, first, "첫 질문", "fast", ANSWER, [], USAGE, [])
    db.save_turn(conn, second, "둘째", "fast", ANSWER, [], USAGE, [])
    db.save_turn(conn, first, "이어서", "fast", ANSWER, [], USAGE, [])  # 같은 초 안에 질문해도 first가 최근이다

    listed = db.list_conversations(conn, dataset_id)

    assert [(c["conversation_id"], c["title"], c["turn_count"]) for c in listed] == [
        (first, "첫 질문", 2),
        (second, "둘째", 1),
    ]
