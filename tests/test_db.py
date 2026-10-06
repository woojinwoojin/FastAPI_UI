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
