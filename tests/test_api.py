"""API 테스트. OpenAI 대신 가짜 클라이언트를 넣고(dependency_overrides), 데이터는 임시 폴더에 저장한다."""

import json

import openai
import pytest
from fastapi.testclient import TestClient

import api
from tests.fakes import FINAL, FakeClient, final_response, function_call, response

CSV = """Order Date,Order ID,Customer ID,Category,Sales,Profit,Region
2024-01-10,O1,C1,A,100,10,East
2024-01-10,O1,C1,B,100,20,East
2024-02-05,O2,C2,A,200,-40,West
"""
MAPPING = {"date": "Order Date", "sales": "Sales", "order_id": "Order ID", "customer_id": "Customer ID",
           "category": "Category", "profit": "Profit", "region": "Region"}


@pytest.fixture
def http(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    api.app.dependency_overrides[api.get_model] = lambda: "test-model"
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def use_fake_llm(responses: list) -> FakeClient:
    client = FakeClient(responses)
    api.app.dependency_overrides[api.get_client] = lambda: client
    return client


def upload(http: TestClient, mapping: dict = MAPPING, csv: str = CSV, currency: str = "USD"):
    return http.post(
        "/datasets",
        files={"file": ("sales.csv", csv.encode(), "text/csv")},
        data={"column_map": json.dumps(mapping), "currency": currency},
    )


@pytest.fixture
def dataset_id(http) -> str:
    return upload(http).json()["dataset_id"]


def test_upload_dataset(http, tmp_path):
    res = upload(http)

    assert res.status_code == 200
    assert res.json()["rows"] == 3
    assert res.json()["period"] == "2024-01 ~ 2024-02"
    assert len(list((tmp_path / "uploads").iterdir())) == 1


@pytest.mark.parametrize(
    "mapping",
    [
        {**MAPPING, "sales": "Revenue"},  # CSV에 없는 컬럼
        {**MAPPING, "price": "Sales"},  # 없는 역할 이름
        {k: v for k, v in MAPPING.items() if k != "date"},  # 필수 역할 누락
    ],
)
def test_upload_with_bad_mapping_is_rejected_and_file_removed(http, tmp_path, mapping):
    res = upload(http, mapping)

    assert res.status_code == 422
    assert list((tmp_path / "uploads").iterdir()) == []


def test_upload_requires_column_map(http):
    res = http.post("/datasets", files={"file": ("sales.csv", CSV.encode(), "text/csv")})

    assert res.status_code == 422  # 필수 form 필드가 없으면 FastAPI가 자동으로 거절한다


def test_same_upload_reuses_dataset(http, tmp_path):
    first = upload(http).json()
    second = upload(http).json()

    assert second["dataset_id"] == first["dataset_id"]
    assert (first["reused"], second["reused"]) == (False, True)
    assert len(list((tmp_path / "uploads").iterdir())) == 1  # 파일도 한 번만 저장한다


@pytest.mark.parametrize(
    "change",
    [
        {"csv": CSV + "2024-03-01,O3,C3,A,50,5,East\n"},  # 내용이 다르다
        {"mapping": {k: v for k, v in MAPPING.items() if k != "region"}},  # 매핑이 다르다
        {"currency": "KRW"},  # 설정이 다르다
    ],
)
def test_different_upload_creates_new_dataset(http, change):
    first = upload(http).json()

    assert upload(http, **change).json()["dataset_id"] != first["dataset_id"]


def test_summary(http, dataset_id):
    res = http.get(f"/datasets/{dataset_id}/summary")

    assert res.status_code == 200
    body = res.json()
    assert body["summary"]["total_sales"] == 400
    assert [r["name"] for r in body["by_category"]["rows"]] == ["A", "B"]
    assert [m["month"] for m in body["monthly_sales"]["months"]] == ["2024-01", "2024-02"]


def test_summary_unknown_dataset(http):
    assert http.get("/datasets/nope/summary").status_code == 404


def test_chat_starts_conversation_and_saves_turn(http, dataset_id):
    use_fake_llm([response([function_call("get_summary", {}, "c1")]), final_response()])

    res = http.post("/chat", json={"dataset_id": dataset_id, "question": "매출 알려줘", "mode": "fast"})

    assert res.status_code == 200
    body = res.json()
    assert {k: body[k] for k in FINAL} == FINAL  # summary, findings, notes, suggested_actions
    assert body["tools_used"] == [{"name": "get_summary", "arguments": "{}"}]
    assert body["usage"]["mode"] == "fast"

    conversation = http.get(f"/conversations/{body['conversation_id']}").json()
    assert [t["question"] for t in conversation["turns"]] == ["매출 알려줘"]
    assert conversation["turns"][0]["suggested_actions"] == FINAL["suggested_actions"]


def test_follow_up_question_sends_previous_history(http, dataset_id):
    use_fake_llm([final_response()])
    first = http.post("/chat", json={"dataset_id": dataset_id, "question": "첫 질문"}).json()

    client = use_fake_llm([final_response()])
    http.post(
        "/chat", json={"dataset_id": dataset_id, "conversation_id": first["conversation_id"], "question": "후속 질문"}
    )

    # 두 번째 요청에는 첫 질문, 첫 답변, 후속 질문이 순서대로 들어 있어야 한다.
    sent = client.requests[0]["input"]
    assert sent[0] == {"role": "user", "content": "첫 질문"}
    assert sent[1]["type"] == "message"
    assert sent[-1] == {"role": "user", "content": "후속 질문"}
    turns = http.get(f"/conversations/{first['conversation_id']}").json()["turns"]
    assert [t["question"] for t in turns] == ["첫 질문", "후속 질문"]


def test_list_conversations_newest_first(http, dataset_id):
    use_fake_llm([final_response(), final_response(), final_response()])
    older = http.post("/chat", json={"dataset_id": dataset_id, "question": "첫 대화"}).json()["conversation_id"]
    newer = http.post("/chat", json={"dataset_id": dataset_id, "question": "둘째 대화"}).json()["conversation_id"]
    # 오래된 대화에 질문을 더하면 그 대화가 맨 위로 올라와야 한다.
    http.post("/chat", json={"dataset_id": dataset_id, "conversation_id": older, "question": "이어서"})

    conversations = http.get(f"/datasets/{dataset_id}/conversations").json()

    assert [(c["conversation_id"], c["title"], c["turn_count"]) for c in conversations] == [
        (older, "첫 대화", 2),
        (newer, "둘째 대화", 1),
    ]


def test_list_conversations_unknown_dataset(http):
    assert http.get("/datasets/nope/conversations").status_code == 404


def test_old_answer_format_is_converted(http, dataset_id, tmp_path):
    # 답변 구조를 바꾸기 전에 저장된 대화({"answer": 문자열})도 같은 구조로 보여야 한다.
    conn = api.db.connect(tmp_path / "app.db")
    conversation_id = api.db.create_conversation(conn, dataset_id)
    old = {"answer": "예전 답변", "suggested_actions": []}
    api.db.save_turn(conn, conversation_id, "q", "fast", old, [], {"mode": "fast", "steps": 1, "input_tokens": 1,
                     "output_tokens": 1, "seconds": 1.0}, [])
    conn.close()

    turn = http.get(f"/conversations/{conversation_id}").json()["turns"][0]

    assert (turn["summary"], turn["findings"], turn["notes"]) == ("예전 답변", [], [])


def test_chat_unknown_conversation(http, dataset_id):
    res = http.post("/chat", json={"dataset_id": dataset_id, "conversation_id": "nope", "question": "q"})

    assert res.status_code == 404


def test_chat_conversation_of_other_dataset(http, dataset_id):
    use_fake_llm([final_response()])
    conversation_id = http.post("/chat", json={"dataset_id": dataset_id, "question": "q"}).json()["conversation_id"]
    other_dataset = upload(http, currency="KRW").json()["dataset_id"]  # 설정이 다르면 다른 데이터셋

    res = http.post("/chat", json={"dataset_id": other_dataset, "conversation_id": conversation_id, "question": "q"})

    assert res.status_code == 400


@pytest.mark.parametrize(
    "body",
    [
        {"question": "q"},  # dataset_id 누락
        {"dataset_id": "x", "question": ""},  # 빈 질문
        {"dataset_id": "x", "question": "q", "mode": "turbo"},  # 없는 모드
    ],
)
def test_chat_validation(http, body):
    assert http.post("/chat", json=body).status_code == 422


class FailingClient:
    def __init__(self):
        self.responses = self

    def create(self, **kwargs):
        raise openai.APIConnectionError(request=None)


def test_llm_failure_returns_502_and_saves_nothing(http, dataset_id, tmp_path):
    api.app.dependency_overrides[api.get_client] = FailingClient

    res = http.post("/chat", json={"dataset_id": dataset_id, "question": "q"})

    assert res.status_code == 502
    # 새 대화였으므로 대화 자체가 만들어지지 않아야 한다.
    conn = api.db.connect(tmp_path / "app.db")
    assert conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 0


def read_sse(http: TestClient, body: dict) -> list[tuple[str, dict]]:
    """SSE 응답을 (event, data) 목록으로 읽는다."""
    events, event = [], None
    with http.stream("POST", "/chat/stream", json=body) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        for line in res.iter_lines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: "):
                events.append((event, json.loads(line.removeprefix("data: "))))
    return events


def test_chat_stream_sends_progress_then_answer(http, dataset_id):
    use_fake_llm([response([function_call("get_summary", {}, "c1")]), final_response()])

    events = read_sse(http, {"dataset_id": dataset_id, "question": "매출 알려줘", "mode": "fast"})

    assert [name for name, _ in events] == ["llm_call", "tool", "llm_call", "done"]
    assert events[1][1] == {"type": "tool", "name": "get_summary", "arguments": "{}"}
    done = events[-1][1]
    assert done["summary"] == FINAL["summary"]
    # 스트리밍으로 받은 답변도 /chat과 똑같이 저장된다.
    turns = http.get(f"/conversations/{done['conversation_id']}").json()["turns"]
    assert [t["question"] for t in turns] == ["매출 알려줘"]


def test_chat_stream_validates_before_streaming(http, dataset_id):
    res = http.post("/chat/stream", json={"dataset_id": dataset_id, "conversation_id": "nope", "question": "q"})

    assert res.status_code == 404  # 스트리밍을 시작하기 전이라 일반 HTTP 오류로 끝난다


def test_chat_stream_llm_failure_is_error_event(http, dataset_id, tmp_path):
    api.app.dependency_overrides[api.get_client] = FailingClient

    events = read_sse(http, {"dataset_id": dataset_id, "question": "q"})

    assert [name for name, _ in events] == ["llm_call", "error"]
    assert "답변 생성에 실패했습니다" in events[-1][1]["detail"]
    conn = api.db.connect(tmp_path / "app.db")
    assert conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 0
