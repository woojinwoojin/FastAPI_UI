"""Streamlit 화면 테스트. 서버를 띄우지 않고, 화면의 HTTP 클라이언트를 FastAPI TestClient로 바꿔 끼운다.

TestClient도 httpx.Client라서 ui_client.ApiClient가 그대로 쓸 수 있다. LLM은 가짜 클라이언트다.
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import api
import ui_client
from analysis import SUPERSTORE
from tests.fakes import FINAL, FakeClient, FakeItem, final_response, function_call, response

pytestmark = pytest.mark.skipif(not SUPERSTORE.path.exists(), reason="기본 데이터(data/) 없음")


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    api.app.dependency_overrides[api.get_model] = lambda: "test-model"
    monkeypatch.setattr(ui_client, "make_http", lambda: TestClient(api.app))
    yield
    api.app.dependency_overrides.clear()


def run_app() -> AppTest:
    return AppTest.from_file("../app.py", default_timeout=60).run()  # 이 파일 기준 상대 경로


def test_example_dataset_shows_kpis(server):
    at = run_app()

    assert not at.exception
    assert not at.error
    assert [m.label for m in at.metric] == ["총 매출 (USD)", "총 이익 (USD)", "주문 수", "고객 수"]
    assert at.metric[2].value == "5,009"
    assert len(at.chat_input) == 1  # 질문은 사용자가 입력해야만 보낸다 (자동으로 LLM을 부르지 않음)


def test_chat_shows_answer_and_tools(server):
    client = FakeClient([response([function_call("get_summary", {}, "c1")]), final_response()])
    api.app.dependency_overrides[api.get_client] = lambda: client
    at = run_app()

    at.chat_input[0].set_value("매출 알려줘").run()

    assert not at.exception
    assert any(FINAL["answer"] in md.value for md in at.markdown)
    assert any("도구 1개" in caption.value for caption in at.caption)
    # 기본 모드는 standard다.
    assert client.requests[0]["reasoning"] == {"effort": "low"}


def test_tilde_ranges_are_not_strikethrough(server):
    # ~ 두 개 사이가 취소선으로 그려지지 않도록, 화면에 그리기 전에 이스케이프해야 한다.
    answer = {
        "answer": "할인 21~40% 구간은 -18.3%, 41%~ 구간은 -70.9%입니다.",
        "suggested_actions": [{"action": "21~40% 할인 축소", "reason": "0~20% 대비 손실", "priority": "high"}],
    }
    client = FakeClient([response([FakeItem(type="message")], json.dumps(answer, ensure_ascii=False))])
    api.app.dependency_overrides[api.get_client] = lambda: client
    at = run_app()

    at.chat_input[0].set_value("할인 1~2단계 비교").run()

    rendered = [md.value for md in at.markdown]
    assert r"할인 21\~40% 구간은 -18.3%, 41%\~ 구간은 -70.9%입니다." in rendered
    assert any(r"21\~40% 할인 축소" in value and r"0\~20% 대비 손실" in value for value in rendered)
    assert r"할인 1\~2단계 비교" in rendered  # 사용자 질문도 같은 방식으로 그린다
    assert not any("~" in value.replace(r"\~", "") for value in rendered)


def test_api_down_shows_error(monkeypatch):
    monkeypatch.setattr(ui_client, "make_http", lambda: httpx.Client(base_url="http://127.0.0.1:9", timeout=1))

    at = run_app()

    assert not at.exception
    assert "API 서버에 연결하지 못했습니다" in at.error[0].value
