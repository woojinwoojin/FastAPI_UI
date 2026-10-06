"""API를 호출하지 않고, 미리 정해 둔 응답을 돌려주는 가짜 클라이언트로 루프를 검사한다."""

import json

import pandas as pd
import pytest

from agent import MODES, Mode, run_agent
from tests.fakes import FINAL, NO_FILTERS, FakeClient, final_response, function_call, response


@pytest.fixture
def df() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-10", "2024-02-05"]),
            "order_id": ["O1", "O2"],
            "customer_id": ["C1", "C2"],
            "category": ["A", "B"],
            "sales": [100, 300],
            "profit": [10, 30],
        }
    )
    return df


def test_answers_directly_without_tools(df):
    client = FakeClient([final_response()])

    result = run_agent(client, "m", df, "sys", [{"role": "user", "content": "안녕"}])

    assert result.answer == FINAL
    assert result.tools_used == []
    assert result.usage["steps"] == 1
    assert len(client.requests) == 1


def test_runs_requested_tools_and_returns_results(df):
    client = FakeClient(
        [
            # 한 번에 도구 두 개를 요청하는 경우
            response(
                [
                    function_call("get_summary", {"start_month": None, "end_month": None}, "call_1"),
                    function_call(
                        "breakdown",
                        {"group_by": "category", "filters": NO_FILTERS, "start_month": None, "end_month": None},
                        "call_2",
                    ),
                ]
            ),
            final_response(),
        ]
    )
    history = [{"role": "user", "content": "매출 알려줘"}]

    result = run_agent(client, "m", df, "sys", history)

    assert result.answer == FINAL
    assert [t["name"] for t in result.tools_used] == ["get_summary", "breakdown"]
    assert result.usage == {"mode": "standard", "steps": 2, "input_tokens": 20, "output_tokens": 10, "seconds": result.usage["seconds"]}

    # 두 번째 요청에는 모델의 도구 요청과, call_id로 짝지은 실행 결과가 들어 있어야 한다.
    second_input = client.requests[1]["input"]
    outputs = {item["call_id"]: json.loads(item["output"]) for item in second_input if isinstance(item, dict) and item.get("type") == "function_call_output"}
    assert outputs["call_1"]["total_sales"] == 400
    assert [r["name"] for r in outputs["call_2"]["rows"]] == ["B", "A"]


def test_tool_error_is_sent_back_to_model(df):
    client = FakeClient([response([function_call("no_such_tool", {}, "call_1")]), final_response()])

    run_agent(client, "m", df, "sys", [{"role": "user", "content": "q"}])

    last = client.requests[1]["input"][-1]
    assert "error" in json.loads(last["output"])


def test_forces_answer_on_last_step(df):
    mode = Mode("test", max_steps=2, max_tool_calls=10, reasoning_effort="low", guide="")
    client = FakeClient([response([function_call("get_summary", {"start_month": None, "end_month": None}, "c1")]), final_response()])

    run_agent(client, "m", df, "sys", [{"role": "user", "content": "q"}], mode)

    assert [r["tool_choice"] for r in client.requests] == ["auto", "none"]


def test_raises_when_model_never_stops_calling_tools(df):
    mode = Mode("test", max_steps=2, max_tool_calls=10, reasoning_effort="low", guide="")
    call = function_call("get_summary", {"start_month": None, "end_month": None}, "c")
    client = FakeClient([response([call]), response([call])])

    with pytest.raises(RuntimeError):
        run_agent(client, "m", df, "sys", [{"role": "user", "content": "q"}], mode)


def test_fast_mode_answers_after_one_tool_round(df):
    client = FakeClient([response([function_call("get_summary", {}, "c1")]), final_response()])

    run_agent(client, "m", df, "sys", [{"role": "user", "content": "q"}], MODES["fast"])

    assert [r["tool_choice"] for r in client.requests] == ["auto", "none"]
    assert client.requests[0]["reasoning"] == {"effort": "none"}
    assert client.requests[0]["instructions"].endswith(MODES["fast"].guide)


def test_tool_calls_over_limit_get_error_output(df):
    calls = [function_call("get_summary", {}, f"c{i}") for i in range(4)]
    client = FakeClient([response(calls), final_response()])

    run_agent(client, "m", df, "sys", [{"role": "user", "content": "q"}], MODES["fast"])

    outputs = [json.loads(item["output"]) for item in client.requests[1]["input"] if isinstance(item, dict) and item.get("type") == "function_call_output"]
    # 상한(3개)을 넘은 네 번째 호출도 call_id 짝을 맞추기 위해 output은 있어야 하고, 내용은 error다.
    assert [o.get("total_sales") for o in outputs[:3]] == [400, 400, 400]
    assert "상한" in outputs[3]["error"]


def test_skipped_calls_are_not_in_tools_used(df):
    calls = [function_call("get_summary", {}, f"c{i}") for i in range(4)]
    client = FakeClient([response(calls), final_response()])

    result = run_agent(client, "m", df, "sys", [{"role": "user", "content": "q"}], MODES["fast"])

    assert len(result.tools_used) == 3


def test_history_is_json_serializable(df):
    # 기록을 DB에 JSON으로 저장했다가 다음 질문 때 그대로 다시 보낼 수 있어야 한다.
    client = FakeClient([response([function_call("get_summary", {}, "c1")]), final_response()])
    history = [{"role": "user", "content": "q"}]

    run_agent(client, "m", df, "sys", history)

    assert json.loads(json.dumps(history)) == history
    assert history[1] == {"type": "function_call", "name": "get_summary", "arguments": "{}", "call_id": "c1"}
    assert history[-1] == {"type": "message"}  # None 값은 저장하지 않는다


def test_reports_progress_events(df):
    client = FakeClient([response([function_call("get_summary", {}, "c1")]), final_response()])
    events = []

    run_agent(client, "m", df, "sys", [{"role": "user", "content": "q"}], on_event=events.append)

    assert events == [
        {"type": "llm_call", "step": 1},
        {"type": "tool", "name": "get_summary", "arguments": "{}"},
        {"type": "llm_call", "step": 2},
    ]


def test_reports_skipped_calls(df):
    calls = [function_call("get_summary", {}, f"c{i}") for i in range(4)]
    client = FakeClient([response(calls), final_response()])
    events = []

    run_agent(client, "m", df, "sys", [{"role": "user", "content": "q"}], MODES["fast"], on_event=events.append)

    assert [e["type"] for e in events] == ["llm_call", "tool", "tool", "tool", "skip", "llm_call"]
