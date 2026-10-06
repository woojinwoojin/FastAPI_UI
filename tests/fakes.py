"""OpenAI API를 호출하지 않고, 미리 정해 둔 응답을 돌려주는 가짜 클라이언트 (Agent·API 테스트 공용)."""

import json
from types import SimpleNamespace

# strict 스키마라 LLM은 필터를 안 쓸 때도 모든 항목을 null로 채워 보낸다.
NO_FILTERS = {"category": None, "sub_category": None, "region": None, "segment": None}
FINAL = {"answer": "답변", "suggested_actions": [{"action": "A", "reason": "R", "priority": "high"}]}


class FakeItem(SimpleNamespace):
    """OpenAI 응답의 output 항목 흉내. 실제 항목처럼 model_dump()로 dict가 된다."""

    def model_dump(self, exclude_none: bool = False) -> dict:
        return {k: v for k, v in vars(self).items() if not (exclude_none and v is None)}


def function_call(name: str, arguments: dict, call_id: str) -> FakeItem:
    return FakeItem(type="function_call", name=name, arguments=json.dumps(arguments), call_id=call_id)


def response(output: list, text: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        output=output,
        output_text=text,
        status="completed",
        incomplete_details=None,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def final_response() -> SimpleNamespace:
    return response([FakeItem(type="message", status=None)], json.dumps(FINAL))


class FakeClient:
    def __init__(self, responses: list):
        self.queue = list(responses)
        self.requests = []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        # history는 이후에도 바뀌므로 호출 시점의 상태를 복사해 둔다.
        self.requests.append({**kwargs, "input": list(kwargs["input"])})
        return self.queue.pop(0)
