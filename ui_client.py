"""Streamlit 화면이 API 서버를 부를 때 쓰는 얇은 클라이언트.

화면 코드에 HTTP 세부 사항(URL, 상태 코드, 오류 본문)이 섞이지 않게 여기로 모은다.
테스트에서는 make_http()를 FastAPI TestClient로 바꿔, 서버를 띄우지 않고 화면 → API → Agent 흐름을 검사한다.
"""

import json
import os
from collections.abc import Iterator

import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000")
# Agent 답변은 careful 모드에서 20초 넘게 걸릴 수 있다.
TIMEOUT = 120


class ApiError(Exception):
    """화면에 그대로 보여줄 수 있는 오류 메시지."""


def make_http() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=TIMEOUT)


def connection_error(e: httpx.HTTPError) -> ApiError:
    return ApiError(f"API 서버에 연결하지 못했습니다 ({API_URL}). 서버가 켜져 있는지 확인하세요. ({e})")


def http_error(response: httpx.Response) -> ApiError:
    # FastAPI 오류 본문은 {"detail": ...} 형태다. 422 검증 오류는 detail이 목록이다.
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = response.text
    return ApiError(f"[{response.status_code}] {detail}")


class ApiClient:
    def __init__(self, http: httpx.Client):
        self.http = http

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = self.http.request(method, path, **kwargs)
        except httpx.HTTPError as e:
            raise connection_error(e) from e
        if response.status_code >= 400:
            raise http_error(response)
        return response.json()

    def chat_stream(
        self, dataset_id: str, conversation_id: str | None, question: str, mode: str
    ) -> Iterator[tuple[str, dict]]:
        """/chat/stream의 SSE 이벤트를 (event, data)로 하나씩 돌려준다. 마지막은 done이다.

        error 이벤트(스트리밍 도중의 실패)는 ApiError로 바꿔 던진다.
        """
        body = {"dataset_id": dataset_id, "conversation_id": conversation_id, "question": question, "mode": mode}
        try:
            with self.http.stream("POST", "/chat/stream", json=body) as response:
                if response.status_code >= 400:
                    response.read()  # 스트리밍 응답은 본문을 직접 읽어야 오류 내용을 볼 수 있다
                    raise http_error(response)
                event = None
                for line in response.iter_lines():
                    if line.startswith("event: "):
                        event = line.removeprefix("event: ")
                    elif line.startswith("data: "):
                        data = json.loads(line.removeprefix("data: "))
                        if event == "error":
                            raise ApiError(data["detail"])
                        yield event, data
        except httpx.HTTPError as e:
            raise connection_error(e) from e

    def upload_dataset(
        self, filename: str, data: bytes, column_map: dict, currency: str, encoding: str, date_format: str | None
    ) -> dict:
        form = {"column_map": json.dumps(column_map), "currency": currency, "encoding": encoding}
        if date_format:
            form["date_format"] = date_format
        return self._request("POST", "/datasets", files={"file": (filename, data, "text/csv")}, data=form)

    def suggest_mapping(self, filename: str, data: bytes, encoding: str) -> dict:
        return self._request(
            "POST", "/datasets/suggest-mapping", files={"file": (filename, data, "text/csv")}, data={"encoding": encoding}
        )

    def summary(self, dataset_id: str) -> dict:
        return self._request("GET", f"/datasets/{dataset_id}/summary")

    def chat(self, dataset_id: str, conversation_id: str | None, question: str, mode: str) -> dict:
        body = {"dataset_id": dataset_id, "conversation_id": conversation_id, "question": question, "mode": mode}
        return self._request("POST", "/chat", json=body)

    def list_conversations(self, dataset_id: str) -> list[dict]:
        return self._request("GET", f"/datasets/{dataset_id}/conversations")

    def conversation(self, conversation_id: str) -> dict:
        return self._request("GET", f"/conversations/{conversation_id}")
