"""Streamlit 화면이 API 서버를 부를 때 쓰는 얇은 클라이언트.

화면 코드에 HTTP 세부 사항(URL, 상태 코드, 오류 본문)이 섞이지 않게 여기로 모은다.
테스트에서는 make_http()를 FastAPI TestClient로 바꿔, 서버를 띄우지 않고 화면 → API → Agent 흐름을 검사한다.
"""

import json
import os

import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000")
# Agent 답변은 careful 모드에서 20초 넘게 걸릴 수 있다.
TIMEOUT = 120


class ApiError(Exception):
    """화면에 그대로 보여줄 수 있는 오류 메시지."""


def make_http() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=TIMEOUT)


class ApiClient:
    def __init__(self, http: httpx.Client):
        self.http = http

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = self.http.request(method, path, **kwargs)
        except httpx.HTTPError as e:
            raise ApiError(f"API 서버에 연결하지 못했습니다 ({API_URL}). 서버가 켜져 있는지 확인하세요. ({e})") from e
        if response.status_code >= 400:
            # FastAPI 오류 본문은 {"detail": ...} 형태다. 422 검증 오류는 detail이 목록이다.
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = response.text
            raise ApiError(f"[{response.status_code}] {detail}")
        return response.json()

    def upload_dataset(
        self, filename: str, data: bytes, column_map: dict, currency: str, encoding: str, date_format: str | None
    ) -> dict:
        form = {"column_map": json.dumps(column_map), "currency": currency, "encoding": encoding}
        if date_format:
            form["date_format"] = date_format
        return self._request("POST", "/datasets", files={"file": (filename, data, "text/csv")}, data=form)

    def summary(self, dataset_id: str) -> dict:
        return self._request("GET", f"/datasets/{dataset_id}/summary")

    def chat(self, dataset_id: str, conversation_id: str | None, question: str, mode: str) -> dict:
        body = {"dataset_id": dataset_id, "conversation_id": conversation_id, "question": question, "mode": mode}
        return self._request("POST", "/chat", json=body)

    def conversation(self, conversation_id: str) -> dict:
        return self._request("GET", f"/conversations/{conversation_id}")
