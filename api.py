"""AI Data Analyst API.

실행: .\\.venv\\Scripts\\uvicorn api:app --reload
문서: http://localhost:8000/docs

엔드포인트는 모두 일반 def다. run_agent와 OpenAI 호출은 응답을 기다리는 동안 멈추는(blocking) 코드라서,
async def 안에서 부르면 그동안 서버 전체가 다른 요청을 처리하지 못한다.
일반 def로 두면 FastAPI가 요청마다 스레드 풀에서 실행한다.
"""

import json
import logging
import os
import sqlite3
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import openai
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

import db
from agent import DEFAULT_MODE, DEFAULT_MODEL, MODES, build_system_prompt, run_agent
from analysis import ColumnMap, Dataset, load_data
from tools import MAX_MONTHS, breakdown, get_summary, monthly_trend

load_dotenv()
logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("agent").setLevel(logging.INFO)

app = FastAPI(title="AI Data Analyst")


# ---------------------------------------------------------------- 설정과 의존성


def data_dir() -> Path:
    # SQLite 파일과 업로드한 CSV를 두는 곳. 4주차 Docker에서는 이 폴더를 volume으로 연결한다.
    path = Path(os.getenv("APP_DATA_DIR", "data"))
    (path / "uploads").mkdir(parents=True, exist_ok=True)
    return path


def get_db() -> Iterator[sqlite3.Connection]:
    """요청마다 연결을 열고, 응답이 끝나면 닫는다."""
    conn = db.connect(data_dir() / "app.db")
    try:
        db.init_db(conn)
        yield conn
    finally:
        conn.close()


@lru_cache
def get_client() -> openai.OpenAI:
    return openai.OpenAI()  # .env의 OPENAI_API_KEY. 테스트에서는 dependency_overrides로 가짜 클라이언트를 넣는다


def get_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


Conn = Annotated[sqlite3.Connection, Depends(get_db)]


@lru_cache(maxsize=8)
def read_dataset(path: str, column_map: str, encoding: str, date_format: str | None) -> pd.DataFrame:
    """업로드된 CSV를 역할 이름 컬럼의 df로 읽는다. 데이터셋은 바뀌지 않으므로 같은 인자면 캐시를 쓴다.

    lru_cache는 인자가 hashable이어야 해서 column_map을 JSON 문자열로 받는다.
    """
    dataset = Dataset(
        path=Path(path),
        columns=ColumnMap(**json.loads(column_map)),
        currency="",
        encoding=encoding,
        date_format=date_format,
    )
    return load_data(dataset)


def dataset_or_404(conn: sqlite3.Connection, dataset_id: str) -> tuple[dict, pd.DataFrame]:
    dataset = db.get_dataset(conn, dataset_id)
    if dataset is None:
        raise HTTPException(404, f"데이터셋이 없습니다: {dataset_id}")
    df = read_dataset(dataset["path"], json.dumps(dataset["column_map"]), dataset["encoding"], dataset["date_format"])
    # 캐시된 df를 도구가 바꾸지 않도록 복사본을 넘긴다.
    return dataset, df.copy()


# ---------------------------------------------------------------- 요청·응답 모델


class DatasetOut(BaseModel):
    dataset_id: str
    filename: str
    rows: int
    period: str


class ChatRequest(BaseModel):
    dataset_id: str
    conversation_id: str | None = None  # 없으면 새 대화를 시작한다
    question: str = Field(min_length=1, max_length=2000)
    mode: Literal["fast", "standard", "careful"] = DEFAULT_MODE


class Action(BaseModel):
    action: str
    reason: str
    priority: Literal["high", "medium", "low"]


class ToolCall(BaseModel):
    name: str
    arguments: str  # LLM이 보낸 인자 JSON 문자열 그대로


class Usage(BaseModel):
    mode: str
    steps: int
    input_tokens: int
    output_tokens: int
    seconds: float


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    suggested_actions: list[Action]
    tools_used: list[ToolCall]
    usage: Usage


class Turn(BaseModel):
    question: str
    mode: str
    answer: str
    suggested_actions: list[Action]
    tools_used: list[ToolCall]
    usage: Usage
    created_at: str


class ConversationOut(BaseModel):
    conversation_id: str
    dataset_id: str
    turns: list[Turn]


# ---------------------------------------------------------------- 엔드포인트


@app.post("/datasets", response_model=DatasetOut)
def upload_dataset(
    conn: Conn,
    file: Annotated[UploadFile, File(description="CSV 파일")],
    column_map: Annotated[str, Form(description='JSON. 예: {"date": "Order Date", "sales": "Sales", ...}')],
    currency: Annotated[str, Form()] = "USD",
    encoding: Annotated[str, Form()] = "utf-8",
    date_format: Annotated[str | None, Form()] = None,
) -> DatasetOut:
    try:
        mapping = json.loads(column_map)
    except json.JSONDecodeError as e:
        raise HTTPException(422, f"column_map이 JSON이 아닙니다: {e}") from e

    path = data_dir() / "uploads" / f"{db.new_id()}.csv"
    path.write_bytes(file.file.read())
    try:
        # 저장 전에 실제로 읽어 본다. 매핑·인코딩·날짜 형식이 틀리면 여기서 걸러진다.
        df = read_dataset(str(path), json.dumps(mapping), encoding, date_format or None)
    except (ValueError, TypeError) as e:  # 없는 컬럼, 인코딩 오류, 날짜 형식 오류, 매핑의 역할 이름 오류
        path.unlink()
        raise HTTPException(422, f"CSV를 읽지 못했습니다. 컬럼 매핑·인코딩·날짜 형식을 확인하세요. ({e})") from e

    dataset_id = db.create_dataset(conn, file.filename or "upload.csv", str(path), mapping, currency, encoding, date_format or None)
    return DatasetOut(
        dataset_id=dataset_id,
        filename=file.filename or "upload.csv",
        rows=len(df),
        period=f"{df['date'].min():%Y-%m} ~ {df['date'].max():%Y-%m}",
    )


@app.get("/datasets/{dataset_id}/summary")
def dataset_summary(dataset_id: str, conn: Conn) -> dict:
    """KPI 카드와 차트용 요약. 2주차 도구를 그대로 재사용한다."""
    dataset, df = dataset_or_404(conn, dataset_id)
    return {
        "currency": dataset["currency"],
        "summary": get_summary(df),
        "by_category": breakdown(df, "category"),
        "monthly_sales": monthly_trend(df, "sales", MAX_MONTHS),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    conn: Conn,
    client: Annotated[openai.OpenAI, Depends(get_client)],
    model: Annotated[str, Depends(get_model)],
) -> ChatResponse:
    dataset, df = dataset_or_404(conn, request.dataset_id)

    history: list[dict] = []
    if request.conversation_id is not None:
        conversation = db.get_conversation(conn, request.conversation_id)
        if conversation is None:
            raise HTTPException(404, f"대화가 없습니다: {request.conversation_id}")
        if conversation["dataset_id"] != request.dataset_id:
            raise HTTPException(400, "이 대화는 다른 데이터셋의 대화입니다.")
        history = db.load_history(conn, request.conversation_id)

    checkpoint = len(history)
    history.append({"role": "user", "content": request.question})
    try:
        result = run_agent(
            client, model, df, build_system_prompt(df, dataset["currency"]), history, MODES[request.mode]
        )
    except (openai.OpenAIError, RuntimeError) as e:
        # LLM 쪽 실패는 저장하지 않는다. 새 대화였다면 대화도 만들지 않는다.
        raise HTTPException(502, f"답변 생성에 실패했습니다: {e}") from e

    # 답변이 나온 뒤에 대화를 만들어서, 실패한 첫 질문 때문에 빈 대화가 남지 않게 한다.
    conversation_id = request.conversation_id or db.create_conversation(conn, request.dataset_id)
    db.save_turn(
        conn,
        conversation_id,
        request.question,
        request.mode,
        result.answer,
        result.tools_used,
        result.usage,
        history[checkpoint:],
    )
    return ChatResponse(conversation_id=conversation_id, **result.answer, tools_used=result.tools_used, usage=result.usage)


@app.get("/conversations/{conversation_id}", response_model=ConversationOut)
def get_conversation(conversation_id: str, conn: Conn) -> ConversationOut:
    conversation = db.get_conversation(conn, conversation_id)
    if conversation is None:
        raise HTTPException(404, f"대화가 없습니다: {conversation_id}")
    turns = [
        Turn(
            question=turn["question"],
            mode=turn["mode"],
            answer=turn["answer"]["answer"],
            suggested_actions=turn["answer"]["suggested_actions"],
            tools_used=turn["tools_used"],
            usage=turn["usage"],
            created_at=turn["created_at"],
        )
        for turn in db.list_turns(conn, conversation_id)
    ]
    return ConversationOut(conversation_id=conversation_id, dataset_id=conversation["dataset_id"], turns=turns)
