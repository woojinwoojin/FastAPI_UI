"""AI Data Analyst API.

실행: .\\.venv\\Scripts\\uvicorn api:app --reload
문서: http://localhost:8000/docs

엔드포인트는 모두 일반 def다. run_agent와 OpenAI 호출은 응답을 기다리는 동안 멈추는(blocking) 코드라서,
async def 안에서 부르면 그동안 서버 전체가 다른 요청을 처리하지 못한다.
일반 def로 두면 FastAPI가 요청마다 스레드 풀에서 실행한다.
"""

import hashlib
import json
import logging
import os
import queue
import sqlite3
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import openai
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import db
from agent import DEFAULT_MODE, DEFAULT_MODEL, MODES, AgentResult, build_system_prompt, normalize_answer, run_agent
from analysis import ColumnMap, Dataset, load_data
from tools import MAX_MONTHS, breakdown, get_summary, monthly_trend
from verify import unverified_evidence

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
    reused: bool  # 같은 파일·설정으로 이미 등록된 데이터셋을 돌려줬는지


def dataset_out(dataset_id: str, filename: str, df: pd.DataFrame, reused: bool) -> DatasetOut:
    return DatasetOut(
        dataset_id=dataset_id,
        filename=filename,
        rows=len(df),
        period=f"{df['date'].min():%Y-%m} ~ {df['date'].max():%Y-%m}",
        reused=reused,
    )


def dataset_hash(data: bytes, mapping: dict, currency: str, encoding: str, date_format: str | None) -> str:
    """파일 내용과 읽기 설정이 모두 같아야 같은 데이터셋이다. 매핑이 다르면 분석 결과도 다르기 때문이다."""
    settings = json.dumps(
        {"column_map": mapping, "currency": currency, "encoding": encoding, "date_format": date_format}, sort_keys=True
    )
    return hashlib.sha256(data + settings.encode()).hexdigest()


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str  # 첫 질문
    turn_count: int
    created_at: str
    updated_at: str


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


class Finding(BaseModel):
    title: str
    detail: str
    evidence: list[str]  # 근거 숫자 조각


class Answer(BaseModel):
    """Agent 답변. 결론 → 발견 → 참고 → 액션으로 나눠 화면이 구조대로 그릴 수 있게 한다."""

    summary: str
    findings: list[Finding]
    notes: list[str]  # 기간 해석, 확인하지 못한 것, 가설
    suggested_actions: list[Action]
    # 도구 결과에서 숫자를 찾지 못한 근거 문구 (verify.py). LLM이 계산했거나 잘못 옮겼을 수 있다.
    unverified_evidence: list[str] = []


class ChatResponse(Answer):
    conversation_id: str
    tools_used: list[ToolCall]
    usage: Usage


class Turn(Answer):
    question: str
    mode: str
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

    data = file.file.read()
    date_format = date_format or None
    # 같은 파일을 같은 설정으로 다시 올리면 기존 데이터셋을 돌려준다.
    # 그래야 화면을 새로고침해도 데이터셋 id가 그대로라 이전 대화를 이어 갈 수 있다.
    content_hash = dataset_hash(data, mapping, currency, encoding, date_format)
    if (existing := db.find_dataset_by_hash(conn, content_hash)) is not None:
        _, df = dataset_or_404(conn, existing["id"])
        return dataset_out(existing["id"], existing["filename"], df, reused=True)

    path = data_dir() / "uploads" / f"{db.new_id()}.csv"
    path.write_bytes(data)
    try:
        # 저장 전에 실제로 읽어 본다. 매핑·인코딩·날짜 형식이 틀리면 여기서 걸러진다.
        df = read_dataset(str(path), json.dumps(mapping), encoding, date_format)
    except (ValueError, TypeError) as e:  # 없는 컬럼, 인코딩 오류, 날짜 형식 오류, 매핑의 역할 이름 오류
        path.unlink()
        raise HTTPException(422, f"CSV를 읽지 못했습니다. 컬럼 매핑·인코딩·날짜 형식을 확인하세요. ({e})") from e

    filename = file.filename or "upload.csv"
    dataset_id = db.create_dataset(conn, filename, str(path), mapping, currency, encoding, date_format, content_hash)
    return dataset_out(dataset_id, filename, df, reused=False)


@app.get("/datasets/{dataset_id}/conversations", response_model=list[ConversationSummary])
def list_conversations(dataset_id: str, conn: Conn) -> list[ConversationSummary]:
    """이전 대화 목록. 최근에 질문한 대화부터."""
    if db.get_dataset(conn, dataset_id) is None:
        raise HTTPException(404, f"데이터셋이 없습니다: {dataset_id}")
    return [ConversationSummary(**c) for c in db.list_conversations(conn, dataset_id)]


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


@dataclass
class ChatContext:
    dataset: dict
    df: pd.DataFrame
    history: list[dict]
    checkpoint: int  # 이번 질문 전까지의 기록 길이. 이후 항목이 이번 질문으로 새로 생긴 기록이다


def prepare_chat(conn: sqlite3.Connection, request: ChatRequest) -> ChatContext:
    """질문을 처리하기 전 검증과 기록 불러오기. 잘못된 요청은 여기서 404/400으로 끝난다."""
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
    return ChatContext(dataset, df, history, checkpoint)


def run_chat_agent(
    client: openai.OpenAI,
    model: str,
    request: ChatRequest,
    ctx: ChatContext,
    on_event: Callable[[dict], None] | None = None,
) -> AgentResult:
    instructions = build_system_prompt(ctx.df, ctx.dataset["currency"])
    return run_agent(client, model, ctx.df, instructions, ctx.history, MODES[request.mode], on_event)


def finish_chat(conn: sqlite3.Connection, request: ChatRequest, ctx: ChatContext, result: AgentResult) -> ChatResponse:
    # 답변이 나온 뒤에 대화를 만들어서, 실패한 첫 질문 때문에 빈 대화가 남지 않게 한다.
    conversation_id = request.conversation_id or db.create_conversation(conn, request.dataset_id)
    # 근거 숫자 검증 결과를 답변과 함께 저장해, 나중에 대화를 불러와도 같은 표시가 보이게 한다.
    answer = {**result.answer, "unverified_evidence": unverified_evidence(result.answer, ctx.history)}
    db.save_turn(
        conn,
        conversation_id,
        request.question,
        request.mode,
        answer,
        result.tools_used,
        result.usage,
        ctx.history[ctx.checkpoint :],
    )
    return ChatResponse(conversation_id=conversation_id, **answer, tools_used=result.tools_used, usage=result.usage)


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    conn: Conn,
    client: Annotated[openai.OpenAI, Depends(get_client)],
    model: Annotated[str, Depends(get_model)],
) -> ChatResponse:
    ctx = prepare_chat(conn, request)
    try:
        result = run_chat_agent(client, model, request, ctx)
    except (openai.OpenAIError, RuntimeError) as e:
        # LLM 쪽 실패는 저장하지 않는다. 새 대화였다면 대화도 만들지 않는다.
        raise HTTPException(502, f"답변 생성에 실패했습니다: {e}") from e
    return finish_chat(conn, request, ctx, result)


def sse(event: str, data: dict) -> str:
    """Server-Sent Events 한 건. 빈 줄로 이벤트를 구분한다."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
def chat_stream(
    request: ChatRequest,
    conn: Conn,
    client: Annotated[openai.OpenAI, Depends(get_client)],
    model: Annotated[str, Depends(get_model)],
) -> StreamingResponse:
    """/chat과 같지만, 진행 상황을 SSE 이벤트로 보낸다.

    이벤트: llm_call, tool, skip (진행 상황) → done (ChatResponse) 또는 error ({"detail"})
    스트리밍이 시작되면 상태 코드는 이미 200이라, 도중의 실패는 error 이벤트로 알린다.
    """
    ctx = prepare_chat(conn, request)  # 잘못된 요청은 스트리밍 전에 일반 HTTP 오류로 끝낸다
    events: queue.Queue[tuple[str, dict]] = queue.Queue()

    def work() -> None:
        # Agent는 별도 스레드에서 돌리고, 진행 상황은 큐를 거쳐 응답으로 흘려보낸다.
        try:
            result = run_chat_agent(client, model, request, ctx, on_event=lambda e: events.put((e["type"], e)))
            # sqlite3 연결은 만든 스레드에서만 쓸 수 있으므로, 이 스레드에서 저장할 연결을 따로 연다.
            worker_conn = db.connect(data_dir() / "app.db")
            try:
                response = finish_chat(worker_conn, request, ctx, result)
            finally:
                worker_conn.close()
            events.put(("done", response.model_dump()))
        except (openai.OpenAIError, RuntimeError) as e:
            events.put(("error", {"detail": f"답변 생성에 실패했습니다: {e}"}))
        except Exception:
            logging.getLogger(__name__).exception("chat_stream 실패")
            events.put(("error", {"detail": "서버 오류로 답변을 만들지 못했습니다."}))

    threading.Thread(target=work, daemon=True).start()

    def stream() -> Iterator[str]:
        while True:
            event, data = events.get()
            yield sse(event, data)
            if event in ("done", "error"):
                return

    # 일반 이터레이터라 Starlette가 스레드 풀에서 돌린다. events.get()에서 기다려도 서버가 멈추지 않는다.
    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/conversations/{conversation_id}", response_model=ConversationOut)
def get_conversation(conversation_id: str, conn: Conn) -> ConversationOut:
    conversation = db.get_conversation(conn, conversation_id)
    if conversation is None:
        raise HTTPException(404, f"대화가 없습니다: {conversation_id}")
    turns = [
        Turn(
            **normalize_answer(turn["answer"]),  # 예전 형식으로 저장된 답변도 같은 구조로 보여준다
            question=turn["question"],
            mode=turn["mode"],
            tools_used=turn["tools_used"],
            usage=turn["usage"],
            created_at=turn["created_at"],
        )
        for turn in db.list_turns(conn, conversation_id)
    ]
    return ConversationOut(conversation_id=conversation_id, dataset_id=conversation["dataset_id"], turns=turns)
