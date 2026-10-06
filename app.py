"""Streamlit 화면. 분석·Agent는 모두 API 서버가 하고, 이 화면은 API를 부르고 결과를 보여주기만 한다.

실행 (API 서버를 먼저 켠다):
  .\\.venv\\Scripts\\uvicorn api:app --reload
  .\\.venv\\Scripts\\streamlit run app.py
"""

import io
import json
from datetime import datetime

import pandas as pd
import streamlit as st

import ui_client
from analysis import SUPERSTORE
from ui_client import ApiClient, ApiError

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

MODES = {"fast": "빠른 답변 (약 4초)", "standard": "기본 (약 8초)", "careful": "신중 (10~20초)"}
# 1주차 매핑 화면에서 가져와, 2주차에 추가한 선택 역할(하위 카테고리·할인·지역·세그먼트)을 더했다.
REQUIRED_ROLES = {
    "date": ("주문 날짜", ["order date", "date", "주문일", "날짜"]),
    "sales": ("매출 금액", ["sales", "revenue", "amount", "매출", "금액"]),
    "order_id": ("주문 ID", ["order id", "order_id", "invoice", "주문"]),
    "customer_id": ("고객 ID", ["customer id", "customer_id", "user_id", "고객"]),
    "category": ("카테고리", ["category", "카테고리", "분류"]),
}
OPTIONAL_ROLES = {
    "profit": ("이익", ["profit", "이익"]),
    "sub_category": ("하위 카테고리", ["sub-category", "sub_category", "subcategory", "하위"]),
    "discount": ("할인율", ["discount", "할인"]),
    "region": ("지역", ["region", "지역"]),
    "segment": ("고객 세그먼트", ["segment", "세그먼트"]),
}
NONE = "(없음)"
NEW_CONVERSATION = "new"  # 대화 선택 상자에서 "새 대화"를 뜻하는 값
PRIORITY = {"high": "🔴 높음", "medium": "🟡 중간", "low": "⚪ 낮음"}


def api() -> ApiClient:
    # make_http를 모듈 속성으로 찾아야 테스트에서 바꿔 끼울 수 있다.
    return ApiClient(ui_client.make_http())


def md(text: str) -> str:
    """마크다운으로 그릴 텍스트의 물결표를 이스케이프한다.

    Streamlit 마크다운은 ~ 두 개 사이를 취소선으로 그린다. LLM 답변에는 "21~40%", "2014-01 ~ 2017-12" 같은
    범위 표현이 자주 나와서, 그대로 두면 문장 중간에 줄이 그어진다.
    """
    return text.replace("~", r"\~")


def guess_column(columns: list[str], keywords: list[str]) -> int:
    """컬럼 이름에 키워드가 들어 있으면 그 위치를, 없으면 0을 돌려준다."""
    for i, col in enumerate(columns):
        if any(k in col.lower() for k in keywords):
            return i
    return 0


# ---------------------------------------------------------------- 데이터셋 선택


def register_example() -> dict:
    """기본 예시(Superstore)를 API에 등록한다. 세션마다 한 번만 올린다."""
    if "example_dataset" not in st.session_state:
        mapping = {role: col for role, col in vars(SUPERSTORE.columns).items() if col is not None}
        st.session_state.example_dataset = api().upload_dataset(
            SUPERSTORE.path.name,
            SUPERSTORE.path.read_bytes(),
            mapping,
            SUPERSTORE.currency,
            SUPERSTORE.encoding,
            SUPERSTORE.date_format,
        )
    return st.session_state.example_dataset


def register_upload() -> dict | None:
    uploaded = st.sidebar.file_uploader("CSV 파일", type="csv")
    if uploaded is None:
        return st.session_state.get("uploaded_dataset")

    data = uploaded.getvalue()
    encoding = st.sidebar.selectbox("인코딩", ["utf-8", "cp1252", "cp949"])
    try:
        columns = list(pd.read_csv(io.BytesIO(data), nrows=0, encoding=encoding).columns)
    except (UnicodeDecodeError, pd.errors.ParserError) as e:
        st.sidebar.error(f"파일을 읽지 못했습니다. 인코딩을 바꿔 보세요. ({e})")
        return None

    # form으로 묶어 매핑을 고르는 동안에는 화면이 다시 실행되지 않게 한다.
    with st.sidebar.form("mapping"):
        st.subheader("컬럼 매핑")
        mapping = {}
        for role, (label, keywords) in REQUIRED_ROLES.items():
            mapping[role] = st.selectbox(label, columns, index=guess_column(columns, keywords))
        for role, (label, keywords) in OPTIONAL_ROLES.items():
            options = [NONE] + columns
            # 키워드가 맞는 컬럼이 없으면 (없음)으로 둔다.
            guessed = guess_column(columns, keywords)
            index = guessed + 1 if any(k in columns[guessed].lower() for k in keywords) else 0
            choice = st.selectbox(f"{label} (선택)", options, index=index)
            if choice != NONE:
                mapping[role] = choice
        date_format = st.text_input("날짜 형식 (비우면 자동)", placeholder="%m/%d/%Y") or None
        currency = st.text_input("통화 단위", value="KRW")
        submitted = st.form_submit_button("데이터셋 등록")

    if submitted:
        try:
            st.session_state.uploaded_dataset = api().upload_dataset(
                uploaded.name, data, mapping, currency, encoding, date_format
            )
        except ApiError as e:
            st.sidebar.error(str(e))
    return st.session_state.get("uploaded_dataset")


def select_dataset() -> dict | None:
    st.sidebar.header("데이터")
    source = st.sidebar.radio("데이터 선택", ["기본 예시 (Superstore)", "CSV 업로드"])
    if source == "기본 예시 (Superstore)":
        if not SUPERSTORE.path.exists():
            st.error(f"기본 데이터가 없습니다: {SUPERSTORE.path}")
            return None
        return register_example()
    return register_upload()


# ---------------------------------------------------------------- 요약


@st.cache_data(show_spinner=False)
def load_summary(dataset_id: str) -> dict:
    return api().summary(dataset_id)


def show_summary(dataset: dict) -> None:
    data = load_summary(dataset["dataset_id"])
    summary, currency = data["summary"], data["currency"]

    st.caption(md(f"{dataset['filename']} · 기간 {dataset['period']} · {dataset['rows']:,}행"))
    cols = st.columns(4)
    cols[0].metric(f"총 매출 ({currency})", f"{summary['total_sales']:,}")
    cols[1].metric(f"총 이익 ({currency})", f"{summary['total_profit']:,}" if "total_profit" in summary else "—")
    cols[2].metric("주문 수", f"{summary['order_count']:,}")
    cols[3].metric("고객 수", f"{summary['customer_count']:,}")

    left, right = st.columns(2)
    months = pd.DataFrame(data["monthly_sales"]["months"]).set_index("month")
    left.line_chart(months["value"], y_label="월별 매출")
    categories = pd.DataFrame(data["by_category"]["rows"]).set_index("name")
    right.bar_chart(categories["sales"], horizontal=True, x_label="카테고리별 매출", y_label="")


# ---------------------------------------------------------------- 대화


def show_answer(answer: dict) -> None:
    """결론 → 발견(근거 숫자) → 참고 → 제안 액션 순서로 그린다."""
    st.markdown(f"**{md(answer['summary'])}**")

    for i, finding in enumerate(answer["findings"], 1):
        st.markdown(md(f"**{i}. {finding['title']}**  \n{finding['detail']}"))
        if finding["evidence"]:
            st.caption(md(" · ".join(finding["evidence"])))

    if answer["notes"]:
        st.caption(md("  \n".join(f"ℹ️ {note}" for note in answer["notes"])))

    if answer["suggested_actions"]:
        st.markdown("**제안 액션**")
        for action in answer["suggested_actions"]:
            badge = PRIORITY.get(action["priority"], action["priority"])
            st.markdown(md(f"- {badge} **{action['action']}**  \n  근거: {action['reason']}"))


def show_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.markdown(md(turn["question"]))
    with st.chat_message("assistant"):
        show_answer(turn)
        usage = turn["usage"]
        st.caption(
            f"{MODES.get(usage['mode'], usage['mode'])} · {usage['seconds']}초 · 도구 {len(turn['tools_used'])}개 · "
            f"토큰 입력 {usage['input_tokens']:,} / 출력 {usage['output_tokens']:,}"
        )
        if turn["tools_used"]:
            with st.expander("사용한 분석 도구"):
                for tool in turn["tools_used"]:
                    st.code(f"{tool['name']}({tool['arguments']})", language="json")


def describe_tool(name: str, arguments: str) -> str:
    """도구 호출을 사람이 읽기 쉽게: breakdown(group_by=category, region=Central). 값이 없는 인자는 뺀다."""
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return f"{name}({arguments})"
    parts = []
    for key, value in args.items():
        if key == "filters" and isinstance(value, dict):
            parts += [f"{k}={v}" for k, v in value.items() if v is not None]
        elif value is not None:
            parts.append(f"{key}={value}")
    return f"{name}({', '.join(parts)})"


def local_time(iso: str) -> str:
    # DB에는 UTC로 저장되어 있으므로, 화면에는 이 컴퓨터의 시간대로 보여준다.
    return datetime.fromisoformat(iso).astimezone().strftime("%m-%d %H:%M")


def pick_conversation(dataset: dict) -> None:
    """사이드바에서 이전 대화를 고르면 그 대화의 질문·답변을 불러온다."""
    conversations = api().list_conversations(dataset["dataset_id"])
    labels = {NEW_CONVERSATION: "＋ 새 대화"}
    for c in conversations:
        title = c["title"] if len(c["title"]) <= 20 else c["title"][:20] + "…"
        labels[c["conversation_id"]] = f"{local_time(c['updated_at'])} · {title} ({c['turn_count']})"

    current = st.session_state.conversation_id or NEW_CONVERSATION
    options = list(labels)
    picked = st.sidebar.selectbox(
        "대화", options, index=options.index(current) if current in options else 0, format_func=labels.get
    )
    if picked == current:
        return
    if picked == NEW_CONVERSATION:
        st.session_state.conversation_id = None
        st.session_state.turns = []
    else:
        st.session_state.conversation_id = picked
        st.session_state.turns = api().conversation(picked)["turns"]


def show_chat(dataset: dict) -> None:
    st.subheader("AI 분석가에게 질문하기")
    mode = st.sidebar.radio("답변 모드", list(MODES), format_func=MODES.get, index=1)
    pick_conversation(dataset)

    for turn in st.session_state.turns:
        show_turn(turn)

    question = st.chat_input("예: Furniture 이익률이 왜 낮아?")
    if not question:
        return
    with st.chat_message("user"):
        st.markdown(md(question))
    result = None
    # 답변을 기다리는 동안 Agent가 무엇을 하고 있는지 한 줄씩 보여준다.
    with st.status("질문을 이해하고 필요한 분석을 고르는 중...", expanded=True) as status:
        try:
            for event, data in api().chat_stream(dataset["dataset_id"], st.session_state.conversation_id, question, mode):
                if event == "llm_call" and data["step"] > 1:
                    status.update(label="분석 결과를 보고 판단하는 중...")
                elif event == "tool":
                    status.write(md(f"🔧 {describe_tool(data['name'], data['arguments'])}"))
                elif event == "skip":
                    status.write(md(f"⏭ {data['name']} 건너뜀 (이 모드의 도구 호출 상한)"))
                elif event == "done":
                    result = data
        except ApiError as e:
            status.update(label="답변을 만들지 못했습니다", state="error")
            st.error(str(e))
            return
        status.update(label=f"완료 · {result['usage']['seconds']}초", state="complete")
    st.session_state.conversation_id = result["conversation_id"]
    st.session_state.turns.append({"question": question, **result})
    # 사이드바의 대화 목록에도 방금 질문이 반영되도록 화면을 다시 그린다.
    st.rerun()


def main() -> None:
    st.title("📊 AI Data Analyst")
    st.caption("CSV 업로드 → 질문 → AI가 분석 도구를 골라 답변")
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("turns", [])

    try:
        dataset = select_dataset()
        if dataset is None:
            st.info("왼쪽에서 데이터를 선택하거나 CSV를 등록하세요.")
            return
        # 데이터셋이 바뀌면 이전 데이터셋의 대화를 이어 갈 수 없으므로 새 대화로 시작한다.
        if st.session_state.get("active_dataset_id") != dataset["dataset_id"]:
            st.session_state.active_dataset_id = dataset["dataset_id"]
            st.session_state.conversation_id = None
            st.session_state.turns = []
        show_summary(dataset)
    except ApiError as e:
        st.error(str(e))
        return
    show_chat(dataset)


main()
