"""Streamlit 화면. 분석·Agent는 모두 API 서버가 하고, 이 화면은 API를 부르고 결과를 보여주기만 한다.

실행 (API 서버를 먼저 켠다):
  .\\.venv\\Scripts\\uvicorn api:app --reload
  .\\.venv\\Scripts\\streamlit run app.py
"""

import io

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


def api() -> ApiClient:
    # make_http를 모듈 속성으로 찾아야 테스트에서 바꿔 끼울 수 있다.
    return ApiClient(ui_client.make_http())


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

    st.caption(f"{dataset['filename']} · 기간 {dataset['period']} · {dataset['rows']:,}행")
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


def show_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        if turn["suggested_actions"]:
            st.markdown("**제안 액션**")
            for action in turn["suggested_actions"]:
                st.markdown(f"- **[{action['priority']}]** {action['action']}  \n  근거: {action['reason']}")
        usage = turn["usage"]
        st.caption(
            f"{MODES.get(usage['mode'], usage['mode'])} · {usage['seconds']}초 · 도구 {len(turn['tools_used'])}개 · "
            f"토큰 입력 {usage['input_tokens']:,} / 출력 {usage['output_tokens']:,}"
        )
        if turn["tools_used"]:
            with st.expander("사용한 분석 도구"):
                for tool in turn["tools_used"]:
                    st.code(f"{tool['name']}({tool['arguments']})", language="json")


def show_chat(dataset: dict) -> None:
    st.subheader("AI 분석가에게 질문하기")
    mode = st.sidebar.radio("답변 모드", list(MODES), format_func=MODES.get, index=1)
    if st.sidebar.button("새 대화"):
        st.session_state.conversation_id = None
        st.session_state.turns = []

    for turn in st.session_state.turns:
        show_turn(turn)

    question = st.chat_input("예: Furniture 이익률이 왜 낮아?")
    if not question:
        return
    with st.spinner("분석하는 중..."):
        try:
            result = api().chat(dataset["dataset_id"], st.session_state.conversation_id, question, mode)
        except ApiError as e:
            st.error(str(e))
            return
    st.session_state.conversation_id = result["conversation_id"]
    turn = {"question": question, **result}
    st.session_state.turns.append(turn)
    show_turn(turn)


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
