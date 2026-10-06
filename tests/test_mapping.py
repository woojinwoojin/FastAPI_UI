import json

from mapping import check_date_format, mapping_format, suggest_mapping
from tests.fakes import FakeClient, FakeItem, response

COLUMNS = ["주문일", "주문번호", "고객번호", "분류", "매출액"]
SAMPLES = [{"주문일": "01/15/2024", "주문번호": "A1", "고객번호": "C1", "분류": "가구", "매출액": 100}]


def llm_returns(suggestion: dict) -> FakeClient:
    return FakeClient([response([FakeItem(type="message")], json.dumps(suggestion, ensure_ascii=False))])


def full_suggestion(**overrides) -> dict:
    base = {role: None for role in mapping_format(COLUMNS)["schema"]["properties"]}
    base.update(date="주문일", sales="매출액", order_id="주문번호", customer_id="고객번호", category="분류",
                date_format="%m/%d/%Y")
    return {**base, **overrides}


def test_schema_limits_choices_to_real_columns():
    schema = mapping_format(COLUMNS)["schema"]

    assert schema["properties"]["sales"]["anyOf"][0]["enum"] == COLUMNS
    assert schema["required"] == list(schema["properties"])  # strict 모드 규칙


def test_suggestion_is_returned_with_checked_date_format():
    client = llm_returns(full_suggestion())

    result = suggest_mapping(client, "m", COLUMNS, SAMPLES)

    assert result["column_map"] == {"date": "주문일", "sales": "매출액", "order_id": "주문번호",
                                    "customer_id": "고객번호", "category": "분류"}
    assert result["date_format"] == "%m/%d/%Y"
    # LLM에는 컬럼 이름과 샘플 행만 간다.
    assert json.loads(client.requests[0]["input"]) == {"columns": COLUMNS, "sample_rows": SAMPLES}


def test_wrong_date_format_is_dropped():
    result = suggest_mapping(llm_returns(full_suggestion(date_format="%Y-%m-%d")), "m", COLUMNS, SAMPLES)

    assert result["date_format"] is None


def test_unknown_or_duplicate_columns_are_dropped():
    suggestion = full_suggestion(profit="Profit", region="매출액")  # 없는 컬럼, 이미 sales에 쓴 컬럼

    result = suggest_mapping(llm_returns(suggestion), "m", COLUMNS, SAMPLES)

    assert "profit" not in result["column_map"]
    assert "region" not in result["column_map"]


def test_check_date_format():
    assert check_date_format("%m/%d/%Y", ["01/15/2024", "12/31/2023"]) == "%m/%d/%Y"
    assert check_date_format("%m/%d/%Y", ["2024-01-15"]) is None
    assert check_date_format(None, ["01/15/2024"]) is None
