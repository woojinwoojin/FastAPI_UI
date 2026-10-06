"""CSV 컬럼 이름과 샘플 몇 행을 보고 LLM이 컬럼 매핑을 추천한다.

- LLM에는 원본 전체가 아니라 컬럼 이름과 샘플 행만 보낸다.
- 응답 스키마의 선택지를 실제 컬럼 이름(enum)으로 제한해, 없는 컬럼을 지어낼 수 없게 한다.
- 추천한 날짜 형식은 샘플로 실제로 읽어 보고, 실패하면 버린다.
- 추천일 뿐이다. 화면에서 사람이 확인한 뒤 등록한다.
"""

import json
from datetime import datetime

from openai import OpenAI

ROLES = {
    "date": "주문(거래) 날짜",
    "sales": "매출 금액",
    "order_id": "주문 ID. 한 주문이 여러 행일 수 있다",
    "customer_id": "고객 ID",
    "category": "상품 카테고리 (가장 큰 분류)",
    "profit": "이익 금액 (선택)",
    "sub_category": "하위 카테고리 (선택)",
    "discount": "할인율 (선택)",
    "region": "지역 (선택)",
    "segment": "고객 세그먼트 (선택)",
}
SAMPLE_ROWS = 5

PROMPT = """CSV 파일의 컬럼 이름과 샘플 행을 보고, 각 역할에 해당하는 컬럼을 고르세요.
- 해당하는 컬럼이 없으면 null로 두세요. 비슷해 보인다는 이유만으로 억지로 고르지 마세요.
- 같은 컬럼을 두 역할에 쓰지 마세요.
- date_format은 날짜 컬럼 값을 읽을 Python strptime 형식입니다 (예: %m/%d/%Y). 모르겠으면 null로 두세요.

역할:
{roles}"""


def mapping_format(columns: list[str]) -> dict:
    column_or_null = {"anyOf": [{"type": "string", "enum": columns}, {"type": "null"}]}
    properties = {role: column_or_null for role in ROLES}
    properties["date_format"] = {"type": ["string", "null"]}
    return {
        "type": "json_schema",
        "name": "column_mapping",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


def check_date_format(date_format: str | None, values: list[str]) -> str | None:
    """샘플 날짜를 모두 읽을 수 있는 형식이면 그대로, 아니면 None."""
    if not date_format or not values:
        return None
    try:
        for value in values:
            datetime.strptime(value, date_format)
    except ValueError:
        return None
    return date_format


def suggest_mapping(client: OpenAI, model: str, columns: list[str], samples: list[dict]) -> dict:
    """{"column_map": {역할: 컬럼}, "date_format": str | None}. 추천하지 못한 역할은 빠진다."""
    response = client.responses.create(
        model=model,
        instructions=PROMPT.format(roles="\n".join(f"- {role}: {desc}" for role, desc in ROLES.items())),
        input=json.dumps({"columns": columns, "sample_rows": samples}, ensure_ascii=False, default=str),
        text={"format": mapping_format(columns)},
        reasoning={"effort": "low"},
    )
    suggestion = json.loads(response.output_text)

    column_map, used = {}, set()
    for role in ROLES:
        column = suggestion.get(role)
        # strict 스키마가 막아 주지만, 한 번 더 확인한다: 실제 컬럼이고 다른 역할에 쓰지 않은 것만.
        if column in columns and column not in used:
            column_map[role] = column
            used.add(column)

    date_values = [str(row[column_map["date"]]) for row in samples] if "date" in column_map else []
    return {"column_map": column_map, "date_format": check_date_format(suggestion.get("date_format"), date_values)}
