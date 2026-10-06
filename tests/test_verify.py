import json

from verify import numbers_in, unverified_evidence


def tool_output(result: dict) -> dict:
    return {"type": "function_call_output", "call_id": "c", "output": json.dumps(result, ensure_ascii=False)}


HISTORY = [
    {"role": "user", "content": "왜 떨어졌어?"},
    tool_output({"rows": [{"name": "Technology", "base_sales": 49919, "compare_sales": 21985, "sales_change": -27934,
                           "sales_change_pct": -56.0, "share_of_total_change_pct": 80.7}]}),
    tool_output({"bands": [{"discount_band": "21~40%", "profit_margin_pct": -18.3}]}),
]


def answer(*evidence: str) -> dict:
    return {"findings": [{"title": "t", "detail": "d", "evidence": list(evidence)}]}


def test_numbers_are_normalized():
    assert numbers_in("매출 49,919달러→21,985달러, -56.0%, 2.50배") == {"49919", "21985", "56", "2.5"}


def test_evidence_from_tool_results_passes():
    result = unverified_evidence(
        answer("Technology 49,919달러→21,985달러", "감소 27,934달러 (-56.0%)", "감소분의 80.7%", "21~40% 이익률 -18.3%"),
        HISTORY,
    )

    assert result == []


def test_numbers_not_in_tool_results_are_flagged():
    result = unverified_evidence(answer("Technology 감소 2.8만 달러", "이익률 -18.4%", "감소분의 80.7%"), HISTORY)

    # 2.8만은 LLM이 직접 환산한 값, -18.4%는 잘못 옮긴 값
    assert result == ["Technology 감소 2.8만 달러", "이익률 -18.4%"]


def test_small_integers_are_not_checked():
    # 월(12), 순위(1위), 개수(3개)는 도구 결과에 없어도 문제 삼지 않는다.
    assert unverified_evidence(answer("12월 1위, 하위 카테고리 3개"), HISTORY) == []


def test_numbers_from_earlier_questions_count():
    history = HISTORY + [{"role": "user", "content": "그럼 East는?"}]  # 이번 질문에서는 도구를 부르지 않은 경우

    assert unverified_evidence(answer("Technology 21,985달러"), history) == []


def test_answer_without_findings():
    assert unverified_evidence({"findings": []}, HISTORY) == []
