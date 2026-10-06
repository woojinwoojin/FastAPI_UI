"""답변의 근거 숫자가 실제 도구 결과에 있는지 대조한다.

LLM에게 "숫자는 도구 결과에서만 인용하라"고 해도 지키는지 확인할 방법이 없었다.
답변을 구조화하면서 근거 숫자가 findings[].evidence에 따로 모이므로, 이제 코드로 대조할 수 있다.

- 부호는 무시한다. "감소 27,934달러"와 도구 결과 -27934는 같은 숫자다.
- 12 이하의 정수는 검사하지 않는다. 월, 순위, 개수처럼 도구 결과에 없어도 자연스러운 숫자라 오탐이 많다.
- LLM이 직접 계산하거나 단위를 바꾼 숫자("74.2만 달러")는 도구 결과에 그대로 없으므로 걸린다. 의도한 동작이다.
"""

import re

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
MAX_IGNORED_INT = 12


def numbers_in(text: str) -> set[str]:
    """텍스트 속 숫자를 비교하기 쉬운 문자열로 모은다: 쉼표 제거, 부호 무시, 소수는 첫째 자리까지."""
    found = set()
    for match in NUMBER.findall(text):
        value = float(match.replace(",", ""))
        found.add(str(int(value)) if value.is_integer() else f"{round(value, 1):g}")
    return found


def tool_numbers(history: list[dict]) -> set[str]:
    """대화 기록에 있는 모든 도구 결과의 숫자. 후속 질문은 앞 질문의 도구 결과를 인용할 수 있으므로 전체를 본다."""
    numbers: set[str] = set()
    for item in history:
        if item.get("type") == "function_call_output":
            numbers |= numbers_in(item["output"])
    return numbers


def needs_check(number: str) -> bool:
    return "." in number or int(number) > MAX_IGNORED_INT


def unverified_evidence(answer: dict, history: list[dict]) -> list[str]:
    """도구 결과에서 찾을 수 없는 숫자가 들어 있는 근거 문구 목록."""
    known = tool_numbers(history)
    unverified = []
    for finding in answer["findings"]:
        for evidence in finding["evidence"]:
            if any(needs_check(n) and n not in known for n in numbers_in(evidence)):
                unverified.append(evidence)
    return unverified
