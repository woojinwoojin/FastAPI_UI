"""질문 → LLM이 도구 선택 → Python 실행 → 결과 → LLM → 답변 + 제안 액션.

프레임워크 없이 OpenAI Responses API로 tool calling 루프를 직접 구현한 Agent 코어.
CLI(cli.py)와 API 서버가 함께 쓴다. 화면 출력이나 저장은 하지 않고, 결과를 AgentResult로 돌려준다.
"""

import json
import logging
import time
from dataclasses import dataclass

import pandas as pd
from openai import OpenAI

from tools import TOOLS, run_tool

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5.4-mini"


@dataclass(frozen=True)
class Mode:
    """답변 속도와 깊이를 정하는 설정. 응답 시간은 LLM 왕복 횟수, 추론 토큰, 출력 길이가 좌우한다."""

    name: str
    max_steps: int  # LLM 호출 횟수 상한. 마지막 차례에는 도구 없이 답하게 한다.
    max_tool_calls: int  # 한 질문에서 실행할 도구 호출 수 상한
    reasoning_effort: str
    guide: str  # 시스템 프롬프트 끝에 덧붙이는 모드별 지침


MODES = {
    "fast": Mode(
        name="fast",
        max_steps=2,  # 도구 1라운드 → 바로 답변
        max_tool_calls=3,  # 도구는 로컬에서 바로 실행되므로, 속도는 호출 수보다 LLM 왕복 횟수가 좌우한다
        reasoning_effort="none",
        guide="""[빠른 답변 모드]
- 질문에 직접 답하는 데 꼭 필요한 도구만, 최대 3개를 한 번에 동시에 호출하세요. 결과를 받은 뒤에는 도구를 더 호출할 수 없습니다.
- "왜"를 묻는 질문에는 원인을 나눠 보여주는 도구(breakdown, compare_periods, discount_impact)를 고르세요.
  breakdown과 compare_periods 결과에는 합계(total)가 들어 있으므로 get_summary를 따로 부를 필요가 없습니다.
- answer는 3~5문장으로, 결론과 핵심 숫자만 쓰세요.
- 원인을 더 좁혀야 하는 질문이면 answer 끝에 "기본/신중 모드에서 더 자세히 확인할 수 있습니다."라고 덧붙이세요.
- suggested_actions는 1개 이내로 제안하세요.""",
    ),
    # experiments/compare_modes.py의 "B. wide". 병렬 fast + 종합보다 빠르고, 답변은 careful과 비슷했다.
    "standard": Mode(
        name="standard",
        max_steps=2,  # 한 라운드에 넓게 조회 → 바로 답변
        max_tool_calls=6,
        reasoning_effort="low",
        guide="""[기본 모드]
- 원인이 될 만한 여러 관점(부문, 할인, 지역, 기간)을 고려해, 필요한 도구를 최대 6개까지 한 번에 동시에 호출하세요.
  결과를 받은 뒤에는 도구를 더 호출할 수 없습니다.
- breakdown과 compare_periods 결과에는 합계(total)가 들어 있으므로 get_summary를 따로 부를 필요가 없습니다.
- "지난달", "요즘"처럼 기간이 모호하면 어떤 기간으로 해석했는지 answer에 밝히세요.
- answer는 5~8문장으로, 결론과 핵심 숫자만 쓰세요.
- suggested_actions는 3개 이내로 제안하세요.""",
    ),
    # 결과를 보고 다음 조회를 정해 한 단계씩 파고든다 (예: Technology 감소 → 그 안의 하위 카테고리).
    "careful": Mode(
        name="careful",
        max_steps=6,
        max_tool_calls=10,
        reasoning_effort="medium",
        guide="""[신중 모드]
- 결론을 내기 전에 비교 기준을 확인하세요. 예: 한 지역의 이익률이 낮다면 다른 지역 또는 전체 평균과 비교하세요.
- 질문 범위와 관계없는 기준(질문에 없는 지역, 세그먼트 등)까지 둘러보지는 마세요.
- "지난달", "요즘"처럼 기간이 모호하면 어떤 기간으로 해석했는지 answer에 밝히세요.
- suggested_actions는 3개 이내로 제안하세요.""",
    ),
}
DEFAULT_MODE = "standard"

SYSTEM_PROMPT = """당신은 이커머스 회사의 시니어 데이터 분석가입니다.
사용자 질문에 답하기 위해 필요한 분석 도구를 골라 호출하고, 도구 결과만 근거로 한국어로 답하세요.

- 숫자는 도구 결과에서만 인용하세요. 필요한 숫자가 없으면 추측하지 말고 도구를 더 호출하세요.
- 원인을 묻는 질문은 전체 → 기간/카테고리 → 하위 카테고리/할인 순으로 좁혀 가며 확인하세요.
- 월별 변화는 MoM과 YoY를 함께 보세요. 매출은 계절성이 강하므로 MoM만으로 급락/급등이라고 단정하지 마세요.
- 데이터로 확인할 수 없는 원인은 가설로 표시하세요.
- 도구가 error를 돌려주면 available 값 등을 참고해 인자를 고쳐 다시 호출하세요.
- 금액에는 통화 단위를 붙이세요.
- suggested_actions는 데이터 근거가 있는 비즈니스 액션(가격, 할인, 상품 구성, 마케팅 등)만 우선순위 순으로 제안하세요.
  "추가 분석", "데이터 확인", "도구 추가" 같은 분석 작업은 액션이 아닙니다. 도구로 확인할 수 없는 한계는 answer에 쓰세요.
  근거가 되는 도구 결과가 없으면 suggested_actions는 빈 배열로 두세요.

데이터 정보:
{data_info}"""

# 최종 답변 형식 (Structured Output). strict 모드 규칙은 도구 스키마와 같다.
ANSWER_FORMAT = {
    "type": "json_schema",
    "name": "analysis_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "질문에 대한 분석 답변 (근거 숫자 포함)"},
            "suggested_actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "reason": {"type": "string", "description": "이 액션을 제안하는 데이터 근거"},
                        "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    },
                    "required": ["action", "reason", "priority"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["answer", "suggested_actions"],
        "additionalProperties": False,
    },
}


def build_system_prompt(df: pd.DataFrame, currency: str) -> str:
    # LLM이 올바른 인자(월, 필터 값)를 고를 수 있도록 데이터의 범위만 알려준다.
    # 하위 카테고리는 개수가 많아서 넣지 않는다. 틀리면 도구의 error에 available 목록이 온다.
    data_info = {
        "period": f"{df['date'].min():%Y-%m} ~ {df['date'].max():%Y-%m}",
        "currency": currency,
        **{f"{col}_values": sorted(df[col].unique()) for col in ("category", "region", "segment") if col in df},
    }
    return SYSTEM_PROMPT.format(data_info=json.dumps(data_info, ensure_ascii=False))


@dataclass
class AgentResult:
    answer: dict  # ANSWER_FORMAT 형식: {"answer", "suggested_actions"}
    tools_used: list[dict]  # 실제로 실행한 도구: [{"name", "arguments"}] (상한을 넘어 건너뛴 호출은 빠진다)
    usage: dict  # {"mode", "steps", "input_tokens", "output_tokens", "seconds"}


def run_agent(
    client: OpenAI, model: str, df: pd.DataFrame, instructions: str, history: list, mode: Mode = MODES[DEFAULT_MODE]
) -> AgentResult:
    """history(대화 기록)에 질문이 들어 있는 상태로 호출한다. 도구 호출과 답변이 history에 추가된다.

    history의 항목은 모두 JSON으로 바꿀 수 있는 dict다. 그대로 저장했다가 다음 질문 때 다시 넘기면 된다.
    """
    input_tokens = output_tokens = 0
    tools_used: list[dict] = []
    started = time.perf_counter()

    for step in range(1, mode.max_steps + 1):
        response = client.responses.create(
            model=model,
            instructions=f"{instructions}\n\n{mode.guide}",
            input=history,
            tools=TOOLS,
            # 마지막 차례에는 도구를 못 쓰게 해서 지금까지의 결과로 답하게 한다.
            tool_choice="none" if step == mode.max_steps else "auto",
            text={"format": ANSWER_FORMAT},
            reasoning={"effort": mode.reasoning_effort},
        )
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens

        # API는 이전 요청을 기억하지 않으므로, 모델의 출력(도구 요청 포함)도 대화 기록에 남긴다.
        # 응답 객체 대신 dict로 바꿔 두면 DB에 JSON으로 저장했다가 그대로 다시 보낼 수 있다.
        history.extend(item.model_dump(exclude_none=True) for item in response.output)

        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            usage = {
                "mode": mode.name,
                "steps": step,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "seconds": round(time.perf_counter() - started, 1),
            }
            logger.info("[usage] %s tools=%d %s", model, len(tools_used), usage)
            if response.status != "completed":
                raise RuntimeError(f"응답이 완료되지 않았습니다: {response.incomplete_details}")
            return AgentResult(json.loads(response.output_text), tools_used, usage)

        for call in calls:
            # 모든 function_call에는 짝이 되는 output이 있어야 하므로, 상한을 넘은 호출에도 결과 대신 error를 돌려준다.
            if len(tools_used) >= mode.max_tool_calls:
                logger.info("[skip] %s (도구 호출 상한 %d개)", call.name, mode.max_tool_calls)
                output = json.dumps(
                    {"error": f"{mode.name} 모드의 도구 호출 상한({mode.max_tool_calls}개)을 넘었습니다. 지금까지의 결과로 답하세요."},
                    ensure_ascii=False,
                )
            else:
                logger.info("[tool] %s(%s)", call.name, call.arguments)
                output = run_tool(df, call.name, call.arguments)
                tools_used.append({"name": call.name, "arguments": call.arguments})
            history.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,  # 어떤 요청에 대한 결과인지 짝을 맞춘다
                    "output": output,
                }
            )

    raise RuntimeError(f"{mode.max_steps}번 안에 답변을 받지 못했습니다.")
