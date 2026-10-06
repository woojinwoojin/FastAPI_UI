"""답변 형식(한 문자열 vs 구조화)이 응답 시간·출력 토큰에 주는 영향과, 근거 숫자 검증 결과를 측정한다.

같은 standard 모드, 같은 질문으로 두 형식을 번갈아 실행한다. 이전 형식은 구조화 전의 스키마와 지침을 재현한다.
실행: .\\.venv\\Scripts\\python experiments\\answer_format.py
결과: experiments/answer_format_results.md
"""

import os
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

import agent  # noqa: E402
from analysis import SUPERSTORE, load_data  # noqa: E402
from verify import unverified_evidence  # noqa: E402

QUESTIONS = ["Furniture 이익률이 왜 낮아?", "지난달 매출이 왜 떨어졌어?", "Central 지역 이익률이 왜 낮아?"]
REPEATS = 2

# 구조화 전 형식: answer 문자열 하나 + 제안 액션
OLD_FORMAT = {
    "type": "json_schema",
    "name": "analysis_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "질문에 대한 분석 답변 (근거 숫자 포함)"},
            "suggested_actions": agent.ANSWER_FORMAT["schema"]["properties"]["suggested_actions"],
        },
        "required": ["answer", "suggested_actions"],
        "additionalProperties": False,
    },
}
OLD_GUIDE_LINES = """- "지난달", "요즘"처럼 기간이 모호하면 어떤 기간으로 해석했는지 answer에 밝히세요.
- answer는 5~8문장으로, 결론과 핵심 숫자만 쓰세요."""


def old_setup(instructions: str) -> tuple[str, agent.Mode]:
    # 구조 설명 문단을 빼고, 모드 지침의 항목 수 제한을 예전 문장 수 제한으로 되돌린다.
    instructions = re.sub(r"- 답변은 정해진 구조로.*?(?=- suggested_actions는)", "", instructions, flags=re.S)
    instructions = instructions.replace("notes에 쓰세요", "answer에 쓰세요")
    standard = agent.MODES["standard"]
    guide = standard.guide.replace("- findings는 4개 이내로 쓰세요.", OLD_GUIDE_LINES)
    return instructions, replace(standard, guide=guide)


def run(client, model, df, instructions, mode, question, answer_format) -> dict:
    agent.ANSWER_FORMAT = answer_format  # run_agent가 모듈 전역을 읽으므로 실행마다 바꿔 끼운다
    history = [{"role": "user", "content": question}]
    started = time.perf_counter()
    result = agent.run_agent(client, model, df, instructions, history, mode)
    return {
        "seconds": time.perf_counter() - started,
        "output": result.usage["output_tokens"],
        "tools": len(result.tools_used),
        "answer": result.answer,
        "history": history,
    }


def main() -> None:
    load_dotenv()
    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", agent.DEFAULT_MODEL)
    df = load_data(SUPERSTORE)
    new_format = agent.ANSWER_FORMAT
    new_instructions = agent.build_system_prompt(df, SUPERSTORE.currency)
    old_instructions, old_mode = old_setup(new_instructions)
    assert "답변은 정해진 구조로" not in old_instructions and "5~8문장" in old_mode.guide

    rows, flagged = [], []
    for question in QUESTIONS:
        for name, instructions, mode, answer_format in [
            ("이전 (문자열)", old_instructions, old_mode, OLD_FORMAT),
            ("구조화", new_instructions, agent.MODES["standard"], new_format),
        ]:
            runs = [run(client, model, df, instructions, mode, question, answer_format) for _ in range(REPEATS)]
            avg = {k: sum(r[k] for r in runs) / REPEATS for k in ("seconds", "output", "tools")}
            times = ", ".join(f"{r['seconds']:.1f}" for r in runs)
            row = f"| {question} | {name} | {avg['seconds']:.1f}초 ({times}) | {avg['output']:,.0f} | {avg['tools']:.1f} |"
            print(row, flush=True)
            rows.append(row)
            if answer_format is new_format:
                for r in runs:
                    evidence = [e for f in r["answer"]["findings"] for e in f["evidence"]]
                    bad = unverified_evidence(r["answer"], r["history"])
                    flagged.append(f"| {question} | {len(evidence)} | {len(bad)} | {' / '.join(bad) or '—'} |")

    out = Path(__file__).parent / "answer_format_results.md"
    out.write_text(
        "\n".join(
            [
                f"# 답변 형식 비교 ({model}, standard 모드, 질문당 {REPEATS}회 평균)",
                "",
                "| 질문 | 형식 | 시간 (각 회) | 출력 토큰 | 도구 호출 |",
                "|---|---|---|---|---|",
                *rows,
                "",
                "## 근거 숫자 검증 (구조화 형식, 실행마다)",
                "",
                "| 질문 | 근거 수 | 걸린 근거 수 | 걸린 근거 |",
                "|---|---|---|---|",
                *flagged,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n".join(flagged))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
