"""터미널에서 Agent와 대화하는 CLI (2주차의 agent.py main).

실행: .\\.venv\\Scripts\\python cli.py [--mode fast|standard|careful]
"""

import argparse
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from agent import DEFAULT_MODE, DEFAULT_MODEL, MODES, answer_to_text, build_system_prompt, run_agent
from analysis import SUPERSTORE, load_data

logger = logging.getLogger("agent")


def print_answer(answer: dict) -> None:
    print("\n" + answer_to_text(answer) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV 분석 에이전트")
    parser.add_argument("--mode", choices=list(MODES), default=DEFAULT_MODE, help="fast: 빠른 답변(약 4초), standard: 기본(약 8초), careful: 신중(10~20초)")
    mode = MODES[parser.parse_args().mode]

    # 진행 상황([tool], [usage])은 답변(stdout)과 섞이지 않도록 stderr로 출력한다.
    # 루트 로거는 WARNING으로 두어 httpx 같은 라이브러리의 요청 로그는 숨긴다.
    logging.basicConfig(format="%(message)s")
    logging.getLogger("agent").setLevel(logging.INFO)

    load_dotenv()
    client = OpenAI()  # .env의 OPENAI_API_KEY를 사용한다
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    df = load_data(SUPERSTORE)
    instructions = build_system_prompt(df, SUPERSTORE.currency)

    # 후속 질문("그럼 Tables는?")이 앞 대화를 참고할 수 있도록 기록을 이어서 쓴다.
    history: list = []
    print("질문을 입력하세요. 모드 전환: /fast, /standard, /careful (종료: 빈 줄 또는 Ctrl+C)")
    while True:
        try:
            question = input(f"[{mode.name}] 질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        if question.startswith("/") and question[1:] in MODES:
            mode = MODES[question[1:]]
            print(f"{mode.name} 모드로 바꿨습니다.")
            continue
        checkpoint = len(history)
        history.append({"role": "user", "content": question})
        try:
            print_answer(run_agent(client, model, df, instructions, history, mode).answer)
        except Exception as e:  # API 키 누락, 크레딧 부족 등 — 대화는 계속한다
            logger.error("[error] %s", e)
            del history[checkpoint:]  # 실패한 질문의 기록은 지워서 다음 질문에 섞이지 않게 한다


if __name__ == "__main__":
    main()
