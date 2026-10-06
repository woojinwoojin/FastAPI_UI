# 3주차 — FastAPI + 간단한 UI

## 목표
Agent를 API 서버로 분리하고 Streamlit UI를 붙여 실제 서비스의 요청/응답 구조를 만든다.

## 구조
```
Streamlit → FastAPI → AI Agent → Pandas
```

## 문제 정의 (2주차 결과에서)
2주차 Agent(`agent.py`)는 터미널에서만 동작한다. 서비스로 만들려면 아래가 바뀌어야 한다.

| 2주차 (CLI) | 3주차에 필요한 것 |
|---|---|
| 데이터는 코드에 고정된 Superstore 하나 | 사용자가 CSV를 올리고, 컬럼 매핑을 지정 |
| 대화 기록은 프로세스 메모리의 `history` 리스트 하나 | 여러 사용자·여러 대화를 구분해서 보관 |
| 모드는 `--mode`, `/fast` 명령 | 요청마다 모드 지정 |
| 진행 상황은 stderr의 `[tool]` 로그 | 응답에 사용한 도구·시간·토큰을 담아 UI에 표시 |
| 질문 1개에 4~20초 동안 터미널이 기다림 | HTTP 요청이 그동안 열려 있음 → 사용자에게 진행 상황을 보여줄지 |

## API 초안 (2주차 결과 반영 전)
- `POST /datasets` — CSV 업로드
- `POST /analysis` — 기본 KPI 분석
- `POST /chat` — 질문 → Agent 응답
- `GET /reports` — 보고서 조회

## 정해야 할 설계 (프로젝트 데이에 함께 결정)
1. ~~저장소~~ → 새 저장소로 결정: https://github.com/woojinwoojin/FastAPI_UI
2. **2주차 코드 재사용:** `tools.py`, `agent.py`를 복사해 올지, 패키지로 가져다 쓸지
3. **대화 기록 보관 위치**
   - (a) 서버 메모리에 `conversation_id`별로 보관 — 가장 단순, 서버를 재시작하면 사라짐
   - (b) 클라이언트가 매번 전체 기록을 전송 — 서버는 상태가 없음. 단, 기록에 OpenAI 응답 객체(추론 항목 등)가 섞여 있어 그대로 JSON으로 주고받기 어렵다
   - (c) OpenAI의 `previous_response_id`로 기록을 OpenAI 쪽에 맡김 — 코드는 줄지만 특정 제공자에 묶임
   - → **(d) SQLite로 결정.** 파일 하나짜리 DB라 설치·서버가 필요 없고, 서버를 재시작해도 기록이 남는다.
     (월 계획의 "DB는 범위 밖"은 PostgreSQL 같은 별도 DB 서버를 뜻하는 것으로 보고, SQLite는 허용)
     - 기록에 섞인 OpenAI 응답 객체는 `model_dump()`로 dict로 바꿔 JSON 문자열로 저장한다
     - 4주차 Docker 배포 때 DB 파일이 컨테이너와 함께 사라지지 않도록 volume이 필요하다
4. **데이터셋 보관** → **결정:** CSV 원본은 파일로 저장하고, SQLite에는 데이터셋 정보(파일 경로, 컬럼 매핑, 통화)만 넣는다
   - CSV를 DB에 통째로 넣으면 DB만 커지고 얻는 게 없다. 분석은 어차피 pandas로 파일을 읽어서 한다
5. **컬럼 매핑:** 1주차 Streamlit의 수동 매핑 화면을 재사용할지, LLM이 추론하고 사용자가 확인하게 할지
6. **응답 방식:** 답변이 끝날 때 한 번에 응답할지, 도구 호출 진행 상황을 스트리밍(SSE)으로 보낼지
7. **엔드포인트 정리:** `/analysis`, `/reports`가 2주차 Agent 구조에서도 필요한지 (`/chat` 하나로 충분할 수 있음)

## 함께 익힐 CS 개념
HTTP, REST API, request/response, JSON, async
- 2주차와 연결: 동기 함수(`run_agent`)를 async 서버에서 부를 때 생기는 문제, 상태(state)를 어디에 두는가

## 완료 체크
- [ ] FastAPI 엔드포인트 + Pydantic 모델
- [ ] Streamlit 업로드/채팅 화면
- [ ] UI ↔ API 연동

## 블로그 주제
Streamlit 스크립트를 FastAPI 서비스로 분리하기 (1주차 블로그 메모의 시리즈 계획)
