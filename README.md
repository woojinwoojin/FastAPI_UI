# 3주차 — FastAPI + 간단한 UI

## 목표
Agent를 API 서버로 분리하고 Streamlit UI를 붙여 실제 서비스의 요청/응답 구조를 만든다.

## 구조
```
Streamlit ──HTTP──► FastAPI ──► Agent (2주차) ──► Pandas
                       │
                       └──► SQLite (데이터셋 정보, 대화 기록) + CSV 파일
```

## 문제 정의 (2주차 결과에서)
2주차 Agent(`agent.py`)는 터미널에서만 동작한다. 서비스로 만들려면 아래가 바뀌어야 한다.

| 2주차 (CLI) | 3주차에 필요한 것 |
|---|---|
| 데이터는 코드에 고정된 Superstore 하나 | 사용자가 CSV를 올리고, 컬럼 매핑을 지정 |
| 대화 기록은 프로세스 메모리의 `history` 리스트 하나 | 여러 대화를 구분해서 보관 |
| 모드는 `--mode`, `/fast` 명령 | 요청마다 모드 지정 |
| 진행 상황은 stderr의 `[tool]` 로그 | 응답에 사용한 도구·시간·토큰을 담아 UI에 표시 |

## 설계 결정
| # | 항목 | 결정 | 이유 |
|---|---|---|---|
| 1 | 저장소 | 새 저장소 [FastAPI_UI](https://github.com/woojinwoojin/FastAPI_UI) | 주차별 저장소 규칙 |
| 2 | 2주차 코드 | **복사**해 온다. `agent.py`는 루프(코어)와 CLI를 분리 | 주차별 저장소라 패키지로 공유하는 수고에 비해 얻는 게 적다 |
| 3 | 대화 기록 | **SQLite** | 설치·서버 불필요, 재시작해도 남는다. "DB는 범위 밖"은 별도 DB 서버를 뜻하는 것으로 본다 |
| 4 | 데이터셋 | CSV 원본은 **파일**, SQLite에는 경로·컬럼 매핑·통화만 | 분석은 pandas로 파일을 읽어서 한다. DB에 CSV를 넣어도 얻는 게 없다 |
| 5 | 컬럼 매핑 | 1주차 Streamlit **수동 매핑 화면 재사용** | 이미 동작하는 코드. LLM 자동 매핑은 확장 과제 |
| 6 | 응답 방식 | 답변이 끝나면 **한 번에 응답**. 응답에 도구·시간·토큰 포함 | 먼저 요청/응답 구조를 완성. 스트리밍(SSE)은 확장 과제 |
| 7 | 엔드포인트 | `/chat` 중심. `/reports` 제외, `/analysis` → 데이터셋 요약 조회 | 2주차 Agent가 질문마다 필요한 분석을 고르므로 고정 보고서 API가 필요 없다 |

- 대화 기록에 섞인 OpenAI 응답 객체는 `model_dump()`로 dict로 바꿔 JSON 문자열로 저장한다. 다시 불러와 API에 보낼 수 있는지는 구현하며 확인한다.
- 4주차 Docker 배포 때 SQLite 파일과 업로드 CSV가 컨테이너와 함께 사라지지 않도록 volume이 필요하다.

## API
| 메서드 | 경로 | 요청 | 응답 |
|---|---|---|---|
| `POST` | `/datasets` | CSV 파일 + 컬럼 매핑·인코딩·날짜 형식·통화 (multipart) | `dataset_id`, 기간, 행 수 |
| `GET` | `/datasets/{id}/summary` | — | KPI 카드·차트용 요약 (`get_summary`, `breakdown`, `monthly_trend` 결과) |
| `POST` | `/chat` | `dataset_id`, `conversation_id`(없으면 새 대화), `question`, `mode` | `conversation_id`, `answer`, `suggested_actions`, `tools_used`, `usage`(시간·토큰) |
| `GET` | `/conversations/{id}` | — | 지난 질문과 답변 목록 (화면 복원용) |

## SQLite 테이블 (초안)
```
datasets       id, filename, path, column_map(JSON), currency, encoding, date_format, created_at
conversations  id, dataset_id, created_at
turns          id, conversation_id, question, mode, answer(JSON), tools_used(JSON), usage(JSON), created_at
                 → 화면에 보여줄 질문·답변
history_items  id, conversation_id, seq, item(JSON)
                 → LLM에 다시 보낼 대화 기록 (도구 요청·결과 포함). 순서를 seq로 보장
```
`turns`와 `history_items`를 나누는 이유: 화면에는 질문·답변만 필요하지만, LLM에는 도구 요청·결과까지 포함한 전체 기록을 보내야 한다.

## 함께 익힐 개념
- **HTTP / REST:** 메서드(GET·POST), 경로, 상태 코드(200, 404, 422), JSON 본문
- **FastAPI + Pydantic:** 요청·응답 모델로 형식 검증 (잘못된 요청 → 자동 422)
- **파일 업로드:** `multipart/form-data`, `UploadFile`
- **sync vs async:** `run_agent`와 OpenAI 호출은 동기(blocking) 코드다. `async def` 엔드포인트에서 그대로 부르면 서버 전체가 멈춘다.
  → 엔드포인트를 일반 `def`로 두면 FastAPI가 스레드 풀에서 실행한다.
- **상태(state)를 어디에 두는가:** API 서버는 요청 사이에 아무것도 기억하지 않는다고 가정하고, 상태는 SQLite에 둔다.
- **SQLite:** `sqlite3` 표준 라이브러리, 테이블·기본 키·외래 키, JSON 문자열 컬럼

## 구현 순서 (안)
1. 2주차 코드 복사 + `agent.py` 코어/CLI 분리 (기존 테스트 통과 확인)
2. SQLite 저장소 모듈 (`db.py`) + 테스트
3. FastAPI: `/datasets` → `/chat` → `/conversations`, `/datasets/{id}/summary` (TestClient로 테스트, API 호출은 가짜 클라이언트)
4. Streamlit: 업로드·매핑 화면(1주차 재사용) → 채팅 화면 (`conversation_id`는 `session_state`)
5. 실제 API로 전체 흐름 확인
6. (확장) 스트리밍, LLM 자동 컬럼 매핑

## 완료 체크
- [ ] FastAPI 엔드포인트 + Pydantic 모델
- [ ] Streamlit 업로드/채팅 화면
- [ ] UI ↔ API 연동

## 블로그 주제
Streamlit 스크립트를 FastAPI 서비스로 분리하기 (1주차 블로그 메모의 시리즈 계획)
