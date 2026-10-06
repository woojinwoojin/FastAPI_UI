# 3주차 — FastAPI + 간단한 UI

## 목표
Agent를 API 서버로 분리하고 Streamlit UI를 붙여 실제 서비스의 요청/응답 구조를 만든다.
결과와 발견은 [REPORT.md](REPORT.md)에 정리했다.

## 구조
```
Streamlit ──HTTP──► FastAPI ──► Agent (2주차) ──► Pandas
 (app.py)  (ui_client)  (api.py)   (agent.py, tools.py)
                          │
                          ├──► SQLite (데이터셋 정보, 대화 기록) + 업로드 CSV 파일   (db.py)
                          ├──► 근거 숫자 검증                                        (verify.py)
                          └──► LLM 컬럼 매핑 추천                                    (mapping.py)
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
| 2 | 2주차 코드 | **복사**해 온다. `agent.py`는 루프(코어)와 CLI(`cli.py`)를 분리 | 주차별 저장소라 패키지로 공유하는 수고에 비해 얻는 게 적다 |
| 3 | 대화 기록 | **SQLite** | 설치·서버 불필요, 재시작해도 남는다. "DB는 범위 밖"은 별도 DB 서버를 뜻하는 것으로 본다 |
| 4 | 데이터셋 | CSV 원본은 **파일**, SQLite에는 경로·컬럼 매핑·통화만 | 분석은 pandas로 파일을 읽어서 한다. DB에 CSV를 넣어도 얻는 게 없다 |
| 5 | 같은 업로드 | 파일 내용 + 읽기 설정이 같으면 **기존 데이터셋 재사용** (해시) | 새로고침해도 데이터셋 id가 같아야 이전 대화를 이어 갈 수 있다 |
| 6 | 컬럼 매핑 | 1주차 **수동 매핑 화면** + **AI 추천 버튼** | 사람이 최종 확인한다. AI에는 컬럼 이름과 샘플 5행만 보낸다 |
| 7 | 응답 방식 | `/chat` (한 번에) + `/chat/stream` (**진행 상황** SSE) | 기다리는 시간 대부분이 LLM 왕복·도구 실행이라 진행 상황만 보여도 체감이 크게 바뀐다 |
| 8 | 답변 형식 | `summary` / `findings`(근거 숫자) / `notes` / `suggested_actions` | 한 문자열이면 결론·숫자·해석·한계가 섞여 읽기 어렵다 |
| 9 | 숫자 검증 | 근거 숫자를 도구 결과와 대조해 없으면 ⚠️ | "도구 결과에서만 인용하라"는 지시를 지키는지 확인할 방법이 없었다 |
| 10 | 엔드포인트 | `/chat` 중심. `/reports` 제외, `/analysis` → 데이터셋 요약 조회 | Agent가 질문마다 필요한 분석을 고르므로 고정 보고서 API가 필요 없다 |

- 엔드포인트는 모두 일반 `def`다. Agent와 OpenAI 호출은 응답을 기다리는 동안 멈추는(blocking) 코드라, `async def` 안에서 부르면 서버 전체가 멈춘다.
- 대화 기록에 섞인 OpenAI 응답 객체(추론 항목 포함)는 `model_dump()`로 dict로 바꿔 JSON으로 저장한다. 실제 API로 다시 보내도 받아 주는 것을 확인했다.
- 4주차 Docker 배포 때 SQLite 파일과 업로드 CSV가 컨테이너와 함께 사라지지 않도록 volume이 필요하다.

## API
| 메서드 | 경로 | 요청 | 응답 |
|---|---|---|---|
| `POST` | `/datasets/suggest-mapping` | CSV 파일, 인코딩 (multipart) | `columns`, `column_map`(추천), `date_format`(샘플로 검증) |
| `POST` | `/datasets` | CSV 파일 + 컬럼 매핑·인코딩·날짜 형식·통화 (multipart) | `dataset_id`, 기간, 행 수, `reused` |
| `GET` | `/datasets/{id}/summary` | — | KPI 카드·차트용 요약 (`get_summary`, `breakdown`, `monthly_trend` 결과) |
| `GET` | `/datasets/{id}/conversations` | — | 이전 대화 목록 (최근 질문 순, 첫 질문을 제목으로) |
| `POST` | `/chat` | `dataset_id`, `conversation_id`(없으면 새 대화), `question`, `mode` | `conversation_id`, `summary`, `findings`, `notes`, `suggested_actions`, `unverified_evidence`, `tools_used`, `usage` |
| `POST` | `/chat/stream` | `/chat`과 같음 | SSE: `llm_call`, `tool`, `skip` (진행 상황) → `done` (`/chat` 응답과 같은 본문) 또는 `error` |
| `GET` | `/conversations/{id}` | — | 지난 질문과 답변 목록 (화면 복원용) |

- `/chat/stream`은 답변 글자가 아니라 **진행 상황**을 스트리밍한다. 답변은 JSON(Structured Output)이라 글자 단위로 흘려보내면 반쯤 만든 JSON을 화면에서 해석해야 한다.
  스트리밍이 시작되면 상태 코드는 이미 200이라, 도중의 실패는 `error` 이벤트로 알린다.
- LLM 호출이 실패하면 아무것도 저장하지 않는다. 새 대화는 답변이 나온 뒤에 만든다.

## SQLite 테이블
```
datasets       id, filename, path, column_map(JSON), currency, encoding, date_format, created_at, content_hash
conversations  id, dataset_id, created_at
turns          id, conversation_id, question, mode, answer(JSON), tools_used(JSON), usage(JSON), created_at
                 → 화면에 보여줄 질문·답변 (근거 검증 결과 포함)
history_items  id, conversation_id, seq, item(JSON)
                 → LLM에 다시 보낼 대화 기록 (도구 요청·결과 포함). 순서를 seq로 보장
```
- `turns`와 `history_items`를 나누는 이유: 화면에는 질문·답변만 필요하지만, LLM에는 도구 요청·결과까지 포함한 전체 기록을 보내야 한다.
- `content_hash`는 나중에 추가한 컬럼이라, 서버 시작 시 없으면 `ALTER TABLE`로 추가한다.

## 함께 익힌 개념
- **HTTP / REST:** 메서드(GET·POST), 경로, 상태 코드(200, 400, 404, 422, 502), JSON 본문
- **FastAPI + Pydantic:** 요청·응답 모델로 형식 검증 (잘못된 요청 → 자동 422), `Depends`로 DB 연결·OpenAI 클라이언트 주입
- **파일 업로드:** `multipart/form-data`, `UploadFile`
- **sync vs async:** blocking 코드는 일반 `def` 엔드포인트로 → FastAPI가 스레드 풀에서 실행
- **스트리밍(SSE):** `StreamingResponse`, 작업 스레드 + 큐, 스트리밍 도중의 오류는 상태 코드가 아니라 이벤트로
- **상태(state)를 어디에 두는가:** API 서버는 요청 사이에 아무것도 기억하지 않는다고 가정하고, 상태는 SQLite에 둔다
- **SQLite:** 외래 키(기본으로 꺼져 있음), 트랜잭션, 스레드마다 별도 연결, 간단한 마이그레이션

## 구현 순서
1. ✅ 2주차 코드 복사 + `agent.py` 코어/CLI 분리 (CLI는 `cli.py`)
2. ✅ SQLite 저장소 모듈 (`db.py`) + 테스트
3. ✅ FastAPI: `/datasets` → `/chat` → `/conversations`, `/datasets/{id}/summary`
4. ✅ Streamlit: 업로드·매핑 화면(1주차 재사용) → 채팅 화면
5. ✅ 실제 API로 전체 흐름 확인
6. ✅ 확장: 진행 상황 스트리밍, 이전 대화 불러오기, 답변 구조화, 근거 숫자 검증, LLM 컬럼 매핑 추천

## 실행 방법
```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env   # 그다음 .env에 OPENAI_API_KEY 입력

.\.venv\Scripts\uvicorn api:app --reload   # API 서버 → http://localhost:8000/docs 에서 직접 호출해 볼 수 있다
.\.venv\Scripts\streamlit run app.py       # 화면 → http://localhost:8501 (API 서버를 먼저 켠다. 주소는 API_URL 환경 변수)
.\.venv\Scripts\python cli.py              # 2주차 터미널 대화 (Superstore 고정)
.\.venv\Scripts\python -m pytest           # 테스트 (OpenAI 호출 없음)
.\.venv\Scripts\python experiments\answer_format.py   # 답변 형식 비교 실험 (API 호출 약 12번)
```
- SQLite 파일(`app.db`)과 업로드한 CSV는 `data/`에 저장된다. 위치는 `APP_DATA_DIR` 환경 변수로 바꿀 수 있다.
- Windows에서 `--reload`가 코드 변경을 감지하고도 새 서버를 띄우지 못한 적이 있다. 응답이 예전 코드 그대로면 서버를 다시 켠다.

## 완료 체크
- [x] FastAPI 엔드포인트 + Pydantic 모델
- [x] Streamlit 업로드/채팅 화면
- [x] UI ↔ API 연동 (TestClient로 화면 → API → Agent 흐름 테스트)

## 블로그 주제
Streamlit 스크립트를 FastAPI 서비스로 분리하기 (1주차 블로그 메모의 시리즈 계획)
