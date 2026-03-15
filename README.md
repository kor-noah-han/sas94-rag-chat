# sas94-rag-chat

SAS 9.4 문서를 참고해서 답하는 사람용 chat 앱이다.

이 저장소는 `chat UI`, `chat API`, `Gemini 응답 생성`을 담당한다. 문서 검색 자체는 외부 패키지 [`sas-94-search-api`](https://github.com/kor-noah-han/sas-94-search-API)가 맡는다.

역할 분리는 이렇게 본다.

- `sas-94-search-api`
  - 검색 엔진
  - Python 패키지
  - Search API 서버
  - 검색 데이터 Release
- `sas94-rag-chat`
  - 사람용 CLI
  - 사람용 웹 UI
  - 사람용 chat API
  - Gemini 기반 답변 생성

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sas94-search-data --tag search-data-20260316-v0.1.5 --output-dir .
```

필수 환경 변수:

- `GEMINI_API_KEY` 또는 `GOOGLE_API_KEY`
- `GEMINI_MODEL`
- `GEMINI_API_BASE_URL`

기본 검색 모드는 `lexical`이라서 Qdrant 없이도 바로 실행할 수 있다. dense/hybrid를 쓰려면 Qdrant를 `server mode only`로 사용하며 기본 URL은 `http://localhost:6333` 이다.

## Run

CLI:

```bash
python3 scripts/chat_sas_rag.py
```

one-shot 질문:

```bash
python3 scripts/chat_sas_rag.py "SAS에서 라이브러리를 어떻게 할당하나요?" --show-sources
```

웹 UI / chat API:

```bash
python3 scripts/serve_sas_rag.py --port 8787
```

브라우저:

```text
http://127.0.0.1:8787
```

## API

- `GET /health`
- `GET /`
- `POST /api/chat`

예시:

```bash
curl -s -X POST http://127.0.0.1:8787/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"상관분석하는 법을 알려줘","mode":"lexical","top_k":3}'
```

## CLI Notes

interactive 명령:

- `/help`
- `/config`
- `/mode hybrid|lexical|dense`
- `/topk 3`
- `/sources on|off`
- `/debug on|off`
- `/context on|off`
- `/retrieval on|off`
- `/exit`

현재 사람용 chat 기본 검색 모드는 `lexical`이다. 속도와 안정성을 우선한 설정이다.

## Project Layout

```text
.
├── README.md
├── docker-compose.qdrant.yml
├── requirements.txt
├── docs/
│   └── project-structure.md
├── sas_rag/
│   ├── app.py
│   ├── generation.py
│   ├── logging_utils.py
│   ├── prompts.py
│   ├── retrieval.py
│   ├── search_package.py
│   ├── settings.py
│   └── service.py
├── scripts/
│   ├── app/
│   │   ├── chat_sas_rag.py
│   │   └── serve_sas_rag.py
│   ├── chat_sas_rag.py
│   └── serve_sas_rag.py
└── web/
    └── chat_ui.html
```

핵심 파일:

- [`sas_rag/search_package.py`](/Users/noahhan/dev/sas94-rag-chat/sas_rag/search_package.py#L1)
  - 외부 search 패키지 import 경계
- [`sas_rag/service.py`](/Users/noahhan/dev/sas94-rag-chat/sas_rag/service.py#L1)
  - search 결과를 chat 응답으로 조립
- [`sas_rag/generation.py`](/Users/noahhan/dev/sas94-rag-chat/sas_rag/generation.py#L1)
  - Gemini 호출
- [`sas_rag/prompts.py`](/Users/noahhan/dev/sas94-rag-chat/sas_rag/prompts.py#L1)
  - generation prompt 구성
- [`sas_rag/settings.py`](/Users/noahhan/dev/sas94-rag-chat/sas_rag/settings.py#L1)
  - Gemini 설정 로딩
- [`scripts/app/chat_sas_rag.py`](/Users/noahhan/dev/sas94-rag-chat/scripts/app/chat_sas_rag.py#L1)
  - CLI
- [`scripts/app/serve_sas_rag.py`](/Users/noahhan/dev/sas94-rag-chat/scripts/app/serve_sas_rag.py#L1)
  - 웹 UI와 chat API
- [`web/chat_ui.html`](/Users/noahhan/dev/sas94-rag-chat/web/chat_ui.html#L1)
  - 브라우저 UI

## Notes

- 이 repo는 검색 엔진 repo가 아니다.
- SAS Viya 전용 문서는 현재 포함하지 않는다.
- 대용량 원본 데이터, Qdrant 로컬 파일, PDF는 Git에 올리지 않는다.
