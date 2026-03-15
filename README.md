# sas9-RAG

SAS 문서를 참고해서 답하는 사람용 챗 애플리케이션이다. 이 저장소는 `chat`과 `Gemini 응답 생성`을 담당하고, 문서 검색은 외부 패키지 [`sas-94-search-api`](https://github.com/kor-noah-han/sas-94-search-API)가 맡는다.

즉 구조는 이렇게 나뉜다.

- `sas-94-search-api`
  - 문서 검색 엔진
  - Python 패키지
  - Search API 서버
  - 검색 데이터 Release
- `sas9-RAG`
  - 사람용 chat CLI
  - 사람용 chat HTTP API
  - Gemini 기반 답변 생성

## 현재 역할

이 저장소는 이제 `사람용 chat 앱`이다.

- 질문을 받는다
- 외부 search 패키지로 관련 문서를 찾는다
- 검색된 문맥을 Gemini에 넘겨 답변을 만든다

검색 전용 기능은 이 저장소의 책임이 아니다.

## 폴더 구조

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
│   ├── search_package.py
│   └── service.py
├── scripts/
│   ├── app/
│   │   ├── chat_sas_rag.py
│   │   └── serve_sas_rag.py
│   ├── _bootstrap.py
│   ├── chat_sas_rag.py
│   └── serve_sas_rag.py
└── data/
    └── ...
```

`data/`, `scripts/ingest/`, `scripts/index/` 아래 파일들은 과거 문서 구축 파이프라인 산출물/유틸리티다. 현재 런타임 핵심은 `chat` 쪽이다.

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

이 requirements에는 외부 검색 패키지 `sas-94-search-api`가 포함되어 있다.

search 런타임 데이터는 GitHub Release에서 받는다.

```bash
sas94-search-data --output-dir .
```

Gemini 환경 변수도 필요하다.

- `GEMINI_API_KEY` 또는 `GOOGLE_API_KEY`
- `GEMINI_MODEL`
- `GEMINI_API_BASE_URL`

## 실행

Qdrant는 server mode 전제입니다. 기본값은 `http://localhost:6333` 입니다.

### CLI

```bash
python3 scripts/chat_sas_rag.py
```

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

one-shot 질문:

```bash
python3 scripts/chat_sas_rag.py "SAS에서 라이브러리를 어떻게 할당하나요?" --show-sources
```

검색 결과만 확인:

```bash
python3 scripts/chat_sas_rag.py "SAS에서 라이브러리를 어떻게 할당하나요?" --retrieval-only
```

### HTTP API

```bash
python3 scripts/serve_sas_rag.py --port 8787
```

- `GET /health`
- `POST /api/chat`
- `GET /`

호출 예시:

```bash
curl -s -X POST http://127.0.0.1:8787/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"SAS에서 라이브러리를 어떻게 할당하나요?","mode":"hybrid","top_k":3}'
```

## 내부 구조

- [`sas_rag/search_package.py`](/Users/noahhan/dev/sas9-RAG/sas94-rag-chat/sas_rag/search_package.py#L1)
  - 외부 `sas-94-search-api` 패키지를 import하는 경계층
- [`sas_rag/service.py`](/Users/noahhan/dev/sas9-RAG/sas94-rag-chat/sas_rag/service.py#L1)
  - 검색 결과를 받아 chat 응답으로 묶는 서비스 레이어
- [`sas_rag/generation.py`](/Users/noahhan/dev/sas9-RAG/sas94-rag-chat/sas_rag/generation.py#L1)
  - Gemini 호출
- [`scripts/app/chat_sas_rag.py`](/Users/noahhan/dev/sas9-RAG/sas94-rag-chat/scripts/app/chat_sas_rag.py#L1)
  - 사람용 CLI
- [`scripts/app/serve_sas_rag.py`](/Users/noahhan/dev/sas9-RAG/sas94-rag-chat/scripts/app/serve_sas_rag.py#L1)
  - 사람용 HTTP API와 간단한 브라우저 UI

## 외부 의존 관계

이 앱은 검색을 직접 구현하지 않는다. 아래 외부 컴포넌트를 사용한다.

- search package:
  - `pip install "sas-94-search-api @ git+https://github.com/kor-noah-han/sas-94-search-API.git@main"`
- search data release:
  - `sas94-search-data --output-dir .`

즉 이 저장소는 `chat app`, 검색 저장소는 `search engine` 역할이다.
