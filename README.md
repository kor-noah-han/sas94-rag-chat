# SAS 문서 기반 챗봇

SAS 9.4 공식 문서를 검색하고, 그 내용을 근거로 답변하는 웹 챗 애플리케이션이다.

외부 검색 패키지 [`sas-94-search-api`](https://github.com/kor-noah-han/sas-94-search-API)가 문서 검색을 담당하고, 이 저장소는 **웹 UI · 챗 API · LLM 응답 생성**을 담당한다.

---

## 아키텍처

```
사용자 브라우저
    │  POST /api/chat
    ▼
serve_sas_rag.py (ThreadingHTTPServer, 포트 8787)
    │
    ├─ 1. 쿼리 재작성 (LLM)
    │       query → 검색에 최적화된 영문 SAS 키워드
    │
    ├─ 2. 문서 검색 (sas-94-search-api)
    │       ├─ lexical  : SQLite FTS5 전문 검색
    │       ├─ dense    : Qdrant 벡터 검색 (FastEmbed)
    │       └─ hybrid   : lexical + dense 결합 + 재순위
    │
    └─ 3. 답변 생성 (LLM)
            context(검색 결과) + query → 한국어 답변
```

**LLM 연동** — OpenAI 호환 엔드포인트(`/chat/completions`)를 사용하므로 Azure AI Foundry, OpenAI, 로컬 서버 등 어디서나 동작한다.

**기본 검색 모드**는 `lexical`로 Qdrant 없이 즉시 실행 가능하다. `dense` / `hybrid` 모드는 Qdrant 서버가 필요하다.

---

## 역할 분리

| 저장소 | 역할 |
|---|---|
| [`sas-94-search-api`](https://github.com/kor-noah-han/sas-94-search-API) | 검색 엔진 · 검색 데이터 릴리즈 |
| **sas94-rag-chat** (이 저장소) | 웹 UI · 챗 API · LLM 답변 생성 |

---

## 빠른 시작

### 1. 환경 설정

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

### 2. 검색 데이터 준비

[sas-94-search-API Releases](https://github.com/kor-noah-han/sas-94-search-API/releases)에서 최신 버전을 받아 `data/` 디렉터리에 압축 해제한다.

```powershell
tar -xzf data/sas94-search-data-v0.1.5.tar.gz -C data/
```

### 3. 환경 변수 설정

프로젝트 루트에 `.env` 파일 생성:

```env
# LLM (OpenAI 호환 엔드포인트)
OPENAI_BASE_URL = https://{리소스명}.services.ai.azure.com/openai/v1/
OPENAI_API_KEY  = {API 키}
OPENAI_MODEL    = {모델명}

# 검색 데이터 경로
CORPUS_PATH = data/sas94-search-data-v0.1.5/data/processed/sas-rag/corpus/sas9-pdf-corpus.jsonl
FTS_DB_PATH = data/sas94-search-data-v0.1.5/data/processed/sas-rag/search/sas9-pdf-fts.db
```

### 4. 서버 실행

```powershell
.venv\Scripts\python.exe scripts\serve_sas_rag.py --port 8787
```

브라우저에서 `http://127.0.0.1:8787` 접속.

---

## API

### `GET /health`

```json
{"ok": true}
```

### `POST /api/chat`

```bash
curl -X POST http://127.0.0.1:8787/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "PROC CORR는 어떻게 써?", "mode": "lexical", "top_k": 5}'
```

**요청 파라미터**

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `query` | string | 필수 | 질문 |
| `mode` | string | `lexical` | 검색 모드 (`lexical` / `dense` / `hybrid`) |
| `top_k` | int | 5 | 반환할 최종 문서 수 |
| `temperature` | float | 0.1 | LLM 생성 온도 |
| `show_sources` | bool | true | 소스 카드 포함 여부 |

**응답**

```json
{
  "answer": "...",
  "sources": [
    {
      "docset": "lepg",
      "section_path_text": "PROC CORR / PROC CORR Statement",
      "page_start": 12,
      "page_end": 13
    }
  ],
  "retrieval": {
    "timings_ms": {"route": 5, "lexical": 210, "rerank": 0}
  }
}
```

---

## CLI

대화형:

```bash
.venv\Scripts\python.exe scripts\chat_sas_rag.py
```

한 번만 질문:

```bash
.venv\Scripts\python.exe scripts\chat_sas_rag.py "매크로 변수 정의하는 법" --show-sources
```

**인터랙티브 명령어**

| 명령 | 설명 |
|---|---|
| `/mode hybrid\|lexical\|dense` | 검색 모드 전환 |
| `/topk 5` | 검색 결과 수 변경 |
| `/sources on\|off` | 소스 출력 토글 |
| `/debug on\|off` | 검색 과정 상세 출력 |
| `/config` | 현재 설정 확인 |
| `/exit` | 종료 |

---

## 프로젝트 구조

```
sas94-rag-chat/
├── sas_rag/
│   ├── app.py            # CLI/서버 공통 인수 파싱, 컨텍스트 조립
│   ├── generation.py     # LLM 호출 (OpenAI 호환 chat/completions)
│   ├── prompts.py        # 시스템 프롬프트 · 유저 프롬프트 빌더
│   ├── retrieval.py      # 검색 래퍼 (재시도, 쿼리 재작성 로직)
│   ├── search_package.py # sas-94-search-api 패키지 import 경계
│   ├── service.py        # 검색 → LLM → 응답 조립 파이프라인
│   └── settings.py       # OpenAI 환경 변수 로딩
├── scripts/
│   ├── app/
│   │   ├── chat_sas_rag.py   # 대화형/one-shot CLI 진입점
│   │   └── serve_sas_rag.py  # HTTP 서버 진입점
│   └── ingest/               # 검색 데이터 빌드 스크립트 (선택)
├── web/
│   └── chat_ui.html      # 단일 파일 브라우저 UI
├── docs/
│   └── deployment.md     # Cloudflare Tunnel 배포 가이드
├── docker-compose.qdrant.yml  # Qdrant 로컬 실행용
└── requirements.txt
```

---

## 핵심 모듈 설명

### `sas_rag/service.py`
검색 → LLM의 전체 파이프라인을 담당한다. 검색 결과가 부족하면 LLM으로 쿼리를 재작성해 재검색하는 fallback 로직이 있다.

### `sas_rag/generation.py`
OpenAI 호환 `chat/completions` 엔드포인트를 `urllib`로 직접 호출한다. `certifi`가 있으면 TLS 검증에 사용한다. `call_llm()`은 최종 답변 생성, `rewrite_query_for_search()`는 쿼리 재작성에 사용된다.

### `sas_rag/prompts.py`
답변 생성과 쿼리 재작성에 사용되는 프롬프트를 정의한다. 기본 답변 언어는 **한국어**이며, 사용자가 다른 언어로 질문하면 해당 언어로 응답한다.

### `web/chat_ui.html`
외부 의존성 없는 단일 HTML 파일이다. KoPubWorld Dotum 웹폰트, SAS 블루 컬러 테마, 마크다운 렌더링, 소스 카드, 검색 과정 표시를 모두 포함한다.

---

## 외부 서비스 의존성

| 서비스 | 필수 여부 | 용도 |
|---|---|---|
| OpenAI 호환 LLM | **필수** | 쿼리 재작성 · 답변 생성 |
| Qdrant | 선택 | dense / hybrid 검색 모드 |
| Cloudflare Tunnel | 선택 | 외부 도메인 노출 |

---

## Qdrant 로컬 실행

dense / hybrid 모드를 사용할 경우:

```bash
docker compose -f docker-compose.qdrant.yml up -d
```

기본 URL: `http://localhost:6333`

---

## 배포

Cloudflare Tunnel을 이용한 외부 노출 방법은 [docs/deployment.md](docs/deployment.md)를 참고한다.

---

## 제약 사항

- SAS Viya 전용 문서는 포함하지 않는다. SAS 9.4 문서 기준이다.
- 각 질문은 독립적으로 처리된다. 대화 히스토리를 유지하지 않는다.
- PC가 꺼지면 서비스가 중단된다. 앱 서버와 Cloudflare Tunnel이 모두 실행 중이어야 한다.
- 대용량 원본 데이터, Qdrant 로컬 파일, PDF는 Git에 포함하지 않는다.
