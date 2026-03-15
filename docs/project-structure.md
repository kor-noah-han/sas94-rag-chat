# Project Structure

이 저장소는 `사람용 chat 앱`이다. 검색 엔진은 외부 패키지 `sas-94-search-api`가 담당한다.

## `sas_rag/`

- `search_package.py`
  - 외부 search 패키지 import 경계층
- `service.py`
  - 검색 결과를 받아 chat 응답을 구성
- `generation.py`
  - Gemini 호출
- `app.py`
  - chat CLI/API에서 공통으로 쓰는 인자, 컨텍스트, 출력 유틸

## `scripts/app/`

- `chat_sas_rag.py`
  - 사람용 CLI
- `serve_sas_rag.py`
  - 사람용 chat API 서버와 간단한 브라우저 UI

## `scripts/ingest/`, `scripts/index/`

이 폴더들은 과거 문서 구축 파이프라인 유틸리티다. 현재 런타임의 핵심 책임은 아니다.

## `data/`

런타임에서 사용하는 검색 데이터는 원칙적으로 외부 search 패키지 Release에서 내려받는다. 이 저장소 안의 `data/`는 로컬 작업 산출물이거나 구축 과정에서 만든 파일일 수 있다.

## 책임 분리

- `sas-94-search-api`
  - 검색 구현
  - 검색용 DB
  - Search API
  - Python 패키지
- `sas9-RAG`
  - 사람용 chat
  - Gemini 응답 생성
  - chat CLI / chat API
