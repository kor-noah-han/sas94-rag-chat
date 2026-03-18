# 배포 가이드 — Cloudflare Tunnel + Windows PC

이 앱은 별도 서버 없이 **내 PC를 서버로** 사용하고, Cloudflare Tunnel을 통해 외부에 노출한다.

---

## 전제 조건

- Python 3.11+
- Cloudflare 계정 및 도메인 (Cloudflare DNS로 관리 중)
- `cloudflared.exe` 다운로드

---

## 1. cloudflared 다운로드

```powershell
curl -L -o C:\tmp\cloudflared.exe https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
```

---

## 2. Python 환경 설정

프로젝트 루트에서:

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

---

## 3. 환경변수 설정

`.env` 파일을 프로젝트 루트에 생성:

```env
# Azure AI Foundry (OpenAI-compatible)
OPENAI_BASE_URL = https://{리소스명}.services.ai.azure.com/openai/v1/
OPENAI_API_KEY  = {API 키}
OPENAI_MODEL    = {모델명}

# 검색 데이터 경로 (릴리즈 압축 해제 후 실제 경로로 수정)
CORPUS_PATH = data/sas94-search-data-{버전}/data/processed/sas-rag/corpus/sas9-pdf-corpus.jsonl
FTS_DB_PATH = data/sas94-search-data-{버전}/data/processed/sas-rag/search/sas9-pdf-fts.db
```

---

## 4. 검색 데이터 준비

[sas-94-search-API Releases](https://github.com/kor-noah-han/sas-94-search-API/releases)에서 최신 버전 다운로드 후 `data/` 디렉토리에 압축 해제:

```powershell
# 예: v0.1.5
tar -xzf data/sas94-search-data-v0.1.5.tar.gz -C data/
```

압축 해제 후 `.env`의 `CORPUS_PATH`, `FTS_DB_PATH` 경로를 실제 경로로 수정한다.

---

## 5. 앱 서버 실행

```powershell
.venv\Scripts\python.exe scripts\serve_sas_rag.py --port 8787
```

정상 실행 확인:

```powershell
curl http://127.0.0.1:8787/health
# {"ok": true}
```

---

## 6. Cloudflare Tunnel 설정

### 6-1. 터널 생성

[Cloudflare Zero Trust 대시보드](https://one.dash.cloudflare.com) → **Networks → Tunnels → Create a tunnel**

- Connector: `Cloudflared`
- 터널 이름 입력 (예: `sas-rag-chat`)
- OS: `Windows` 선택
- 토큰 포함 명령어 복사: `cloudflared.exe service install eyJ...`

### 6-2. ingress 설정 파일 생성

`C:\Users\{사용자명}\.cloudflared\config.yml`:

```yaml
ingress:
  - hostname: sas94-rag-chat.axiomark.org
    service: http://localhost:8787
  - service: http_status:404
```

### 6-3. Cloudflare DNS에 CNAME 추가

[Cloudflare 대시보드](https://dash.cloudflare.com) → 도메인 선택 → **DNS → Records → Add record**

| 항목 | 값 |
|------|----|
| Type | CNAME |
| Name | `sas94-rag-chat` |
| Target | `{터널 ID}.cfargotunnel.com` |
| Proxy | 주황 구름 (Proxied) ON |

터널 ID는 대시보드 터널 상세 페이지 Overview에서 확인.

### 6-4. 터널 실행

```powershell
C:\tmp\cloudflared.exe tunnel --config "C:\Users\{사용자명}\.cloudflared\config.yml" run --protocol http2 --token eyJ...
```

연결 확인 로그:

```
INF Registered tunnel connection connIndex=0 ... location=icn06 protocol=http2
```

---

## 7. 접속 확인

브라우저 또는 핸드폰에서:

```
https://sas94-rag-chat.axiomark.org
```

> DNS 변경 직후에는 1~2분 전파 시간이 필요할 수 있다. PC의 로컬 DNS 캐시가 남아있으면 핸드폰(LTE)으로 먼저 확인한다.

---

## 8. 운영 시 주의사항

- PC가 꺼지면 서비스가 중단된다.
- 앱 서버(`serve_sas_rag.py`)와 터널(`cloudflared`)이 **둘 다** 실행 중이어야 한다.
- 코드 변경 시 앱 서버만 재시작하면 된다. 터널은 재시작 불필요.
- 자동 시작이 필요하면 Windows 작업 스케줄러 또는 서비스로 등록한다.
