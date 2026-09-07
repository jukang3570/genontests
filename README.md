# GenOS LLM 코드서빙

POST /chat에서 question을 받아 GenOS LLM을 한 번 호출하고 JSON으로 반환합니다.
MCP, RAG, 의도 분류, Redis, SSE, 프론트엔드는 제거했습니다.
대화 이력을 저장하지 않으며 각 질문은 독립적으로 처리합니다.

## 배포

배포 파일은 main.py, requirements.txt입니다. dist/code-serving.zip에도 두 파일만 포함됩니다.
코드서빙에 파일을 등록하고 실행 명령을 `python main.py`로 설정하세요.
ASGI 진입점은 main:app, 기본 포트는 8080이며 PORT로 변경합니다.
플랫폼 화면의 정확한 등록 절차는 실제 배포 대상 확인이 필요합니다.

.env.example의 값을 배포 환경변수에 등록하세요. .env를 자동으로 읽지는 않습니다.
GENOS_BEARER_TOKEN에는 실제 LLM 접근 토큰을 보안 설정으로 주입해야 합니다.
기본 URL, 서빙 ID 850, 모델 qwen/qwen3.7-flash는 기존 저장소 설정을 유지한 값입니다.
현재 가용성은 별도 확인이 필요합니다. GENOS_SERVING_ID는 호출할 **LLM 서빙 ID**이며
새 코드서빙 워크플로우 ID가 아닙니다.

LLM 호출 주소: GENOS_URL/api/gateway/rep/serving/GENOS_SERVING_ID/v1/chat/completions

## 요청과 응답

요청: `{"question":"안녕하세요"}`

응답: `{"code":0,"data":{"text":"LLM 답변"}}`

- GET /health: `{"status":"ok"}` (프로세스 확인용)
- question이 **verify**: `{"code":0,"data":{"text":"verified"}}` (LLM 호출 없음)
- 빈 질문: 샘플과 동일하게 HTTP 200, code 0, text "[ERROR] question is empty"
- 잘못된 JSON/입력 타입: HTTP 400, code 400
- LLM 설정 누락/오류: HTTP 503, code 503
- LLM 연결/응답 오류: HTTP 502, code 502
- LLM 타임아웃: HTTP 504, code 504

오류도 {code, data: {text}}로 반환합니다. LLM 원문 오류와 인증 토큰은 노출하지 않습니다.
/health와 **verify** 성공만으로 실제 LLM 연결이 검증되지는 않습니다.

## 로컬 실행과 검증

```powershell
python -m pip install -r requirements.txt
# GENOS_BEARER_TOKEN 등은 실행 전에 환경변수로 주입
python main.py
```

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8080/chat -ContentType 'application/json' -Body '{"question":"**verify**"}'
python -m unittest discover -s tests -v
```

테스트는 가짜 HTTP 응답을 사용하며 실제 LLM이나 배포 서버에 접속하지 않습니다.
