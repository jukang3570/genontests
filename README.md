# 코드서빙 요청 확인용 프로젝트

GenOS 코드서빙에 요청을 보냈을 때 컨테이너까지 어떤 헤더와 본문이 전달되는지
확인하는 최소 FastAPI 서비스와 요청 스크립트입니다.

## 서버 (`main.py`)

- `GET /health` → `{"status": "ok"}`
- 그 밖의 모든 경로·메서드 → 받은 요청(method, path, query, client, 헤더, 본문)을
  컨테이너 로그에 출력하고 같은 내용을 응답으로 돌려줍니다.

```json
{"code": 0, "data": {"text": "요청 수신 완료", "request": {"method": "POST", "headers": {...}, "body": {...}}}}
```

- `{"question": "__verify__"}` 요청에는 `{"code": 0, "data": {"text": "verified"}}`를 반환합니다.
- `Authorization`, `Cookie` 등은 앞뒤 일부와 길이만 보여줍니다. 원문이 필요하면
  `MASK_SENSITIVE_HEADERS=false`로 실행합니다.
- 실행: `python main.py` (기본 포트 8080, `PORT` 환경변수로 변경)
- 업로드용 압축본: `dist/code-serving.zip` (`main.py`, `requirements.txt`)

## 요청 스크립트 (`send_request.py`)

```powershell
pip install -r requirements.txt
$env:GENOS_TOKEN = "GenOS에서 발급한 인증키"
python send_request.py                                # POST /chat (serving 431)
python send_request.py --question "안녕하세요"
python send_request.py --method GET --path /health
python send_request.py -H "X-Test: hello"             # 헤더 추가
python send_request.py --serving-id 431 --path /predict
python send_request.py --base http://localhost:8080   # 로컬 서버로 직접 요청
```

요청 정보, 응답 상태·헤더, 응답 본문을 순서대로 출력합니다.
