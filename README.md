# 코드서빙 리비전 테스트

`POST /chat`을 호출하면 항상 아래 JSON을 반환하는 최소 FastAPI 서비스입니다.

```json
{
  "code": 0,
  "data": {
    "text": "코드서빙 리비전 버전 1입니다."
  }
}
```

실행 명령은 `python main.py`이며 기본 포트는 8080입니다. `PORT` 환경 변수로 포트를 변경할 수 있습니다.
