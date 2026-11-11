"""GenOS 코드서빙에 요청을 보내고 응답을 그대로 출력하는 스크립트.

토큰은 코드에 적지 말고 환경변수로 넘긴다.
    PowerShell:  $env:GENOS_TOKEN = "발급받은 인증키"
    bash:        export GENOS_TOKEN="발급받은 인증키"

사용 예:
    python send_request.py                          # POST /chat
    python send_request.py --question "안녕하세요"
    python send_request.py --method GET --path /health
    python send_request.py -H "X-Test: hello"       # 헤더 추가 (여러 번 가능)
    python send_request.py --base http://localhost:8080   # 로컬 서버로 직접 요청
"""

import argparse
import json
import os
import sys

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GenOS 코드서빙 요청 테스트")
    parser.add_argument("--genos-url", default=os.getenv("GENOS_URL", "https://genos.genon.ai"))
    parser.add_argument("--serving-id", default=os.getenv("GENOS_SERVING_ID", "431"))
    parser.add_argument(
        "--base",
        help="게이트웨이 대신 이 주소로 직접 요청 (예: http://localhost:8080)",
    )
    parser.add_argument("--path", default="/chat", help="코드서빙에서 구현한 경로")
    parser.add_argument("--method", default="POST")
    parser.add_argument("--question", default="헤더 확인 테스트")
    parser.add_argument("--token", default=os.getenv("GENOS_TOKEN"))
    parser.add_argument(
        "-H",
        "--header",
        action="append",
        default=[],
        help='추가 헤더 "Name: value"',
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def print_section(title: str) -> None:
    print(f"\n===== {title} =====")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")

    args = parse_args()
    base = args.base or f"{args.genos_url.rstrip('/')}/api/gateway/code_serving/{args.serving_id}"
    path = args.path if args.path.startswith("/") else f"/{args.path}"
    url = f"{base.rstrip('/')}{path}"
    method = args.method.upper()

    headers = {"Content-Type": "application/json", "X-Probe": "send_request.py"}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"
    elif not args.base:
        print("GENOS_TOKEN 환경변수 또는 --token 이 필요합니다.", file=sys.stderr)
        return 1
    for item in args.header:
        name, sep, value = item.partition(":")
        if not sep:
            print(f'헤더 형식 오류: "{item}" ("Name: value" 형식)', file=sys.stderr)
            return 1
        headers[name.strip()] = value.strip()

    body = None
    if method not in {"GET", "HEAD", "OPTIONS"}:
        body = {"question": args.question, "history": [], "overrideConfig": {}}

    print_section("요청")
    print(f"{method} {url}")
    for name, value in headers.items():
        shown = "Bearer ***" if name.lower() == "authorization" else value
        print(f"{name}: {shown}")
    if body is not None:
        print(json.dumps(body, ensure_ascii=False, indent=2))

    try:
        response = requests.request(method, url, headers=headers, json=body, timeout=args.timeout)
    except requests.RequestException as exc:
        print_section("요청 실패")
        print(repr(exc))
        return 1

    print_section(f"응답 {response.status_code} {response.reason}")
    for name, value in response.headers.items():
        print(f"{name}: {value}")

    print_section("응답 본문")
    try:
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except ValueError:
        print(response.text)
    return 0 if response.ok else 1


if __name__ == "__main__":
    sys.exit(main())
