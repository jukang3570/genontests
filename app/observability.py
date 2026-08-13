"""단계별 로그와 실행시간 측정을 위한 공통 도구."""

import inspect
import json
import logging
import sys
import time
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, AsyncIterator, Callable, Iterator, ParamSpec, TypeVar


P = ParamSpec("P")
R = TypeVar("R")

_LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s %(message)s | "
    "추적정보=request_id=%(request_id)s, session_id=%(session_id)s, "
    "thread_id=%(thread_id)s, user_id=%(user_id)s | "
    "호출코드=%(module)s.%(funcName)s:%(lineno)d"
)

# 애플리케이션 전체에서 같은 이름의 로거를 사용한다.
logger = logging.getLogger("master_agent")

# FastAPI의 비동기 요청은 같은 프로세스 안에서 동시에 실행될 수 있다. 전역
# 변수에 요청 ID를 넣으면 서로 다른 사용자의 로그가 섞이므로 ContextVar로 현재
# 비동기 실행 문맥에만 추적 식별자를 보관한다.
_LOG_CONTEXT: ContextVar[dict[str, str]] = ContextVar(
    "master_agent_log_context",
    default={},
)


class _RequestContextFilter(logging.Filter):
    """모든 로그 레코드에 요청 추적용 공통 필드를 추가한다."""

    def filter(self, record: logging.LogRecord) -> bool:
        context = _LOG_CONTEXT.get()
        record.request_id = context.get("request_id", "-")
        record.session_id = context.get("session_id", "-")
        record.thread_id = context.get("thread_id", "-")
        record.user_id = context.get("user_id", "-")
        return True


class _ReadableMultilineFormatter(logging.Formatter):
    """구분 필드와 복합 자료형을 터미널용 여러 줄 로그로 변환한다.

    기존 호출부는 ``제목 | 필드=값 | 필드=값`` 형식을 계속 사용한다. 포맷터가
    각 ``|`` 필드를 새 줄의 ``-`` 항목으로 바꾸므로 새로운 logger 호출을
    추가해도 별도의 줄바꿈 코드를 반복해서 작성할 필요가 없다.
    """

    def format(self, record: logging.LogRecord) -> str:
        message_template = str(record.msg)
        original_args = record.args
        try:
            record.args = _pretty_log_arguments(original_args)
            rendered = super().format(record)
        finally:
            # 같은 LogRecord를 다른 핸들러도 사용할 수 있으므로 원본 인자를
            # 반드시 복원해 포맷터 사이의 부작용을 막는다.
            record.args = original_args
        expanded = _expand_log_fields(rendered)
        # 함수/요청의 시작과 실패 진단은 앞에 빈 줄을 두어 연속된 터미널
        # 출력에서도 새로운 로그 블록의 경계를 즉시 식별할 수 있게 한다.
        if message_template.startswith(
            (
                "======== 단계 시작",
                "======== 요청 도착",
                "======== 애플리케이션 시작",
                "!!!!!!!! 실패 진단",
                "!!!!!!!! 처리 중단 진단",
            )
        ):
            return f"\n{expanded}"
        return expanded


def _pretty_log_arguments(arguments: Any) -> Any:
    """logger의 dict/list/tuple 값을 들여쓰기된 문자열로 변환한다."""

    if isinstance(arguments, tuple):
        return tuple(_pretty_log_value(value) for value in arguments)
    if isinstance(arguments, dict):
        # ``logger.info("%(name)s", {"name": ...})`` 형식도 보존한다.
        return {
            key: _pretty_log_value(value)
            for key, value in arguments.items()
        }
    return arguments


def _pretty_log_value(value: Any) -> Any:
    """복합 객체만 보기 좋은 JSON으로 만들고 숫자·문자열은 그대로 둔다."""

    if not isinstance(value, (dict, list, tuple, set)):
        return value
    serializable = sorted(value, key=str) if isinstance(value, set) else value
    pretty = json.dumps(
        serializable,
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    # 첫 줄은 ``필드=`` 뒤에 붙고 이후 JSON 줄은 하위 정보처럼 들여쓴다.
    return pretty.replace("\n", "\n        ")


def _expand_log_fields(rendered: str) -> str:
    """각 로그 줄의 파이프 구분 필드를 줄바꿈된 목록으로 펼친다."""

    output: list[str] = []
    for line in rendered.splitlines():
        fields = line.split(" | ")
        output.append(fields[0])
        output.extend(f"    - {field}" for field in fields[1:])
    return "\n".join(output)


def configure_logging(level: str = "INFO") -> None:
    """애플리케이션 시작 시 UTF-8 출력과 공통 로그 형식을 설정한다."""

    # Windows PowerShell의 기본 코드페이지와 Python 출력 인코딩이 다르면 한글
    # 로그가 깨질 수 있다. 지원되는 스트림은 UTF-8로 명시해 그대로 출력한다.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_LOG_FORMAT,
    )

    # uvicorn이 먼저 로깅 핸들러를 만든 환경에서는 basicConfig가 기존 핸들러를
    # 교체하지 않는다. 현재 루트 핸들러에도 필터와 포맷을 명시적으로 적용해
    # 개발·운영 실행 방식과 관계없이 같은 추적 필드가 출력되도록 한다.
    formatter = _ReadableMultilineFormatter(_LOG_FORMAT)
    for handler in logging.getLogger().handlers:
        if not any(
            isinstance(item, _RequestContextFilter)
            for item in handler.filters
        ):
            handler.addFilter(_RequestContextFilter())
        handler.setFormatter(formatter)


@contextmanager
def log_context(**values: Any) -> Iterator[None]:
    """현재 요청의 로그에 공통 추적 식별자를 자동으로 붙인다.

    호출부는 매 단계마다 request_id 등을 반복해서 전달하지 않아도 된다. 값은
    컨텍스트 종료 시 반드시 이전 상태로 복원되므로 동시 요청끼리 섞이지 않는다.
    """

    current = dict(_LOG_CONTEXT.get())
    current.update(
        {
            key: str(value)
            for key, value in values.items()
            if value is not None
        }
    )
    token = _LOG_CONTEXT.set(current)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


def log_failure_diagnostic(
    *,
    stage: str,
    code_location: str,
    exc: BaseException,
    likely_cause: str,
    corrective_action: str,
    retry_count: int = 0,
    context: Any | None = None,
) -> None:
    """실패 원인과 운영자가 확인할 수정 지점을 여러 줄과 스택으로 남긴다.

    호출자는 토큰·비밀번호를 제거한 ``context``만 전달해야 한다. 예외 체인의
    가장 안쪽 원인까지 함께 출력하므로 LangChain/httpx가 예외를 감싸더라도
    실제 연결·검증 오류를 찾기 쉽다. 이 함수는 예외를 삼키지 않으며 호출부가
    바로 ``raise``하여 최초 실패를 사용자 오류 이벤트까지 전달해야 한다.
    """

    root = exc
    visited: set[int] = set()
    while id(root) not in visited:
        visited.add(id(root))
        nested = root.__cause__ or root.__context__
        if nested is None:
            break
        root = nested

    logger.error(
        "!!!!!!!! 실패 진단 | 실패단계=%s | 코드위치=%s | "
        "예외유형=%s | 오류메시지=%s | 근본예외유형=%s | "
        "근본오류메시지=%s | 가능한원인=%s | 확인및수정=%s | "
        "자동재시도횟수=%d | 입력요약=%s",
        stage,
        code_location,
        type(exc).__name__,
        str(exc),
        type(root).__name__,
        str(root),
        likely_cause,
        corrective_action,
        retry_count,
        context if context is not None else "없음",
        exc_info=(type(exc), exc, exc.__traceback__),
    )


def timed(
    stage: str,
    *,
    expected_exceptions: tuple[type[BaseException], ...] = (),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """동기·비동기 함수의 시작, 완료, 중단, 실패와 소요시간을 기록한다.

    예외는 로그를 남긴 뒤 그대로 다시 발생시킨다. 따라서 이 데코레이터를
    적용해도 기존 비즈니스 동작이나 예외 처리 흐름은 바뀌지 않는다.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs):
                started = time.perf_counter()
                logger.info(
                    "======== 단계 시작 | %s | 함수=%s",
                    stage,
                    func.__qualname__,
                )
                try:
                    result = await func(*args, **kwargs)
                except expected_exceptions:
                    logger.info(
                        "======== 단계 중단 | %s | 함수=%s | 소요시간=%.3f초",
                        stage,
                        func.__qualname__,
                        _elapsed_seconds(started),
                    )
                    raise
                except Exception as exc:
                    logger.error(
                        "======== 단계 실패 | %s | 함수=%s | 코드위치=%s.%s | "
                        "예외유형=%s | 오류=%s | 자동재시도=없음 | "
                        "소요시간=%.3f초",
                        stage,
                        func.__qualname__,
                        func.__module__,
                        func.__qualname__,
                        type(exc).__name__,
                        exc,
                        _elapsed_seconds(started),
                        exc_info=(type(exc), exc, exc.__traceback__),
                    )
                    raise
                logger.info(
                    "======== 단계 완료 | %s | 함수=%s | 소요시간=%.3f초",
                    stage,
                    func.__qualname__,
                    _elapsed_seconds(started),
                )
                return result

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs):
            started = time.perf_counter()
            logger.info(
                "======== 단계 시작 | %s | 함수=%s",
                stage,
                func.__qualname__,
            )
            try:
                result = func(*args, **kwargs)
            except expected_exceptions:
                logger.info(
                    "======== 단계 중단 | %s | 함수=%s | 소요시간=%.3f초",
                    stage,
                    func.__qualname__,
                    _elapsed_seconds(started),
                )
                raise
            except Exception as exc:
                logger.error(
                    "======== 단계 실패 | %s | 함수=%s | 코드위치=%s.%s | "
                    "예외유형=%s | 오류=%s | 자동재시도=없음 | "
                    "소요시간=%.3f초",
                    stage,
                    func.__qualname__,
                    func.__module__,
                    func.__qualname__,
                    type(exc).__name__,
                    exc,
                    _elapsed_seconds(started),
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                raise
            logger.info(
                "======== 단계 완료 | %s | 함수=%s | 소요시간=%.3f초",
                stage,
                func.__qualname__,
                _elapsed_seconds(started),
            )
            return result

        return sync_wrapper

    return decorator


@contextmanager
def timed_block(stage: str) -> Iterator[None]:
    """함수 내부의 특정 동기 코드 구간만 별도로 측정한다."""

    started = time.perf_counter()
    logger.info("======== 구간 시작 | %s", stage)
    try:
        yield
    except Exception:
        logger.exception(
            "======== 구간 실패 | %s | 소요시간=%.3f초",
            stage,
            _elapsed_seconds(started),
        )
        raise
    logger.info(
        "======== 구간 완료 | %s | 소요시간=%.3f초",
        stage,
        _elapsed_seconds(started),
    )


@asynccontextmanager
async def async_timed_block(stage: str) -> AsyncIterator[None]:
    """LLM·MCP처럼 비동기로 실행되는 특정 코드 구간을 측정한다."""

    started = time.perf_counter()
    logger.info("======== 구간 시작 | %s", stage)
    try:
        yield
    except Exception:
        logger.exception(
            "======== 구간 실패 | %s | 소요시간=%.3f초",
            stage,
            _elapsed_seconds(started),
        )
        raise
    logger.info(
        "======== 구간 완료 | %s | 소요시간=%.3f초",
        stage,
        _elapsed_seconds(started),
    )


def _elapsed_seconds(started: float) -> float:
    """perf_counter 시작값을 기준으로 경과시간을 초 단위로 반환한다."""

    return time.perf_counter() - started
