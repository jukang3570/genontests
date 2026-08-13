"""LangChain을 통해 GenOS LLM의 1차 의도분류를 호출하는 모듈."""

from typing import Protocol

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.domain import IntentClassification, create_structured_output_model
from app.observability import (
    async_timed_block,
    log_failure_diagnostic,
    logger,
    timed,
)
from app.prompt_loader import PromptBundle


class IntentClassifier(Protocol):
    """LangGraph가 의존하는 의도분류기 인터페이스."""

    async def classify(
        self,
        message: str,
        history: list[dict[str, str]],
        frontend_agent_code: str | None = None,
    ) -> IntentClassification: ...


class GenOSIntentClassifier:
    """LangChain ChatOpenAI로 GenOS의 OpenAI 호환 엔드포인트를 호출한다."""

    @timed("GenOS 분류기 초기화")
    def __init__(self, settings: Settings, prompt: PromptBundle) -> None:
        # 토큰이 없을 때 키워드 분류기로 대체하지 않는다. 운영 요청은 반드시
        # 설정된 LLM을 호출해야 하므로 시작 단계에서 명확하게 실패시킨다.
        if not settings.genos_bearer_token:
            raise ValueError("GENOS_BEARER_TOKEN 환경변수가 필요합니다.")

        # manifest.yaml의 에이전트 코드로 Enum과 JSON Schema를 동적으로 만든다.
        # 새 에이전트를 추가해도 이 Python 파일을 변경할 필요가 없다.
        output_model = create_structured_output_model(prompt.agent_codes)

        # 결합된 Markdown 프롬프트는 SystemMessage에 일반 문자열로 전달한다.
        # 이렇게 하면 프롬프트 내부 JSON 중괄호가 템플릿 변수로 해석되지 않는다.
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=prompt.system_prompt),
                (
                    "human",
                    "프론트에서 선택한 에이전트:\n{frontend_agent_code}\n\n"
                    "이전 대화:\n{history}\n\n"
                    "[이력 사용 지시]\n"
                    "현재 질문을 먼저 독립적으로 해석하세요. 현재 질문만으로 "
                    "대상과 요청이 명확하면 이전 대화의 주제를 추가하지 마세요. "
                    "현재 질문에 생략된 참조가 있을 때만 직접 관련된 이력을 "
                    "사용하세요.\n\n"
                    "현재 사용자 질문:\n{message}",
                ),
            ]
        )

        # base_url은 GenOS OpenAI 호환 API의 /v1 경로까지 포함한다.
        # 제한시간·재시도·최대 토큰은 응답이 끝나지 않는 상황을 막기 위해
        # 환경설정으로 명시한다. 상세 값은 Settings.llm_client_options 참고.
        llm = ChatOpenAI(
            base_url=settings.genos_openai_base_url,
            model=settings.genos_model,
            api_key=settings.genos_bearer_token,
            temperature=prompt.temperature,
            max_retries=settings.llm_max_retries,
        )

        # strict JSON Schema를 적용해 LLM 응답을 Pydantic 모델로 직접 받는다.
        # 별도의 수동 JSON 파싱이나 하드코딩된 키워드 분류는 사용하지 않는다.
        self._chain = chat_prompt | llm.with_structured_output(
            output_model,
            method="json_schema",
            strict=True,
        )
        # 실제 호출 직전에 전체 전달 내용을 로그로 확인할 수 있도록 결합된
        # 시스템 프롬프트를 보관한다. 토큰이나 API 키는 포함되지 않는다.
        self._system_prompt = prompt.system_prompt
        self._model = settings.genos_model
        self._endpoint = settings.genos_openai_base_url
        self._max_retries = settings.llm_max_retries
        logger.info(
            "======== LLM 준비 완료 | 모델=%s | 프롬프트버전=%s | "
            "에이전트개수=%d | 자동재시도=%d회 | "
            "코드위치=app/classifier.py:GenOSIntentClassifier.classify",
            settings.genos_model,
            prompt.version,
            len(prompt.agent_codes),
            settings.llm_max_retries,
        )

    @timed("1차 의도분류")
    async def classify(
        self,
        message: str,
        history: list[dict[str, str]],
        frontend_agent_code: str | None = None,
    ) -> IntentClassification:
        """선택 에이전트·동일 범위 이력·현재 질문을 LLM에 전달한다."""

        # Redis가 반환한 role/content 딕셔너리를 대화 순서대로 문자열화한다.
        history_text = "\n".join(
            f"{item['role']}: {item['content']}" for item in history
        )
        logger.info(
            "======== LLM 전달 준비 | 단계=마스터1차의도분류 | "
            "코드위치=app/classifier.py:GenOSIntentClassifier.classify | "
            "엔드포인트=%s | 모델=%s | 대화이력개수=%d | "
            "프론트에이전트=%s | 자동재시도=%d회",
            self._endpoint,
            self._model,
            len(history),
            frontend_agent_code or "선택하지 않음",
            self._max_retries,
        )

        # 네트워크 및 모델 응답 시간만 별도로 확인할 수 있도록 내부 구간을 잰다.
        try:
            async with async_timed_block("LLM 전달 및 응답 대기"):
                logger.info(
                    "======== LLM 요청 직전 | 단계=마스터1차의도분류 | "
                    "질문=%s | 대화이력개수=%d | 질문길이=%d | "
                    "StructuredOutput=IntentClassification JSON Schema",
                    message,
                    len(history),
                    len(message),
                )
                # 전체 프롬프트와 대화 내용을 운영 로그에 출력하지 않도록 주석
                # 처리한다. 문제 분석을 위해 다시 확인해야 할 때만 아래 블록의
                # 주석을 해제한다. 모델명, 이력 개수, 질문 길이 로그는 유지된다.
                # logger.info(
                #     "\n"
                #     "======== LLM 전체 프롬프트 시작 ========\n"
                #     "[시스템 프롬프트]\n%s\n\n"
                #     "[이전 대화]\n%s\n\n"
                #     "[현재 사용자 질문]\n%s\n"
                #     "======== LLM 전체 프롬프트 끝 ========",
                #     self._system_prompt,
                #     history_text or "(이전 대화 없음)",
                #     message,
                # )
                structured = await self._chain.ainvoke(
                    {
                        "history": history_text or "(이전 대화 없음)",
                        "message": message,
                        "frontend_agent_code": (
                            frontend_agent_code or "(선택하지 않음)"
                        ),
                    }
                )
        except Exception as exc:
            log_failure_diagnostic(
                stage="마스터 1차 의도분류 LLM 호출",
                code_location=(
                    "app/classifier.py:GenOSIntentClassifier.classify"
                ),
                exc=exc,
                likely_cause=(
                    "GenOS LLM 연결·인증·모델 설정 오류, 응답 시간 초과, 또는 "
                    "LLM 응답이 마스터 Structured Output JSON Schema와 불일치"
                ),
                corrective_action=(
                    ".env의 GENOS_URL/GENOS_SERVING_ID/GENOS_MODEL/"
                    "GENOS_BEARER_TOKEN을 확인하고 prompts/intent-classification의 "
                    "활성 프롬프트와 app/domain.py 출력 스키마를 확인하세요."
                ),
                retry_count=self._max_retries,
                context={
                    "endpoint": self._endpoint,
                    "model": self._model,
                    "history_count": len(history),
                    "frontend_agent_code": frontend_agent_code,
                    "message": message,
                },
            )
            raise

        # 동적으로 생성된 Enum은 API와 그래프에서 쓰기 쉬운 문자열로 변환한다.
        result = IntentClassification(
            refined_query=structured.refined_query,
            classification_type=structured.classification_type,
            agent_code=(
                structured.agent_code.value
                if structured.agent_code is not None
                else None
            ),
        )
        logger.info(
            "======== 1차 의도분류 LLM 원본 구조화 결과 | 값=%s",
            structured.model_dump(mode="json"),
        )
        logger.info(
            "======== 1차 의도분류 결과 | 분류유형=%s | 에이전트코드=%s | "
            "보정질문=%s",
            result.classification_type.value,
            result.agent_code,
            result.refined_query,
        )
        return result


@timed("의도분류기 생성")
def create_classifier(
    settings: Settings,
    prompt: PromptBundle,
) -> IntentClassifier:
    """운영 환경에서 사용할 실제 GenOS 의도분류기를 생성한다."""

    return GenOSIntentClassifier(settings, prompt)
