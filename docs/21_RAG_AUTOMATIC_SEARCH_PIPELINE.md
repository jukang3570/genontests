# Databricks RAG 자동 검색·Reranking·대화 이력 답변

## 1. 최종 실행 흐름

RP `RP_DOCUMENT_SEARCH`와 QUALIFICATION의 다섯 문서 세부 시나리오는 사용자에게
검색어를 다시 입력받지 않는다. 즉 `action`과 `humanInput` 왕복이 없다.

```text
원본 질문 + 같은 agent의 Redis history
→ 마스터 LLM: 오타·멀티턴 문맥을 반영한 refined_query
→ 서브에이전트 LLM: detail 1~N개 + detail별 rag_query + keywords[]
→ 각 detail Python handler
→ Databricks hybrid search(query=rag_query, keywords=keywords)
→ 검색 점수 임계값 필터
→ 문서가 있을 때만 GenOS reranker
→ reranking 점수 임계값 필터
→ 문서 답변 가능성 판별
→ refined_query + rag_query + keywords + 같은 agent의 과거 대화 + 최종 문서
→ 답변 LLM streaming
```

문서가 없으면 RAG LLM을 호출하지 않고 detail별 `no_documents_answer`를 반환한다.
MCP·파싱·reranker·답변 가능성 판별 중 오류가 나면 테이블이나 부분 답변을 만들지
않고 공통 안전 오류답변으로 전환한다.

## 2. 서브에이전트 구조화 출력

RAG detail의 manifest는 다음 배열 타입을 사용한다.

```yaml
parameter_definitions:
  rag_query:
    description: "현재 detail 요청만 분리한 완결된 검색 질문"
  keywords:
    value_type: "string_list"
    min_items: 1
    max_items: 10
    description: "세부 시나리오 검색 핵심어"
```

detail에는 `rag_query`와 `keywords`만 선언한다. `search_query`나
interaction/action은 선언하지 않는다.

```yaml
details:
  - code: "RP_DOCUMENT_SEARCH"
    parameters:
      - "rag_query"
      - "keywords"
```

LLM 구조화 출력 예시는 다음과 같다.

```json
{
  "matches": [
    {
      "scenario_code": "RP_DOCUMENTS",
      "detail_scenario_code": "RP_DOCUMENT_SEARCH",
      "parameters": {
        "rag_query": "RP 자동납부 연결 제한 기준을 알려줘",
        "keywords": ["RP", "자동납부", "연결 제한"]
      }
    }
  ]
}
```

배열 스키마 생성과 정규화 위치는
[`app/subagents/router.py`](../app/subagents/router.py)다. keywords가 비정상적으로
비어도 [`keyword_list()`](../app/mcp/scenarios/helpers.py)가 refined query에서
fallback 키워드를 만들기 때문에 Action을 발생시키지 않는다.

## 3. 하이브리드 검색 payload

QUALIFICATION은 [`qualification_document_search()`](../app/mcp/scenarios/qualification.py),
RP는 [`rp_document_search()`](../app/mcp/scenarios/rp.py)에서 직접 수정한다.

```python
query = (
    text(context.subagent.parameters.get("rag_query"))
    or context.refined_query
)
keywords = keyword_list(
    context.subagent.parameters.get("keywords"),
    fallback_query=query,
)

return await context.call(
    step_code="DOCUMENT_SEARCH",
    tool_name="databricks_hybrid_search",
    arguments={
        "query": query,
        "keywords": keywords,
        "query_type": "HYBRID",
        "index_name": "...",
        "columns": [...],
        "num_results": 5,
        "filters": {...},
    },
)
```

실제 Databricks MCP 계약에서 keywords 필드명이나 filter 형식이 달라지면 이 두
handler의 `arguments`만 변경한다. 프롬프트나 공통 MCP transport를 수정할 필요가
없다.

## 4. 검색 문서와 Reranking

[`app/answers.py`](../app/answers.py)의 `DefaultAnswerService.prepare()`가 다음 순서로
처리한다.

1. `structuredContent.data[]`를 공통 문서 객체로 변환
2. detail별 retrieval score 임계값 적용
3. 문서가 0건이면 `no_documents_answer`
4. `reranking_enabled=true`이면 detail의 rag query와 keywords를 GenOS reranker에 전달
5. reranking score와 `top_n` 적용
6. 최종 문서가 0건이면 `no_documents_answer`
7. 선택적으로 answerability LLM 판별
8. 최종 RAG 답변 스트리밍

detail별 임계점수와 문구는
[`app/rag_policies.py`](../app/rag_policies.py)의 `RAG_SCENARIO_POLICIES`에서 바꾼다.
실제 reranker 연결은 `.env`의 다음 값으로 제어한다.

```dotenv
RERANKING_ENABLED=true
RERANKING_SERVING_ID=226
RERANKING_MODEL=bge-reranker-v2-m3
RERANKING_BEARER_TOKEN=<secret>
RERANKING_ENDPOINT_PATH=rerank
```

`RERANKING_ENABLED=false`이면 같은 코드 경로에서 문서 순서를 유지하고 외부
reranker HTTP 호출만 생략한다.

## 5. 최종 답변 LLM 입력

최종 Human prompt에는 다음 네 블록을 넣는다.

```text
마스터 보정 사용자 질문
현재 세부 시나리오 검색 질문
세부 시나리오 검색 키워드
같은 에이전트의 이전 대화 이력
하이브리드 검색 및 Reranking 통과 문서
```

대화 이력은 [`_persist_user_message()`](../app/graph.py)에서 현재 질문을 저장하기
직전에 최종 `agent_code` 범위로 다시 조회한다. 따라서 프론트가 다른 agent를
선택했다가 HITL 승인으로 전환해도 다른 서브에이전트 이력이 섞이지 않는다.
Redis key나 저장 형식은 변경하지 않고 기존 history 조회 API만 재사용한다.

과거 대화는 생략된 문맥을 해석하는 보조정보일 뿐 업무 사실의 근거가 아니다.
업무 답변 근거는 Reranking을 통과한 문서로 제한한다. 이 규칙은
[`prompts/answer-generation/v1/rag/system.md`](../prompts/answer-generation/v1/rag/system.md)에
있다.

## 6. 운영 커스터마이징 순서

새 RAG detail을 추가할 때는 다음 순서로 작업한다.

1. subagent manifest에 detail과 `keywords` parameter 등록
2. system/scenario prompt에 detail 선택 규칙과 keyword 추출 규칙 추가
3. `app/mcp/scenarios/<agent>.py`에 hybrid search handler 작성
4. `app/mcp/scenarios/registry.py`에 handler 연결
5. `app/rag_policies.py`에 retrieval/reranking/answerability 정책 등록
6. 실제 MCP `structuredContent.data` 컬럼을 `app/answers.py` 문서 매핑과 맞춤
7. 무문서, 낮은 검색점수, 낮은 reranking 점수, 오류, 정상 답변을 각각 테스트

RAG detail에는 검색어 Action을 추가하지 않는다. 검색 품질이 낮으면 프론트 입력을
다시 받는 대신 prompt의 keywords 규칙, Databricks query/filter, 검색 임계값,
reranking 정책을 조정한다.
