import json
from typing import Any, NotRequired, cast

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    ToolCallLimitMiddleware,
    after_agent,
    before_agent,
)
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import DEFAULT_RAG_CONFIG, RAGConfig
from backend.modules.rag.generation import INSTRUCTIONS
from backend.modules.retrieval.contracts import (
    RetrievalRequest,
    RetrievedPassage,
    Retriever,
)


class RAGState(AgentState):
    question: str
    retrieved_passages: NotRequired[list[RetrievedPassage]]
    retrieval_context: NotRequired[list[str]]
    retrieval_count: NotRequired[int]
    answer: NotRequired[str]


@before_agent(state_schema=RAGState)
def initialize(state: RAGState, runtime: Runtime) -> dict[str, Any]:
    question = state.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonblank string")
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            HumanMessage(content=question),
        ],
        "retrieved_passages": [],
        "retrieval_context": [],
        "retrieval_count": 0,
        "answer": "",
    }


@after_agent(state_schema=RAGState)
def expose_answer(state: RAGState, runtime: Runtime) -> dict[str, str]:
    message = state["messages"][-1]
    if not isinstance(message, AIMessage) or message.invalid_tool_calls or message.tool_calls:
        raise ValueError("model returned an invalid final response")
    if not message.text.strip():
        raise ValueError("generator returned an empty or non-text answer")
    return {"answer": message.text}


def build_rag_graph(
    retriever: Retriever,
    session: AsyncSession,
    generator: BaseChatModel,
    config: RAGConfig = DEFAULT_RAG_CONFIG,
) -> CompiledStateGraph:
    if config.top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    @tool
    async def retrieve(query: str, runtime: ToolRuntime) -> Command:
        """Search the document corpus for evidence using one focused search query."""
        if not query.strip():
            raise ValueError("retrieval tool query must not be blank")
        passages = list(
            await retriever.retrieve(
                RetrievalRequest(query=query, top_k=config.top_k), session
            )
        )
        context = [f"{passage.title}\n{passage.text}" for passage in passages]
        return Command(
            update={
                "retrieved_passages": passages,
                "retrieval_context": context,
                "retrieval_count": 1,
                "messages": [
                    ToolMessage(
                        content=json.dumps(context), tool_call_id=runtime.tool_call_id
                    )
                ],
            }
        )

    return create_agent(
        model=generator,
        tools=[retrieve],
        system_prompt=INSTRUCTIONS,
        state_schema=RAGState,
        middleware=[
            initialize,
            # create_agent merges middleware state schemas at runtime.
            cast(
                Any,
                ToolCallLimitMiddleware(
                    tool_name="retrieve", run_limit=1, exit_behavior="continue"
                ),
            ),
            expose_answer,
        ],
    )
