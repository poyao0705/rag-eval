import json
from collections.abc import Awaitable, Callable
from typing import Any, NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    after_agent,
    before_agent,
    wrap_model_call,
)
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession

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
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), HumanMessage(content=question)],
        "retrieved_passages": [],
        "retrieval_context": [],
        "retrieval_count": 0,
        "answer": "",
    }


@wrap_model_call(state_schema=RAGState)
async def single_retrieval(
    request: ModelRequest,
    handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
) -> ModelResponse:
    retrieved = request.state.get("retrieval_count", 0) == 1
    if retrieved:
        request = request.override(
            tools=[], tool_choice="none", model_settings={"tool_choice": "none"}
        )
    else:
        request = request.override(
            tool_choice="retrieve", model_settings={"parallel_tool_calls": False}
        )
    response = await handler(request)
    message = response.result[-1]
    if not isinstance(message, AIMessage) or message.invalid_tool_calls:
        raise ValueError("model returned an invalid tool response")
    if retrieved:
        if message.tool_calls:
            raise ValueError("retrieval tool may only be called once")
        if not message.text.strip():
            raise ValueError("generator returned an empty or non-text answer")
    else:
        if len(message.tool_calls) != 1:
            raise ValueError("model must request exactly one retrieval tool call")
        call = message.tool_calls[0]
        if call["name"] != "retrieve" or not (call.get("id") or "").strip():
            raise ValueError("invalid retrieval tool name or call ID")
        args = call["args"]
        if set(args) != {"query"} or not isinstance(args["query"], str):
            raise ValueError("retrieval tool requires only a string query")
        RetrievalRequest(query=args["query"], top_k=5)
    return response


@after_agent(state_schema=RAGState)
def expose_answer(state: RAGState, runtime: Runtime) -> dict[str, str]:
    return {"answer": state["messages"][-1].text}


def build_rag_graph(
    retriever: Retriever,
    session: AsyncSession,
    generator: BaseChatModel,
) -> CompiledStateGraph:
    @tool
    async def retrieve(query: str, runtime: ToolRuntime) -> Command:
        """Search the document corpus for evidence using one focused search query."""
        if runtime.state.get("retrieval_count", 0) != 0:
            raise ValueError("retrieval tool may only be called once")
        passages = list(
            await retriever.retrieve(RetrievalRequest(query=query, top_k=5), session)
        )
        context = [f"{passage.title}\n{passage.text}" for passage in passages]
        return Command(
            update={
                "retrieved_passages": passages,
                "retrieval_context": context,
                "retrieval_count": 1,
                "messages": [
                    ToolMessage(content=json.dumps(context), tool_call_id=runtime.tool_call_id)
                ],
            }
        )

    return create_agent(
        model=generator,
        tools=[retrieve],
        system_prompt=INSTRUCTIONS,
        state_schema=RAGState,
        middleware=[initialize, single_retrieval, expose_answer],
    )
