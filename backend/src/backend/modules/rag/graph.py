from collections.abc import Sequence
from typing import NotRequired, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from backend.modules.retrieval.contracts import (
    RetrievalRequest,
    RetrievedPassage,
    Retriever,
)


class RAGState(TypedDict):
    question: str
    retrieved_passages: NotRequired[list[RetrievedPassage]]
    retrieval_context: NotRequired[list[str]]
    answer: NotRequired[str]


class AnswerGenerator(Protocol):
    async def answer(self, question: str, context: Sequence[str]) -> str: ...


def build_rag_graph(
    retriever: Retriever,
    session: AsyncSession,
    generator: AnswerGenerator,
) -> CompiledStateGraph:
    async def retrieve(state: RAGState) -> dict[str, list[RetrievedPassage] | list[str]]:
        passages = list(
            await retriever.retrieve(
                RetrievalRequest(query=state["question"], top_k=5),
                session,
            )
        )
        return {
            "retrieved_passages": passages,
            "retrieval_context": [f"{passage.title}\n{passage.text}" for passage in passages],
        }

    async def generate(state: RAGState) -> dict[str, str]:
        answer = await generator.answer(
            state["question"], state.get("retrieval_context", [])
        )
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("generator returned an empty or non-text answer")
        return {"answer": answer}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
