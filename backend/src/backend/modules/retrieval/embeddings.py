from dataclasses import dataclass
from typing import Any

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


def validated_embeddings(
    response: Any,
    expected_count: int,
) -> list[list[float]]:
    try:
        data = list(response.data)
    except AttributeError as error:
        raise ValueError("embedding response must include data") from error

    if len(data) != expected_count:
        raise ValueError(
            "embedding response count does not match requested passage count"
        )

    try:
        ordered_data = sorted(data, key=lambda item: item.index)
        indexes = [item.index for item in ordered_data]
    except AttributeError as error:
        raise ValueError("embedding response items must include an index") from error

    if indexes != list(range(expected_count)):
        raise ValueError("embedding response indexes are not contiguous and ordered")

    embeddings = [list(item.embedding) for item in ordered_data]
    for embedding in embeddings:
        if len(embedding) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"embedding dimension must be {EMBEDDING_DIMENSIONS}")
    return embeddings


@dataclass(slots=True)
class OpenAIQueryEmbedder:
    client: Any

    async def embed_query(self, query: str) -> list[float]:
        response = await self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query],
        )
        return validated_embeddings(response, expected_count=1)[0]
