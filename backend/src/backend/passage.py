from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import unicodedata


@dataclass(frozen=True, slots=True)
class PassageCandidate:
    title: str
    normalized_title: str
    sentences: tuple[str, ...]
    text: str
    content_hash: str

    @property
    def identity(self) -> tuple[str, str]:
        return self.normalized_title, self.content_hash


def normalize_text(value: str) -> str:
    """Canonicalize Unicode text and whitespace."""
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\u00ad", "")
    return " ".join(value.split())


def normalize_title(value: str) -> str:
    return normalize_text(value).casefold()


def extract_context(context: object) -> list[PassageCandidate]:
    if not isinstance(context, Mapping):
        raise ValueError("context must be a mapping")
    if "title" not in context or "sentences" not in context:
        raise ValueError("context must contain title and sentences")

    titles = context["title"]
    sentence_blocks = context["sentences"]
    if not isinstance(titles, Sequence) or isinstance(titles, (str, bytes)):
        raise ValueError("title must be a sequence")
    if not isinstance(sentence_blocks, Sequence) or isinstance(
        sentence_blocks, (str, bytes)
    ):
        raise ValueError("sentences must be a sequence")
    if len(titles) != len(sentence_blocks):
        raise ValueError("title and sentences must have equal lengths")

    passages: list[PassageCandidate] = []
    for title, block in zip(titles, sentence_blocks):
        if not isinstance(title, str):
            raise ValueError("every title must be a string")
        if not isinstance(block, Sequence) or isinstance(block, (str, bytes)):
            raise ValueError("every sentence block must be a sequence of strings")
        if any(not isinstance(sentence, str) for sentence in block):
            raise ValueError("every sentence must be a string")

        normalized_title = normalize_title(title)
        normalized_sentences = tuple(normalize_text(sentence) for sentence in block)
        text = " ".join(normalized_sentences)
        if not normalized_title or not text:
            raise ValueError("title and text must not be empty after normalization")
        content_hash = sha256(text.encode("utf-8")).hexdigest()
        passages.append(
            PassageCandidate(
                title=title,
                normalized_title=normalized_title,
                sentences=tuple(block),
                text=text,
                content_hash=content_hash,
            )
        )
    return passages


def deduplicate_candidates(
    candidates: Iterable[PassageCandidate],
) -> list[PassageCandidate]:
    seen: set[tuple[str, str]] = set()
    unique: list[PassageCandidate] = []
    for candidate in candidates:
        if candidate.identity not in seen:
            seen.add(candidate.identity)
            unique.append(candidate)
    return unique
