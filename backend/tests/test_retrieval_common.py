import unittest

from backend.db.models import SourcePassage
from backend.modules.retrieval.pipelines._common import ranked_passages


def make_passage(title: str, content_hash: str) -> SourcePassage:
    return SourcePassage(
        title=title,
        normalized_title=title.lower(),
        sentences=[f"{title} text."],
        text=f"{title} text.",
        content_hash=content_hash,
    )


class RankedPassagesTests(unittest.TestCase):
    def test_maps_ordered_rows_to_one_based_contract_results(self):
        first = make_passage("First", "first")
        second = make_passage("Second", "second")

        results = ranked_passages(
            [(first, 2), (second, 1.25)],
            retriever="example",
        )

        self.assertEqual([result.passage_id for result in results], [first.id, second.id])
        self.assertEqual([result.title for result in results], ["First", "Second"])
        self.assertEqual([result.text for result in results], ["First text.", "Second text."])
        self.assertEqual([result.rank for result in results], [1, 2])
        self.assertEqual([result.score for result in results], [2.0, 1.25])
        self.assertEqual([result.retriever for result in results], ["example", "example"])

    def test_empty_rows_return_empty_results(self):
        self.assertEqual(ranked_passages([], retriever="example"), [])
