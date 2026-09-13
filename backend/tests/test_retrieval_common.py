import unittest
from uuid import UUID

from backend.db.models import SourcePassage
from backend.modules.retrieval.utils import ranked_passages, reciprocal_rank_fusion


def make_passage(
    title: str,
    content_hash: str,
    passage_id: UUID | None = None,
) -> SourcePassage:
    passage = SourcePassage(
        title=title,
        normalized_title=title.lower(),
        sentences=[f"{title} text."],
        text=f"{title} text.",
        content_hash=content_hash,
    )
    if passage_id is not None:
        passage.id = passage_id
    return passage


class RankedPassagesTests(unittest.TestCase):
    def test_maps_ordered_rows_to_one_based_contract_results(self):
        first = make_passage("First", "first")
        second = make_passage("Second", "second")

        results = ranked_passages(
            [(first, 2.0), (second, 1.25)],
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


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_fuses_overlap_with_one_based_ranks_and_resets_output_rank(self):
        first = make_passage("First", "first")
        second = make_passage("Second", "second")
        third = make_passage("Third", "third")
        fts = ranked_passages([(first, 100.0), (second, 1.0)], retriever="fts")
        vector = ranked_passages([(second, 0.0), (third, 1000.0)], retriever="vector")
        input_lists = [fts.copy(), vector.copy()]

        results = reciprocal_rank_fusion(input_lists, retriever="hybrid")

        self.assertEqual(
            [result.passage_id for result in results],
            [second.id, first.id, third.id],
        )
        self.assertEqual([result.rank for result in results], [1, 2, 3])
        self.assertEqual([result.retriever for result in results], ["hybrid"] * 3)
        self.assertEqual(
            [result.score for result in results],
            [1 / 61 + 1 / 62, 1 / 61, 1 / 62],
        )
        self.assertEqual(input_lists, [fts, vector])

    def test_uses_rank_positions_instead_of_component_scores(self):
        first = make_passage("First", "first")
        second = make_passage("Second", "second")
        results = reciprocal_rank_fusion(
            [ranked_passages([(first, 1.0), (second, 1000.0)], retriever="fts")],
            retriever="hybrid",
        )

        self.assertEqual([result.passage_id for result in results], [first.id, second.id])

    def test_breaks_fused_score_ties_by_passage_id(self):
        higher_id = make_passage(
            "Higher",
            "higher",
            UUID("00000000-0000-0000-0000-000000000002"),
        )
        lower_id = make_passage(
            "Lower",
            "lower",
            UUID("00000000-0000-0000-0000-000000000001"),
        )
        lists = [
            ranked_passages([(higher_id, 1.0)], retriever="fts"),
            ranked_passages([(lower_id, 1.0)], retriever="vector"),
        ]

        results = reciprocal_rank_fusion(lists, retriever="hybrid")

        self.assertEqual([result.passage_id for result in results], [lower_id.id, higher_id.id])

    def test_empty_lists_return_empty_results(self):
        self.assertEqual(reciprocal_rank_fusion([], retriever="hybrid"), [])
        self.assertEqual(reciprocal_rank_fusion([[], []], retriever="hybrid"), [])

    def test_counts_duplicate_passage_once_per_component(self):
        first = make_passage("First", "first")
        second = make_passage("Second", "second")
        component = ranked_passages([(first, 1.0), (second, 1.0)], retriever="fts")

        results = reciprocal_rank_fusion(
            [[component[0], component[0], component[1]]],
            retriever="hybrid",
        )

        self.assertEqual([result.passage_id for result in results], [first.id, second.id])
        self.assertEqual([result.score for result in results], [1 / 61, 1 / 63])

    def test_rejects_negative_or_non_integer_rank_constant(self):
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            reciprocal_rank_fusion([], rank_constant=-1)
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            reciprocal_rank_fusion([], rank_constant=1.5)  # type: ignore[arg-type]
