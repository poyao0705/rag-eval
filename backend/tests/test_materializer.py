import unittest

from backend.db.models import HotpotQA, HotpotQASplit
from backend.scripts.passages import build_passage_rows


class MaterializerTests(unittest.TestCase):
    def test_build_passage_rows_reuses_one_passage_for_two_questions(self):
        shared_context = {
            "title": ["Shared"],
            "sentences": [["Shared sentence."]],
        }
        rows = [
            HotpotQA(
                id="q1", question="q1", answer="a", type="bridge", level="easy",
                supporting_facts={}, context=shared_context, split=HotpotQASplit.TRAIN,
            ),
            HotpotQA(
                id="q2", question="q2", answer="a", type="bridge", level="easy",
                supporting_facts={}, context=shared_context, split=HotpotQASplit.TRAIN,
            ),
        ]

        passage_rows, link_rows, candidate_count = build_passage_rows(rows)

        self.assertEqual(candidate_count, 2)
        self.assertEqual(len(passage_rows), 1)
        self.assertEqual(len(link_rows), 2)
        self.assertEqual({link["position"] for link in link_rows}, {0})

    def test_build_passage_rows_keeps_same_title_with_different_text(self):
        rows = [
            HotpotQA(
                id="q1", question="q1", answer="a", type="bridge", level="easy",
                supporting_facts={},
                context={"title": ["A", "A"], "sentences": [["One."], ["Two."]]},
                split=HotpotQASplit.TRAIN,
            ),
        ]

        passage_rows, _, _ = build_passage_rows(rows)

        self.assertEqual(len(passage_rows), 2)
