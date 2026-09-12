import unittest
import uuid

from backend.db.models import HotpotQAContext, SourcePassage


class PassageModelTests(unittest.TestCase):
    def test_source_passage_has_uuid_identity_and_natural_dedup_constraint(self):
        passage = SourcePassage(
            title="Example",
            normalized_title="example",
            sentences=["First sentence."],
            text="First sentence.",
            content_hash="abc",
        )

        self.assertIsInstance(passage.id, uuid.UUID)
        constraints = SourcePassage.__table__.constraints
        self.assertIn(
            {"normalized_title", "content_hash"},
            [
                set(c.columns.keys())
                for c in constraints
                if c.__class__.__name__ == "UniqueConstraint"
            ],
        )

    def test_context_identity_preserves_question_and_position(self):
        primary_keys = {
            column.name
            for column in HotpotQAContext.__table__.primary_key.columns
        }

        self.assertEqual(primary_keys, {"hotpot_qa_id", "position"})
