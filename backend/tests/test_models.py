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
        constraints = SourcePassage.__table__.constraints  # pyright: ignore[reportAttributeAccessIssue]
        self.assertIn(
            {"normalized_title", "content_hash"},
            [
                set(c.columns.keys())
                for c in constraints
                if c.__class__.__name__ == "UniqueConstraint"
            ],
        )

    def test_source_passage_has_stored_english_search_vector(self):
        column = SourcePassage.__table__.c.search_vector  # pyright: ignore[reportAttributeAccessIssue]

        self.assertEqual(column.type.__class__.__name__, "TSVECTOR")
        self.assertIsNotNone(column.computed)
        self.assertEqual(
            str(column.computed.sqltext),
            "to_tsvector('english', text)",
        )
        self.assertTrue(column.computed.persisted)

    def test_source_passage_has_nullable_1536_dimension_embedding(self):
        column = SourcePassage.__table__.c.embedding  # pyright: ignore[reportAttributeAccessIssue]

        self.assertEqual(column.type.__class__.__name__, "VECTOR")
        self.assertEqual(column.type.dim, 1536)
        self.assertTrue(column.nullable)
        index = next(
            index
            for index in SourcePassage.__table__.indexes  # pyright: ignore[reportAttributeAccessIssue]
            if index.name == "source_passage_embedding_hnsw_idx"
        )
        self.assertEqual(index.dialect_options["postgresql"]["using"], "hnsw")
        self.assertEqual(
            index.dialect_options["postgresql"]["ops"],
            {"embedding": "vector_cosine_ops"},
        )

    def test_context_identity_preserves_question_and_position(self):
        primary_keys = {
            column.name
            for column in HotpotQAContext.__table__.primary_key.columns  # pyright: ignore[reportAttributeAccessIssue]
        }

        self.assertEqual(primary_keys, {"hotpot_qa_id", "position"})
