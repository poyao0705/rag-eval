import unittest
from pathlib import Path


class PassageMigrationTests(unittest.TestCase):
    def test_migration_declares_both_tables_and_current_parent_revision(self):
        migration = Path(
            "alembic/versions/d2e4f6a8b0c1_add_source_passages.py"
        ).read_text()

        self.assertIn(
            'down_revision: Union[str, Sequence[str], None] = "c24b04d30087"',
            migration,
        )
        self.assertIn('op.create_table("source_passage"', migration)
        self.assertIn('op.create_table("hotpot_qa_context"', migration)
        self.assertIn('"uq_source_passage_identity"', migration)
        self.assertIn('"hotpot_qa_context_pkey"', migration)

    def test_string_alignment_migration_matches_sqlmodel_string_columns(self):
        migration = Path(
            "alembic/versions/e7f9a1b3c5d7_align_passage_string_types.py"
        ).read_text()

        for table, column in (
            ("source_passage", "title"),
            ("source_passage", "normalized_title"),
            ("source_passage", "text"),
            ("source_passage", "content_hash"),
            ("hotpot_qa_context", "hotpot_qa_id"),
        ):
            self.assertIn(
                f'("{table}", "{column}")',
                migration,
            )
        self.assertIn("type_=sa.String()", migration)

    def test_embedding_migration_adds_reversible_cosine_hnsw_index(self):
        migration = Path(
            "alembic/versions/a9b8c7d6e5f4_add_source_passage_embedding.py"
        ).read_text()

        self.assertIn(
            'down_revision: Union[str, Sequence[str], None] = "f1a2b3c4d5e6"',
            migration,
        )
        self.assertIn('op.add_column(', migration)
        self.assertIn('"source_passage"', migration)
        self.assertIn('"embedding"', migration)
        self.assertIn('Vector(1536)', migration)
        self.assertIn('"source_passage_embedding_hnsw_idx"', migration)
        self.assertIn('postgresql_using="hnsw"', migration)
        self.assertIn('"embedding": "vector_cosine_ops"', migration)
        self.assertIn('op.drop_index(', migration)
        self.assertIn('op.drop_column("source_passage", "embedding")', migration)

    def test_search_vector_migration_adds_reversible_gin_index(self):
        migration = Path(
            "alembic/versions/f1a2b3c4d5e6_add_source_passage_search_vector.py"
        ).read_text()

        self.assertIn(
            'down_revision: Union[str, Sequence[str], None] = "e7f9a1b3c5d7"',
            migration,
        )
        self.assertIn('op.add_column(', migration)
        self.assertIn('"source_passage"', migration)
        self.assertIn('sa.Computed("to_tsvector', migration)
        self.assertIn('"search_vector"', migration)
        self.assertIn('"ix_source_passage_search_vector"', migration)
        self.assertIn('postgresql_using="gin"', migration)
        self.assertIn('op.drop_index(', migration)
        self.assertIn('op.drop_column("source_passage", "search_vector")', migration)
