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
