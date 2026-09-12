import unittest
from pathlib import Path

from backend.db.models import HotpotQA, HotpotQASplit
from backend.scripts.passages import build_passage_rows


class PipelineContractTests(unittest.TestCase):
    def test_readme_documents_raw_ingestion_then_passage_materialization(self):
        readme = Path("../README.md").read_text()
        self.assertLess(readme.index("uv run ingest"), readme.index("uv run build-passages"))

    def test_materializer_and_embedding_commands_are_registered(self):
        pyproject = Path("pyproject.toml").read_text()
        self.assertIn('build-passages = "backend.scripts.passages:main"', pyproject)
        self.assertIn('embed-passages = "backend.scripts.embeddings:main"', pyproject)

    def test_readme_documents_embedding_after_passage_materialization(self):
        readme = Path("../README.md").read_text()
        self.assertIn("OPENAI_API_KEY", readme)
        self.assertIn("uv run embed-passages", readme)
        self.assertLess(
            readme.index("uv run build-passages"),
            readme.index("uv run embed-passages"),
        )

    def test_passage_insert_fields_exclude_supporting_facts(self):
        row = HotpotQA(
            id="q1",
            question="q1",
            answer="a",
            type="bridge",
            level="easy",
            supporting_facts={"title": ["Shared"], "sent_id": [["Shared", 0]]},
            context={"title": ["Shared"], "sentences": [["Shared sentence."]]},
            split=HotpotQASplit.TRAIN,
        )

        passage_rows, _, _ = build_passage_rows([row])

        self.assertNotIn("supporting_facts", passage_rows[0])
