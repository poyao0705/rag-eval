import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from rag_eval.cohort import QAExample, load_cohort


class CohortLoaderTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _session(rows):
        result = SimpleNamespace(
            mappings=lambda: SimpleNamespace(all=lambda: rows)
        )
        return SimpleNamespace(execute=AsyncMock(return_value=result))

    async def test_cohort_query_is_fixed_and_labeled(self):
        rows = [
            dict(id=f"qa-{i}", question=f"Question {i}?", answer=f"Answer {i}")
            for i in range(20)
        ]
        session = self._session(rows)

        cohort = await load_cohort(session)

        statement, params = session.execute.await_args.args
        sql = str(statement)
        self.assertIn("q.split = 'VALIDATION'", sql)
        self.assertIn(
            "ORDER BY md5(CAST(:seed AS text) || ':' || q.id), q.id", sql
        )
        self.assertNotIn("random()", sql.lower())
        self.assertEqual(params, {"seed": 42, "sample_size": 20})
        self.assertEqual(cohort[0].answer, "Answer 0")
        self.assertEqual(len(cohort), 20)

    async def test_rejects_incomplete_duplicate_and_blank_cohorts(self):
        valid_rows = [
            dict(id=f"qa-{i}", question=f"Question {i}?", answer=f"Answer {i}")
            for i in range(20)
        ]
        cases = [
            valid_rows[:19],
            [*valid_rows[:19], {**valid_rows[19], "id": valid_rows[0]["id"]}],
            [*valid_rows[:19], {**valid_rows[19], "answer": "   "}],
        ]

        for rows in cases:
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    await load_cohort(self._session(rows))

    async def test_loader_is_repeatable_and_preserves_answer_whitespace(self):
        rows = [
            dict(id=f"qa-{i}", question=f"Question {i}?", answer=f"Answer {i}")
            for i in range(20)
        ]
        rows[0]["answer"] = "  Answer 0  "
        session = self._session(rows)

        first = await load_cohort(session)
        second = await load_cohort(session)

        self.assertEqual(first, second)
        self.assertEqual(first[0].answer, "  Answer 0  ")
        self.assertIsInstance(first[0], QAExample)
        self.assertEqual(session.execute.await_count, 2)

    def test_qa_example_is_immutable(self):
        example = QAExample("qa-1", "Question?", "Answer")
        with self.assertRaises((AttributeError, TypeError)):
            example.answer = "Changed"


if __name__ == "__main__":
    unittest.main()
