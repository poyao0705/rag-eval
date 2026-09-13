import unittest

from backend.scripts.utils.passage import (
    deduplicate_candidates,
    extract_context,
    normalize_text,
    normalize_title,
)


class PassageTransformationTests(unittest.TestCase):
    def test_normalization_canonicalizes_unicode_soft_hyphen_and_whitespace(self):
        self.assertEqual(
            normalize_text("  the\u00ad  ﬁnal\n report\t"),
            "the final report",
        )
        self.assertEqual(normalize_title("  Café  "), "café")

    def test_extract_context_pairs_title_and_sentence_blocks_by_position(self):
        passages = extract_context(
            {
                "title": ["A", "B"],
                "sentences": [["A one.", " A two."], ["B one."]],
            }
        )

        self.assertEqual([p.title for p in passages], ["A", "B"])
        self.assertEqual(passages[0].sentences, ("A one.", " A two."))
        self.assertEqual(passages[0].text, "A one. A two.")

    def test_same_title_different_text_is_not_deduplicated(self):
        passages = extract_context(
            {
                "title": ["A", "A"],
                "sentences": [["First."], ["Second."]],
            }
        )

        self.assertEqual(len(deduplicate_candidates(passages)), 2)

    def test_identical_normalized_passage_is_deduplicated(self):
        passages = extract_context(
            {
                "title": ["A", " A "],
                "sentences": [["First."], [" First. "]],
            }
        )

        self.assertEqual(len(deduplicate_candidates(passages)), 1)

    def test_mismatched_parallel_arrays_are_rejected(self):
        with self.assertRaises(ValueError):
            extract_context({"title": ["A"], "sentences": []})

    def test_empty_normalized_sentence_block_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_context({"title": ["A"], "sentences": [["  ", "\t"]]})

    def test_blank_sentences_do_not_change_canonical_text(self):
        with_blank = extract_context(
            {"title": ["A"], "sentences": [["One.", "  ", "Two."]]}
        )[0]
        without_blank = extract_context(
            {"title": ["A"], "sentences": [["One.", "Two."]]}
        )[0]

        self.assertEqual(with_blank.sentences, ("One.", "  ", "Two."))
        self.assertEqual(with_blank.text, "One. Two.")
        self.assertEqual(with_blank.identity, without_blank.identity)
