import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from data import (
    build_negative_sampling_distribution,
    build_vocab,
    encode_tokens,
    generate_training_pairs,
    read_text,
    sample_negative_ids,
    tokenize,
)


class DataModuleTests(unittest.TestCase):
    def test_read_text_returns_file_contents(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.txt"
            path.write_text("Hello\nworld", encoding="utf-8")

            self.assertEqual(read_text(path), "Hello\nworld")

    def test_tokenize_honors_lower_flag(self):
        self.assertEqual(tokenize("Hello WoRLD"), ["hello", "world"])
        self.assertEqual(tokenize("Hello WoRLD", lower=False), ["Hello", "WoRLD"])

    def test_build_vocab_applies_min_count_and_deterministic_order(self):
        tokens = ["pear", "apple", "banana", "apple", "pear", "pear"]

        word_to_index, index_to_word, counts = build_vocab(tokens, min_count=2)

        expected_word_to_index = {"pear": 0, "apple": 1}
        self.assertEqual(word_to_index, expected_word_to_index)
        self.assertEqual(index_to_word, {0: "pear", 1: "apple"})
        assert_array_equal(counts, np.array([3, 2], dtype=np.int64))

    def test_encode_tokens_filters_out_of_vocab_items(self):
        encoded = encode_tokens(
            ["pear", "banana", "apple", "banana"],
            {"pear": 0, "apple": 1},
        )

        self.assertEqual(encoded, [0, 1])

    def test_generate_training_pairs_uses_symmetric_window(self):
        pairs = generate_training_pairs([0, 1, 2, 3], window_size=1)

        self.assertEqual(
            pairs,
            [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)],
        )

    def test_build_negative_sampling_distribution_normalizes_powered_counts(self):
        counts = np.array([1, 4, 9], dtype=np.int64)

        distribution = build_negative_sampling_distribution(counts, power=0.5)

        expected = np.array([1, 2, 3], dtype=np.float64) / 6.0
        assert_allclose(distribution, expected)
        self.assertAlmostEqual(distribution.sum(), 1.0)

    def test_sample_negative_ids_respects_forbidden_indices(self):
        rng = np.random.default_rng(0)
        neg_probs = np.array([0.1, 0.7, 0.2], dtype=np.float64)

        negative_ids = sample_negative_ids(
            rng,
            neg_probs,
            k=6,
            forbidden_indices={1},
        )

        self.assertEqual(negative_ids.dtype, np.int64)
        self.assertEqual(len(negative_ids), 6)
        self.assertTrue(np.all(negative_ids != 1))


if __name__ == "__main__":
    unittest.main()
