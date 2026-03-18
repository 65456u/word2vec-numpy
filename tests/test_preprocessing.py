import unittest

import numpy as np

from word2vec.preprocessing import build_vocab, encode_tokens, split_tokens


class PreprocessingTests(unittest.TestCase):
    def test_build_vocab_returns_counts_aligned_with_indices(self) -> None:
        tokens = split_tokens("alpha beta alpha gamma beta alpha")

        word_to_index, index_to_word, vocab, counts = build_vocab(tokens, min_count=1)

        self.assertEqual(vocab.shape[0], 3)
        self.assertEqual(counts.shape[0], 3)

        for word, index in word_to_index.items():
            self.assertEqual(index_to_word[index], word)

        recovered_counts = {vocab[index]: int(counts[index]) for index in range(len(vocab))}
        self.assertEqual(recovered_counts["alpha"], 3)
        self.assertEqual(recovered_counts["beta"], 2)
        self.assertEqual(recovered_counts["gamma"], 1)

    def test_encode_tokens_skips_oov_tokens(self) -> None:
        tokens = ["alpha", "missing", "beta"]
        word_to_index = {"alpha": 0, "beta": 1}

        encoded = encode_tokens(tokens, word_to_index)

        self.assertEqual(encoded, [0, 1])


if __name__ == "__main__":
    unittest.main()
