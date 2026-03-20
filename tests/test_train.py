import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from train import create_batches, sample_negative_matrix, train_epoch


class TrainModuleTests(unittest.TestCase):
    def test_create_batches_preserves_order_without_shuffle(self):
        pairs = [(0, 1), (1, 2), (2, 3)]

        batches = list(create_batches(pairs, batch_size=2, shuffle=False))

        self.assertEqual(len(batches), 2)
        assert_array_equal(batches[0][0], np.array([0, 1], dtype=np.int64))
        assert_array_equal(batches[0][1], np.array([1, 2], dtype=np.int64))
        assert_array_equal(batches[1][0], np.array([2], dtype=np.int64))
        assert_array_equal(batches[1][1], np.array([3], dtype=np.int64))

    def test_create_batches_uses_rng_for_deterministic_shuffle(self):
        pairs = [(0, 10), (1, 11), (2, 12), (3, 13)]
        rng = np.random.default_rng(9)

        actual_batches = list(create_batches(pairs, batch_size=2, shuffle=True, rng=rng))

        expected_indices = np.arange(len(pairs))
        np.random.default_rng(9).shuffle(expected_indices)
        expected_pairs = [pairs[idx] for idx in expected_indices]

        assert_array_equal(
            actual_batches[0][0],
            np.array([expected_pairs[0][0], expected_pairs[1][0]], dtype=np.int64),
        )
        assert_array_equal(
            actual_batches[0][1],
            np.array([expected_pairs[0][1], expected_pairs[1][1]], dtype=np.int64),
        )
        assert_array_equal(
            actual_batches[1][0],
            np.array([expected_pairs[2][0], expected_pairs[3][0]], dtype=np.int64),
        )
        assert_array_equal(
            actual_batches[1][1],
            np.array([expected_pairs[2][1], expected_pairs[3][1]], dtype=np.int64),
        )

    def test_sample_negative_matrix_matches_batch_shape_and_excludes_context_ids(self):
        rng = np.random.default_rng(0)
        neg_probs = np.array([0.2, 0.5, 0.3], dtype=np.float64)
        context_ids = np.array([1, 2], dtype=np.int64)

        negative_matrix = sample_negative_matrix(
            rng,
            neg_probs,
            context_ids,
            num_negatives=5,
        )

        self.assertEqual(negative_matrix.shape, (2, 5))
        self.assertTrue(np.all(negative_matrix[0] != 1))
        self.assertTrue(np.all(negative_matrix[1] != 2))

    @patch("train.sample_negative_matrix")
    @patch("train.train_batch")
    def test_train_epoch_returns_average_batch_loss(self, mock_train_batch, mock_sample_negative_matrix):
        pairs = [(0, 1), (1, 2), (2, 0)]
        w_in = np.zeros((3, 2), dtype=np.float32)
        w_out = np.zeros((3, 2), dtype=np.float32)
        neg_probs = np.array([0.2, 0.3, 0.5], dtype=np.float64)

        mock_sample_negative_matrix.side_effect = [
            np.zeros((2, 2), dtype=np.int64),
            np.zeros((1, 2), dtype=np.int64),
        ]
        mock_train_batch.side_effect = [1.0, 3.0]

        avg_loss = train_epoch(
            pairs,
            w_in,
            w_out,
            neg_probs,
            batch_size=2,
            num_negatives=2,
            lr=0.05,
            rng=np.random.default_rng(1),
        )

        self.assertEqual(avg_loss, 2.0)
        self.assertEqual(mock_train_batch.call_count, 2)

    def test_train_epoch_returns_zero_for_empty_pair_list(self):
        avg_loss = train_epoch(
            pairs=[],
            w_in=np.zeros((2, 2), dtype=np.float32),
            w_out=np.zeros((2, 2), dtype=np.float32),
            neg_probs=np.array([0.5, 0.5], dtype=np.float64),
            batch_size=2,
            num_negatives=2,
            lr=0.05,
            rng=np.random.default_rng(0),
        )

        self.assertEqual(avg_loss, 0.0)


if __name__ == "__main__":
    unittest.main()
