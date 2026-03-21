import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from data import generate_training_pairs_array, sample_dynamic_window_sizes
from train import compute_decayed_lr, create_batches, sample_negative_matrix, train_epoch


class TrainModuleTests(unittest.TestCase):
    def test_create_batches_preserves_order_without_shuffle(self):
        token_ids = np.array([0, 1, 2, 3], dtype=np.int32)

        batches = list(
            create_batches(
                token_ids,
                window_size=1,
                batch_size=2,
                shuffle=False,
            )
        )

        self.assertEqual(len(batches), 3)
        assert_array_equal(batches[0][0], np.array([1, 2], dtype=np.int32))
        assert_array_equal(batches[0][1], np.array([0, 1], dtype=np.int32))
        assert_array_equal(batches[1][0], np.array([3, 0], dtype=np.int32))
        assert_array_equal(batches[1][1], np.array([2, 1], dtype=np.int32))
        assert_array_equal(batches[2][0], np.array([1, 2], dtype=np.int32))
        assert_array_equal(batches[2][1], np.array([2, 3], dtype=np.int32))

    def test_create_batches_uses_rng_for_deterministic_shuffle(self):
        token_ids = np.array([0, 1, 2, 3], dtype=np.int32)
        rng = np.random.default_rng(9)

        actual_batches = list(
            create_batches(
                token_ids,
                window_size=1,
                batch_size=2,
                shuffle=True,
                rng=rng,
                shuffle_buffer_size=6,
            )
        )

        expected_pairs = generate_training_pairs_array(token_ids, window_size=1)
        expected_indices = np.arange(len(expected_pairs))
        np.random.default_rng(9).shuffle(expected_indices)
        expected_pairs = expected_pairs[expected_indices]

        assert_array_equal(
            actual_batches[0][0],
            np.array([expected_pairs[0][0], expected_pairs[1][0]], dtype=np.int32),
        )
        assert_array_equal(
            actual_batches[0][1],
            np.array([expected_pairs[0][1], expected_pairs[1][1]], dtype=np.int32),
        )
        assert_array_equal(
            actual_batches[1][0],
            np.array([expected_pairs[2][0], expected_pairs[3][0]], dtype=np.int32),
        )
        assert_array_equal(
            actual_batches[1][1],
            np.array([expected_pairs[2][1], expected_pairs[3][1]], dtype=np.int32),
        )
        assert_array_equal(
            actual_batches[2][0],
            np.array([expected_pairs[4][0], expected_pairs[5][0]], dtype=np.int32),
        )
        assert_array_equal(
            actual_batches[2][1],
            np.array([expected_pairs[4][1], expected_pairs[5][1]], dtype=np.int32),
        )

    def test_create_batches_honors_dynamic_window_sizes(self):
        token_ids = np.array([0, 1, 2, 3], dtype=np.int32)
        window_sizes = np.array([1, 2, 1, 2], dtype=np.int32)

        batches = list(
            create_batches(
                token_ids,
                window_size=2,
                batch_size=8,
                shuffle=False,
                dynamic_window_sizes=window_sizes,
            )
        )

        self.assertEqual(len(batches), 1)
        assert_array_equal(
            batches[0][0],
            np.array([1, 2, 3, 0, 1, 2, 3, 1], dtype=np.int32),
        )
        assert_array_equal(
            batches[0][1],
            np.array([0, 1, 2, 1, 2, 3, 1, 3], dtype=np.int32),
        )

    def test_create_batches_dynamic_windows_can_be_sampled_from_rng(self):
        rng = np.random.default_rng(5)

        window_sizes = sample_dynamic_window_sizes(6, 3, rng)

        self.assertTrue(np.all(window_sizes >= 1))
        self.assertTrue(np.all(window_sizes <= 3))

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

    def test_compute_decayed_lr_matches_word2vec_linear_schedule(self):
        self.assertAlmostEqual(compute_decayed_lr(0.05, 0, 8), 0.05)
        self.assertAlmostEqual(compute_decayed_lr(0.05, 4, 8), 0.05 * (1 - 4 / 9))
        self.assertAlmostEqual(compute_decayed_lr(0.05, 8, 8), 0.05 * (1 - 8 / 9))
        self.assertAlmostEqual(compute_decayed_lr(0.05, 9, 8), 0.05 * 1e-4)

    @patch("train.sample_negative_matrix")
    @patch("train.train_batch")
    def test_train_epoch_returns_average_batch_loss(
        self, mock_train_batch, mock_sample_negative_matrix
    ):
        token_ids = np.array([0, 1, 2], dtype=np.int32)
        w_in = np.zeros((3, 2), dtype=np.float32)
        w_out = np.zeros((3, 2), dtype=np.float32)
        neg_probs = np.array([0.2, 0.3, 0.5], dtype=np.float64)

        mock_sample_negative_matrix.side_effect = [
            np.zeros((2, 2), dtype=np.int64),
            np.zeros((2, 2), dtype=np.int64),
        ]
        mock_train_batch.side_effect = [1.0, 3.0]

        avg_loss = train_epoch(
            token_ids,
            1,
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
        self.assertAlmostEqual(mock_train_batch.call_args_list[0].args[-1], 0.05)
        self.assertAlmostEqual(
            mock_train_batch.call_args_list[1].args[-1],
            0.05 * (1 - 2 / 5),
        )

    @patch("train.sample_negative_matrix")
    @patch("train.train_batch")
    def test_train_epoch_uses_global_progress_for_lr_decay(
        self, mock_train_batch, mock_sample_negative_matrix
    ):
        token_ids = np.array([0, 1, 2], dtype=np.int32)
        w_in = np.zeros((3, 2), dtype=np.float32)
        w_out = np.zeros((3, 2), dtype=np.float32)
        neg_probs = np.array([0.2, 0.3, 0.5], dtype=np.float64)

        mock_sample_negative_matrix.side_effect = [
            np.zeros((2, 2), dtype=np.int64),
            np.zeros((2, 2), dtype=np.int64),
        ]
        mock_train_batch.side_effect = [1.0, 1.0]

        train_epoch(
            token_ids,
            1,
            w_in,
            w_out,
            neg_probs,
            batch_size=2,
            num_negatives=2,
            lr=0.05,
            rng=np.random.default_rng(1),
            total_training_pairs=8,
            pairs_processed_before_epoch=4,
        )

        self.assertAlmostEqual(
            mock_train_batch.call_args_list[0].args[-1],
            0.05 * (1 - 4 / 9),
        )
        self.assertAlmostEqual(
            mock_train_batch.call_args_list[1].args[-1],
            0.05 * (1 - 6 / 9),
        )

    def test_train_epoch_returns_zero_for_empty_pair_list(self):
        avg_loss = train_epoch(
            token_ids=[],
            window_size=1,
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
