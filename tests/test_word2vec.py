import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from word2vec import (
    backward_skipgram_negative_sampling,
    compute_sgns_loss,
    forward_skipgram_negative_sampling,
    init_parameters,
    train_batch,
)


class Word2VecModuleTests(unittest.TestCase):
    def setUp(self):
        self.center_ids = np.array([0, 2], dtype=np.int64)
        self.context_ids = np.array([1, 0], dtype=np.int64)
        self.negative_ids = np.array([[2, 2], [1, 2]], dtype=np.int64)
        self.w_in = np.array(
            [
                [0.20, -0.10],
                [0.00, 0.30],
                [-0.40, 0.25],
            ],
            dtype=np.float64,
        )
        self.w_out = np.array(
            [
                [0.30, 0.10],
                [-0.20, 0.40],
                [0.50, -0.30],
            ],
            dtype=np.float64,
        )

    def _loss(self, w_in, w_out):
        cache = forward_skipgram_negative_sampling(
            self.center_ids,
            self.context_ids,
            self.negative_ids,
            w_in,
            w_out,
        )
        return compute_sgns_loss(cache["pos_scores"], cache["neg_scores"])

    def _numerical_gradient(self, parameter_name, epsilon=1e-6):
        base = self.w_in if parameter_name == "w_in" else self.w_out
        numerical_grad = np.zeros_like(base)

        for index in np.ndindex(base.shape):
            plus = base.copy()
            minus = base.copy()
            plus[index] += epsilon
            minus[index] -= epsilon

            if parameter_name == "w_in":
                loss_plus = self._loss(plus, self.w_out)
                loss_minus = self._loss(minus, self.w_out)
            else:
                loss_plus = self._loss(self.w_in, plus)
                loss_minus = self._loss(self.w_in, minus)

            numerical_grad[index] = (loss_plus - loss_minus) / (2.0 * epsilon)

        return numerical_grad

    def test_init_parameters_shapes_ranges_and_dtypes(self):
        rng = np.random.default_rng(7)

        w_in, w_out = init_parameters(vocab_size=4, embed_dim=3, rng=rng)

        self.assertEqual(w_in.shape, (4, 3))
        self.assertEqual(w_out.shape, (4, 3))
        self.assertEqual(w_in.dtype, np.float32)
        self.assertEqual(w_out.dtype, np.float32)
        self.assertTrue(np.all(w_out == 0.0))
        self.assertTrue(np.all(w_in <= (0.5 / 3)))
        self.assertTrue(np.all(w_in >= (-0.5 / 3)))

    def test_forward_scores_and_loss_match_manual_computation(self):
        cache = forward_skipgram_negative_sampling(
            self.center_ids,
            self.context_ids,
            self.negative_ids,
            self.w_in,
            self.w_out,
        )

        expected_pos_scores = np.array([-0.08, -0.095], dtype=np.float64)
        expected_neg_scores = np.array([[0.13, 0.13], [0.18, -0.275]], dtype=np.float64)

        assert_array_equal(cache["center_embeds"], self.w_in[self.center_ids])
        assert_array_equal(cache["context_embeds"], self.w_out[self.context_ids])
        assert_array_equal(cache["negative_embeds"], self.w_out[self.negative_ids])
        assert_allclose(cache["pos_scores"], expected_pos_scores)
        assert_allclose(cache["neg_scores"], expected_neg_scores)

        loss = compute_sgns_loss(cache["pos_scores"], cache["neg_scores"])
        manual_loss = np.mean(
            -np.log(1.0 / (1.0 + np.exp(-expected_pos_scores)))
            - np.sum(np.log(1.0 / (1.0 + np.exp(expected_neg_scores))), axis=1)
        )
        self.assertAlmostEqual(loss, manual_loss)

    def test_backward_gradients_match_numerical_gradients(self):
        cache = forward_skipgram_negative_sampling(
            self.center_ids,
            self.context_ids,
            self.negative_ids,
            self.w_in,
            self.w_out,
        )
        grad_w_in, grad_w_out = backward_skipgram_negative_sampling(
            self.center_ids,
            self.context_ids,
            self.negative_ids,
            self.w_in,
            self.w_out,
            cache,
        )

        numerical_grad_w_in = self._numerical_gradient("w_in")
        numerical_grad_w_out = self._numerical_gradient("w_out")

        assert_allclose(grad_w_in, numerical_grad_w_in, rtol=1e-4, atol=1e-6)
        assert_allclose(grad_w_out, numerical_grad_w_out, rtol=1e-4, atol=1e-6)

    def test_train_batch_returns_preupdate_loss_and_improves_batch_objective(self):
        w_in = self.w_in.copy()
        w_out = self.w_out.copy()

        loss_before = self._loss(w_in, w_out)
        returned_loss = train_batch(
            self.center_ids,
            self.context_ids,
            self.negative_ids,
            w_in,
            w_out,
            lr=0.1,
        )
        loss_after = self._loss(w_in, w_out)

        self.assertAlmostEqual(returned_loss, loss_before)
        self.assertLess(loss_after, loss_before)


if __name__ == "__main__":
    unittest.main()
