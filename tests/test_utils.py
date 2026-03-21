import random
import unittest

import numpy as np
from numpy.testing import assert_allclose

from utils import cosine_similarity, log_sigmoid, nearest_neighbors, set_seed, sigmoid


class UtilsModuleTests(unittest.TestCase):
    def test_sigmoid_and_log_sigmoid_handle_large_inputs(self):
        values = np.array([-1000.0, 0.0, 1000.0], dtype=np.float64)

        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            sigmoid_values = sigmoid(values)
            log_sigmoid_values = log_sigmoid(values)

        assert_allclose(sigmoid_values, np.array([0.0, 0.5, 1.0]), atol=1e-12)
        self.assertTrue(np.isfinite(log_sigmoid_values).all())
        assert_allclose(
            log_sigmoid_values, np.array([-1000.0, -np.log(2.0), 0.0]), atol=1e-10
        )

    def test_cosine_similarity_returns_zero_for_zero_vector(self):
        similarity = cosine_similarity(np.array([0.0, 0.0]), np.array([2.0, 3.0]))
        self.assertEqual(similarity, 0.0)

    def test_nearest_neighbors_returns_sorted_matches(self):
        word_to_id = {"king": 0, "queen": 1, "man": 2, "apple": 3}
        id_to_word = {idx: word for word, idx in word_to_id.items()}
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.8, 0.2],
                [-1.0, 0.0],
            ],
            dtype=np.float64,
        )

        neighbors = nearest_neighbors(
            "king", word_to_id, id_to_word, embeddings, top_k=2
        )

        self.assertEqual([word for word, _ in neighbors], ["queen", "man"])
        self.assertGreater(neighbors[0][1], neighbors[1][1])

    def test_nearest_neighbors_returns_empty_for_unknown_word(self):
        neighbors = nearest_neighbors(
            "unknown",
            {"known": 0},
            {0: "known"},
            np.array([[1.0, 0.0]], dtype=np.float64),
        )

        self.assertEqual(neighbors, [])

    def test_set_seed_makes_numpy_and_random_reproducible(self):
        set_seed(123)
        first_numpy = np.random.rand(3)
        first_random = random.random()

        set_seed(123)
        second_numpy = np.random.rand(3)
        second_random = random.random()

        assert_allclose(first_numpy, second_numpy)
        self.assertEqual(first_random, second_random)


if __name__ == "__main__":
    unittest.main()
