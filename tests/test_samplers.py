import unittest

import numpy as np

from word2vec.samplers import UnigramSampler


class SamplerTests(unittest.TestCase):
    def test_negative_sampler_excludes_positive_id(self) -> None:
        sampler = UnigramSampler.from_counts(np.array([10, 5, 1]), seed=0)

        negatives = sampler.sample_negative(positive_id=1, num_samples=50)

        self.assertFalse(np.any(negatives == 1))


if __name__ == "__main__":
    unittest.main()
