from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class UnigramSampler:
    probabilities: np.ndarray
    rng: np.random.Generator

    @classmethod
    def from_counts(
        cls, counts: np.ndarray, power: float = 0.75, seed: int = 42
    ) -> "UnigramSampler":
        adjusted = counts.astype(np.float64) ** power
        probabilities = adjusted / adjusted.sum()
        return cls(probabilities=probabilities, rng=np.random.default_rng(seed))

    def sample(self, num_samples: int) -> np.ndarray:
        """
        Draws token ids from the configured unigram distribution.
        """
        vocabulary = np.arange(self.probabilities.shape[0])
        return self.rng.choice(vocabulary, size=num_samples, p=self.probabilities)

    def sample_one(self, forbidden_ids: set[int] | None = None) -> int:
        """
        Draws a single token id, retrying until it is not forbidden.
        """
        forbidden_ids = forbidden_ids or set()

        while True:
            sample = int(self.sample(1)[0])
            if sample not in forbidden_ids:
                return sample

    def sample_negative(self, positive_id: int, num_samples: int) -> np.ndarray:
        """
        Draws negative samples while excluding the positive target id.
        """
        negatives = np.empty(num_samples, dtype=np.int64)
        forbidden_ids = {positive_id}

        for index in range(num_samples):
            negatives[index] = self.sample_one(forbidden_ids=forbidden_ids)

        return negatives
