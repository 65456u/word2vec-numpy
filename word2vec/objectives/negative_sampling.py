from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from word2vec.model import Word2Vec
from word2vec.objectives.base import Word2VecObjective
from word2vec.samplers import UnigramSampler
from word2vec.utils import sigmoid


@dataclass(slots=True)
class NegativeSamplingObjective(Word2VecObjective):
    counts: np.ndarray
    num_negative: int = 5
    power: float = 0.75
    seed: int = 42
    sampler: UnigramSampler | None = None

    def initialize(self, model: Word2Vec) -> None:
        if self.sampler is None:
            self.sampler = UnigramSampler.from_counts(
                self.counts,
                power=self.power,
                seed=self.seed,
            )

    def train_step(
        self,
        model: Word2Vec,
        hidden: np.ndarray,
        target_id: int,
        learning_rate: float,
    ) -> tuple[float, np.ndarray]:
        if self.sampler is None:
            raise RuntimeError("NegativeSamplingObjective must be initialized before training.")

        negative_ids = self.sampler.sample_negative(target_id, self.num_negative)

        positive_output = model.output_embeddings[target_id].copy()
        negative_outputs = model.output_embeddings[negative_ids].copy()

        positive_score = float(np.dot(positive_output, hidden))
        negative_scores = negative_outputs @ hidden

        positive_probability = float(sigmoid(positive_score))
        negative_probabilities = np.asarray(sigmoid(negative_scores))

        positive_output_gradient = (positive_probability - 1.0) * hidden
        negative_output_gradients = negative_probabilities[:, None] * hidden[None, :]

        hidden_gradient = (positive_probability - 1.0) * positive_output
        hidden_gradient += np.sum(
            negative_probabilities[:, None] * negative_outputs,
            axis=0,
        )

        model.output_embeddings[target_id] -= learning_rate * positive_output_gradient
        model.output_embeddings[negative_ids] -= learning_rate * negative_output_gradients

        epsilon = 1e-12
        loss = -np.log(positive_probability + epsilon)
        loss -= np.sum(np.log(1.0 - negative_probabilities + epsilon))

        return float(loss), hidden_gradient
