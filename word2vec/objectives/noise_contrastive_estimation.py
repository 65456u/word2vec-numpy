from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from word2vec.model import Word2Vec
from word2vec.objectives.base import Word2VecObjective
from word2vec.samplers import UnigramSampler
from word2vec.utils import sigmoid


@dataclass(slots=True)
class NCEObjective(Word2VecObjective):
    counts: np.ndarray
    num_noise: int = 5
    power: float = 0.75
    seed: int = 42
    sampler: UnigramSampler | None = None
    log_z: float | None = None

    def initialize(self, model: Word2Vec) -> None:
        if self.sampler is None:
            self.sampler = UnigramSampler.from_counts(
                self.counts,
                power=self.power,
                seed=self.seed,
            )
        if self.log_z is None:
            self.log_z = float(np.log(model.vocab_size))

    def train_step(
        self,
        model: Word2Vec,
        hidden: np.ndarray,
        target_id: int,
        learning_rate: float,
    ) -> tuple[float, np.ndarray]:
        if self.sampler is None or self.log_z is None:
            raise RuntimeError("NCEObjective must be initialized before training.")

        noise_ids = self.sampler.sample_negative(target_id, self.num_noise)
        positive_output = model.output_embeddings[target_id].copy()
        noise_outputs = model.output_embeddings[noise_ids].copy()

        positive_model_score = float(np.dot(positive_output, hidden) - self.log_z)
        positive_noise_score = np.log(
            self.num_noise * self.sampler.probabilities[target_id] + 1e-12
        )
        positive_logit = positive_model_score - positive_noise_score
        positive_probability = float(sigmoid(positive_logit))

        noise_model_scores = noise_outputs @ hidden - self.log_z
        noise_noise_scores = np.log(
            self.num_noise * self.sampler.probabilities[noise_ids] + 1e-12
        )
        noise_logits = noise_model_scores - noise_noise_scores
        noise_probabilities = np.asarray(sigmoid(noise_logits))

        positive_output_gradient = (positive_probability - 1.0) * hidden
        noise_output_gradients = noise_probabilities[:, None] * hidden[None, :]

        hidden_gradient = (positive_probability - 1.0) * positive_output
        hidden_gradient += np.sum(noise_probabilities[:, None] * noise_outputs, axis=0)

        model.output_embeddings[target_id] -= learning_rate * positive_output_gradient
        np.add.at(
            model.output_embeddings,
            noise_ids,
            -learning_rate * noise_output_gradients,
        )

        epsilon = 1e-12
        loss = -np.log(positive_probability + epsilon)
        loss -= np.sum(np.log(1.0 - noise_probabilities + epsilon))
        return float(loss), hidden_gradient
