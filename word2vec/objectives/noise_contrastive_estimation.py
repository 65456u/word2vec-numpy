from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from word2vec.model import Word2Vec
from word2vec.objectives.base import Word2VecObjective


@dataclass(slots=True)
class NCEObjective(Word2VecObjective):
    num_noise: int = 5

    def train_step(
        self,
        model: Word2Vec,
        hidden: np.ndarray,
        target_id: int,
        learning_rate: float,
    ) -> tuple[float, np.ndarray]:
        raise NotImplementedError("Noise-contrastive estimation is not implemented yet.")
