from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from word2vec.model import Word2Vec


class Word2VecObjective(ABC):
    def initialize(self, model: Word2Vec) -> None:
        """
        Hook for objective-specific preprocessing before training.
        """

    @abstractmethod
    def train_step(
        self,
        model: Word2Vec,
        hidden: np.ndarray,
        target_id: int,
        learning_rate: float,
    ) -> tuple[float, np.ndarray]:
        """
        Applies one SGD update for the objective and returns loss and hidden-state gradient.
        """
