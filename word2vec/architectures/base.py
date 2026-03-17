from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from word2vec.model import Word2Vec


ExampleT = TypeVar("ExampleT")
CacheT = TypeVar("CacheT")


class Word2VecArchitecture(ABC, Generic[ExampleT, CacheT]):
    @abstractmethod
    def forward(self, model: Word2Vec, example: ExampleT) -> tuple[np.ndarray, int, CacheT]:
        """
        Builds the hidden representation used by the objective and returns the target id.
        """

    @abstractmethod
    def backward(
        self,
        model: Word2Vec,
        cache: CacheT,
        hidden_gradient: np.ndarray,
        learning_rate: float,
    ) -> None:
        """
        Applies the hidden-state gradient to the architecture-specific input embeddings.
        """
