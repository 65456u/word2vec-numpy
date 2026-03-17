from __future__ import annotations

import numpy as np

from word2vec.architectures.base import Word2VecArchitecture
from word2vec.model import Word2Vec


class SkipGramArchitecture(Word2VecArchitecture[tuple[int, int], int]):
    def forward(self, model: Word2Vec, example: tuple[int, int]) -> tuple[np.ndarray, int, int]:
        center_id, target_id = example
        hidden = model.get_input_embedding(center_id).copy()
        return hidden, target_id, center_id

    def backward(
        self,
        model: Word2Vec,
        cache: int,
        hidden_gradient: np.ndarray,
        learning_rate: float,
    ) -> None:
        center_id = cache
        model.input_embeddings[center_id] -= learning_rate * hidden_gradient
