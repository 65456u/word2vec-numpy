from __future__ import annotations

import numpy as np

from word2vec.architectures.base import Word2VecArchitecture
from word2vec.model import Word2Vec


class CBOWArchitecture(Word2VecArchitecture[tuple[list[int], int], tuple[np.ndarray, int]]):
    def forward(
        self, model: Word2Vec, example: tuple[list[int], int]
    ) -> tuple[np.ndarray, int, tuple[np.ndarray, int]]:
        context_ids, target_id = example
        context_index_array = np.asarray(context_ids, dtype=np.int64)
        hidden = model.input_embeddings[context_index_array].mean(axis=0)
        return hidden, target_id, (context_index_array, len(context_ids))

    def backward(
        self,
        model: Word2Vec,
        cache: tuple[np.ndarray, int],
        hidden_gradient: np.ndarray,
        learning_rate: float,
    ) -> None:
        context_index_array, context_size = cache
        scaled_gradient = hidden_gradient / context_size
        model.input_embeddings[context_index_array] -= learning_rate * scaled_gradient
