from __future__ import annotations

import numpy as np

from word2vec.model import Word2Vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def most_similar(
    model: Word2Vec, word_id: int, top_k: int = 5
) -> list[tuple[int, float]]:
    query = model.get_input_embedding(word_id)
    similarities: list[tuple[int, float]] = []

    for candidate_id in range(model.vocab_size):
        if candidate_id == word_id:
            continue
        score = cosine_similarity(query, model.get_input_embedding(candidate_id))
        similarities.append((candidate_id, score))

    similarities.sort(key=lambda item: item[1], reverse=True)
    return similarities[:top_k]
