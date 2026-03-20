import numpy as np
from collections import Counter


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def tokenize(corpus: str, lower: bool = True) -> list[str]:
    if lower:
        corpus = corpus.lower()
    return corpus.split()


def build_vocab(
    tokens: list[str], min_count: int = 5
) -> tuple[dict[str, int], dict[int, str], np.ndarray]:
    token_counts = Counter(tokens)
    vocab_items = sorted(
        [(word, count) for word, count in token_counts.items() if count >= min_count],
        key=lambda x: (-x[1], x[0]),
    )

    word_to_index = {word: idx for idx, (word, _) in enumerate(vocab_items)}
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    counts_array = np.array([count for _, count in vocab_items], dtype=np.int64)

    return word_to_index, index_to_word, counts_array


def encode_tokens(tokens: list[str], word_to_index: dict[str, int]) -> list[int]:
    return [word_to_index[token] for token in tokens if token in word_to_index]


def generate_training_pairs(
    token_indices: list[int], window_size: int
) -> list[tuple[int, int]]:
    training_pairs = []
    for i in range(len(token_indices)):
        center_index = token_indices[i]
        start = max(0, i - window_size)
        end = min(len(token_indices), i + window_size + 1)

        for j in range(start, end):
            if j == i:
                continue
            context_index = token_indices[j]
            training_pairs.append((center_index, context_index))

    return training_pairs


def build_negative_sampling_distribution(
    counts_array: np.ndarray, power: float = 0.75
) -> np.ndarray:
    adjusted_counts = counts_array.astype(np.float64) ** power
    return adjusted_counts / adjusted_counts.sum()


def sample_negative_ids(
    rng: np.random.Generator,
    neg_probs: np.ndarray,
    k: int,
    forbidden_indices: set[int] | None = None,
) -> np.ndarray:
    if forbidden_indices is None:
        forbidden_indices = set()

    negative_indices = []
    while len(negative_indices) < k:
        sampled_index = rng.choice(len(neg_probs), p=neg_probs)
        if sampled_index not in forbidden_indices:
            negative_indices.append(sampled_index)

    return np.array(negative_indices, dtype=np.int64)
