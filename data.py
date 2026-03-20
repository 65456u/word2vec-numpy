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


def generate_training_pairs_array(
    token_indices: list[int] | np.ndarray, window_size: int
) -> np.ndarray:
    token_indices = np.asarray(token_indices, dtype=np.int32)
    num_tokens = token_indices.shape[0]

    if num_tokens == 0 or window_size < 1:
        return np.empty((0, 2), dtype=np.int32)

    total_pairs = 2 * sum(num_tokens - offset for offset in range(1, min(window_size, num_tokens - 1) + 1))
    training_pairs = np.empty((total_pairs, 2), dtype=np.int32)

    cursor = 0
    for offset in range(1, window_size + 1):
        span = num_tokens - offset
        if span <= 0:
            break

        next_cursor = cursor + span
        training_pairs[cursor:next_cursor, 0] = token_indices[offset:]
        training_pairs[cursor:next_cursor, 1] = token_indices[:-offset]
        cursor = next_cursor

        next_cursor = cursor + span
        training_pairs[cursor:next_cursor, 0] = token_indices[:-offset]
        training_pairs[cursor:next_cursor, 1] = token_indices[offset:]
        cursor = next_cursor

    return training_pairs


def count_training_pairs(num_tokens: int, window_size: int) -> int:
    if num_tokens <= 1 or window_size < 1:
        return 0

    max_offset = min(window_size, num_tokens - 1)
    return 2 * sum(num_tokens - offset for offset in range(1, max_offset + 1))


def stream_training_pair_chunks(
    token_indices: list[int] | np.ndarray,
    window_size: int,
    chunk_size: int,
):
    token_indices = np.asarray(token_indices, dtype=np.int32)
    num_tokens = token_indices.shape[0]

    if num_tokens == 0 or window_size < 1:
        return

    chunk_size = max(int(chunk_size), 1)

    for offset in range(1, window_size + 1):
        span = num_tokens - offset
        if span <= 0:
            break

        for start in range(0, span, chunk_size):
            end = min(span, start + chunk_size)

            forward_chunk = np.empty((end - start, 2), dtype=np.int32)
            forward_chunk[:, 0] = token_indices[start + offset : end + offset]
            forward_chunk[:, 1] = token_indices[start:end]
            yield forward_chunk

        for start in range(0, span, chunk_size):
            end = min(span, start + chunk_size)

            backward_chunk = np.empty((end - start, 2), dtype=np.int32)
            backward_chunk[:, 0] = token_indices[start:end]
            backward_chunk[:, 1] = token_indices[start + offset : end + offset]
            yield backward_chunk


def build_negative_sampling_distribution(
    counts_array: np.ndarray, power: float = 0.75
) -> np.ndarray:
    adjusted_counts = counts_array.astype(np.float64) ** power
    return adjusted_counts / adjusted_counts.sum()


def build_negative_sampling_cdf(
    counts_array: np.ndarray, power: float = 0.75
) -> np.ndarray:
    neg_cdf = np.cumsum(
        build_negative_sampling_distribution(counts_array, power=power)
    )
    if neg_cdf.size > 0:
        neg_cdf[-1] = 1.0
    return neg_cdf


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


def subsample_token_ids(
    token_ids: list[int] | np.ndarray,
    counts_array: np.ndarray,
    threshold: float,
    rng: np.random.Generator,
) -> np.ndarray:
    token_ids = np.asarray(token_ids, dtype=np.int32)

    if threshold <= 0.0 or token_ids.size == 0:
        return token_ids

    token_frequencies = counts_array.astype(np.float64)
    token_frequencies /= token_frequencies.sum()

    keep_probs = (np.sqrt(token_frequencies / threshold) + 1.0) * (
        threshold / token_frequencies
    )
    keep_probs = np.clip(keep_probs, 0.0, 1.0)

    keep_mask = rng.random(token_ids.shape[0]) < keep_probs[token_ids]
    return token_ids[keep_mask]
