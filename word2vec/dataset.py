from __future__ import annotations


def generate_skipgram_pairs(token_ids: list[int], window_size: int) -> list[tuple[int, int]]:
    """
    Generates skip-gram pairs from a list of token IDs based on a specified window size.
    Args:
        token_ids (list[int]): A list of token IDs representing the encoded corpus.
        window_size (int): The size of the context window to consider for generating pairs.
    Returns:
        list[tuple[int, int]]: A list of tuples, where each tuple contains a center word ID and a context word ID.
    """
    pairs = []
    for i in range(len(token_ids)):
        center_id = token_ids[i]
        start = max(0, i - window_size)
        end = min(len(token_ids), i + window_size + 1)
        for j in range(start, end):
            if j != i:
                context_id = token_ids[j]
                pairs.append((center_id, context_id))
    return pairs


def iter_skipgram_pairs(
    token_ids: list[int], window_size: int
) -> list[tuple[int, int]]:
    """
    Alias for generate_skipgram_pairs to support a more explicit dataset API.
    """
    return generate_skipgram_pairs(token_ids, window_size)


def generate_cbow_examples(
    token_ids: list[int], window_size: int
) -> list[tuple[list[int], int]]:
    """
    Generates CBOW examples where surrounding context ids predict the center word.
    """
    examples: list[tuple[list[int], int]] = []

    for i in range(len(token_ids)):
        start = max(0, i - window_size)
        end = min(len(token_ids), i + window_size + 1)

        context_ids = [token_ids[j] for j in range(start, end) if j != i]
        if context_ids:
            target_id = token_ids[i]
            examples.append((context_ids, target_id))

    return examples


def iter_cbow_examples(
    token_ids: list[int], window_size: int
) -> list[tuple[list[int], int]]:
    """
    Alias for generate_cbow_examples to support a parallel dataset API.
    """
    return generate_cbow_examples(token_ids, window_size)
