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
        # Define the context window boundaries
        start = max(0, i - window_size)
        end = min(len(token_ids), i + window_size + 1)
        # Generate pairs for the context words within the window
        for j in range(start, end):
            if j != i:  # Skip the center word itself
                context_id = token_ids[j]
                pairs.append((center_id, context_id))
    return pairs