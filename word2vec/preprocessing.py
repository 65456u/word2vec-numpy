import numpy as np
from collections import Counter

def read_text8(path: str) -> str:
    """
    Reads the contents of a (text8) file at the specified path and returns it as a string.
    Args:
        path (str): The file path to the text8 dataset.
    Returns:
        str: The entire contents of the file as a single string.
    """
    with open(path, "r") as f:
        return f.read()


def split_tokens(corpus: str) -> list[str]:
    """
    Splits the input corpus string into a list of tokens (words).
    Args:
        corpus (str): The input text corpus as a single string.
    Returns:
        list[str]: A list of individual tokens (words) extracted from the corpus.
    """
    return corpus.split()


def build_vocab(
    tokens: list[str], min_count: int = 0
) -> tuple[dict[str, int], dict[int, str], np.ndarray]:
    """
    Builds a vocabulary from a list of tokens, filtering by minimum count.
    Args:
        tokens (list[str]): A list of individual tokens (words).
        min_count (int): The minimum frequency a token must have to be included in the vocabulary.
    Returns:
        tuple[dict[str, int], dict[int, str], np.ndarray]: A tuple containing:
            - word_to_index (dict[str, int]): A mapping from words to their corresponding indices.
            - index_to_word (dict[int, str]): A mapping from indices back to their corresponding words.
            - vocab_array (np.ndarray): An array of the unique words in the vocabulary.
    """
    token_counts = Counter(tokens)
    vocab = {word: count for word, count in token_counts.items() if count >= min_count}
    word_to_index = {word: idx for idx, word in enumerate(vocab.keys())}
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    vocab_array = np.array(list(vocab.keys()))
    return word_to_index, index_to_word, vocab_array
