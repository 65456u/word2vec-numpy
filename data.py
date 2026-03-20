from collections import Counter
import numpy as np
def load_text8(path: str) -> str:
    """
    Loads the text8 dataset from the specified file path.
    Args:
        path (str): The file path to the text8 dataset.
    Returns:
        str: The entire contents of the text8 file as a single string.
    """
    with open(path, "r") as f:
        return f.read()
    
def tokenize(corpus: str, lower: bool = True) -> list[str]:
    """
    Tokenizes the input corpus string into a list of individual tokens (words).
    Args:
        corpus (str): The input text corpus as a single string.
        lower (bool): Whether to convert tokens to lowercase.
    Returns:
        list[str]: A list of individual tokens (words) extracted from the corpus.
    """
    if lower:
        corpus = corpus.lower()
    return corpus.split()

def build_vocabulary(tokens: list[str], min_count: int = 5) -> tuple[dict[str, int], dict[int, str], np.ndarray, np.ndarray]:
    """
    Builds a vocabulary from a list of tokens, filtering out tokens that occur less than a specified minimum count.
    Args:
        tokens (list[str]): A list of individual tokens (words).
        min_count (int): The minimum frequency a token must have to be included in the vocabulary.
    Returns:
        tuple[dict[str, int], dict[int, str], np.ndarray, np.ndarray]: A tuple containing:
            - word_to_index (dict[str, int]): A mapping from words to their corresponding indices.
            - index_to_word (dict[int, str]): A mapping from indices back to their corresponding words.
            - vocab_array (np.ndarray): An array of the unique words in the vocabulary.
            - counts_array (np.ndarray): An array of token counts aligned with vocab_array / word indices.
    """
    token_counts = Counter(tokens)
    vocab_items = [(word, count) for word, count in token_counts.items() if count >= min_count]
    word_to_index = {word: idx for idx, (word, _) in enumerate(vocab_items)}
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    vocab_array = np.array([word for word, _ in vocab_items])
    counts_array = np.array([count for _, count in vocab_items], dtype=np.int64)
    return word_to_index, index_to_word, vocab_array, counts_array
