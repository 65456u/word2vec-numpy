import numpy as np


class Word2Vec:
    def __init__(self, vocab_size: int, embedding_dim: int, seed: int = 42):
        """
        Initializes the Word2Vec model with random embeddings for the input and output layers.
        Args:
            vocab_size (int): The size of the vocabulary (number of unique words).
            embedding_dim (int): The dimensionality of the word embeddings.
            seed (int): A random seed for reproducibility of the initial embeddings.
        """
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_embeddings = np.random.rand(vocab_size, embedding_dim) * 0.01
        self.output_embeddings = np.random.rand(vocab_size, embedding_dim) * 0.01

    def get_input_embedding(self, word_id: int) -> np.ndarray:
        """
        Retrieves the input embedding vector for a given word ID.
        Args:
            word_id (int): The ID of the word for which to retrieve the embedding.
        Returns:
            np.ndarray: The embedding vector corresponding to the given word ID.
        """
        return self.input_embeddings[word_id]

    def get_output_embedding(self, word_id: int) -> np.ndarray:
        """
        Retrieves the output embedding vector for a given word ID.
        Args:
            word_id (int): The ID of the word for which to retrieve the embedding.
        Returns:
            np.ndarray: The output embedding vector corresponding to the given word ID.
        """
        return self.output_embeddings[word_id]

    def score(self, center_id: int, target_id: int) -> float:
        """
        Computes the dot product between a center word embedding and a target word embedding.
        Args:
            center_id (int): The ID of the center word.
            target_id (int): The ID of the target word.
        Returns:
            float: The compatibility score for the center-target pair.
        """
        center_embedding = self.input_embeddings[center_id]
        target_embedding = self.output_embeddings[target_id]
        return float(np.dot(center_embedding, target_embedding))

    def score_hidden_to_output(self, hidden: np.ndarray, output_id: int) -> float:
        """
        Computes the score between an arbitrary hidden representation and an output row.
        """
        return float(np.dot(hidden, self.output_embeddings[output_id]))

    def ensure_output_capacity(self, size: int) -> None:
        """
        Grows the output embedding matrix to the requested number of rows.
        """
        current_size = self.output_embeddings.shape[0]
        if size <= current_size:
            return

        extra_rows = np.random.rand(size - current_size, self.embedding_dim) * 0.01
        self.output_embeddings = np.vstack([self.output_embeddings, extra_rows])

    def get_embedding(self, word_id: int) -> np.ndarray:
        """
        Backwards-compatible alias for the input embedding lookup.
        """
        return self.get_input_embedding(word_id)

    def forward(self, center_id: int, context_id: int) -> float:
        """
        Backwards-compatible alias for score().
        """
        return self.score(center_id, context_id)
