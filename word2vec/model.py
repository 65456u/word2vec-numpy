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
        # Initialize input and output embeddings with small random values
        self.input_embeddings = np.random.rand(vocab_size, embedding_dim) * 0.01
        self.output_embeddings = np.random.rand(vocab_size, embedding_dim) * 0.01

    def get_embedding(self, word_id: int) -> np.ndarray:
        """
        Retrieves the embedding vector for a given word ID from the input embeddings.
        Args:
            word_id (int): The ID of the word for which to retrieve the embedding.
        Returns:
            np.ndarray: The embedding vector corresponding to the given word ID.
        """
        return self.input_embeddings[word_id]
    
    def forward(self, center_id: int, context_id: int) -> float:
        """
        Computes the dot product between the input embedding of the center word and the output embedding of the context word.
        Args:
            center_id (int): The ID of the center word.
            context_id (int): The ID of the context word.
        Returns:
            float: The dot product of the center word's input embedding and the context word's output embedding.
        """
        center_embedding = self.input_embeddings[center_id]
        context_embedding = self.output_embeddings[context_id]
        return np.dot(center_embedding, context_embedding)
    