# Numpy Based Word2Vec Implementation

## Task

Implement the core training loop of word2vec in pure NumPy (no PyTorch / TensorFlow or other ML frameworks). The applicant is free to choose any suitable text dataset. The task is to implement the optimization procedure (forward pass, loss, gradients, and parameter updates) for a standard word2vec variant (e.g. skip-gram with negative sampling or CBOW).

The submitted solution should be fully understood by the applicant: during follow-up we will ask questions about the ideas behind word2vec, the code, gradient derivations, and possible alternative implementations or optimizations.
Preferably, solutions should be provided as a link to a public GitHub repository.

## Dataset

We use [text8](https://www.mattmahoney.net/dc/textdata.html) as our dataset, which is a cleaned version of the first 100MB of Wikipedia text. It contains approximately 17 million words and is commonly used for training word embedding models.
