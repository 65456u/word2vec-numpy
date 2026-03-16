# Numpy Based Word2Vec Implementation

## Overview

This project implements Word2Vec from scratch using only NumPy. The implementation follows the original Word2Vec formulations introduced in
[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
and
[Distributed Representations of Words and Phrases and their Compositionality](https://proceedings.neurips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf).

## Task

Implement the core training loop of word2vec in pure NumPy (no PyTorch / TensorFlow or other ML frameworks). The applicant is free to choose any suitable text dataset. The task is to implement the optimization procedure (forward pass, loss, gradients, and parameter updates) for a standard word2vec variant (e.g. skip-gram with negative sampling or CBOW).

The submitted solution should be fully understood by the applicant: during follow-up we will ask questions about the ideas behind word2vec, the code, gradient derivations, and possible alternative implementations or optimizations.
Preferably, solutions should be provided as a link to a public GitHub repository.

## Dataset

We use [text8](https://www.mattmahoney.net/dc/textdata.html) as our dataset, which is a cleaned version of the first 100MB of Wikipedia text. It contains approximately 17 million words and is commonly used for training word embedding models.

## References

- Mikolov, T., Chen, K., Corrado, G., and Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. <https://arxiv.org/abs/1301.3781>
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., and Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS 2013. <https://proceedings.neurips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf>
