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

## Implemented Variant

This repository implements the **skip-gram with negative sampling (SGNS)** variant of Word2Vec in pure NumPy.

More specifically:

- **Architecture:** skip-gram, where the model predicts surrounding context words from a center word.
- **Training objective:** negative sampling rather than full softmax or hierarchical softmax.
- **Parameters:** separate input and output embedding matrices (`w_in` and `w_out`) are learned during training.

In addition to the core SGNS formulation, the training pipeline includes several standard Word2Vec training techniques:

- **Dynamic window size:** each token uses a randomly sampled context window up to the configured maximum window size.
- **Frequent-word subsampling:** very common tokens are downsampled before training pair generation.
- **Negative sampling distribution:** negative words are sampled from the unigram distribution raised to the `0.75` power.

## Quick Start

1. Install dependencies with `pip install -r requirements.txt`.
2. Download the dataset from <https://www.mattmahoney.net/dc/text8.zip>, unzip it, and place the `text8` corpus at `data/text8`.
3. Run training:

```bash
python train.py
```

4. Inspect or export embeddings from a checkpoint:

```bash
python embed.py --checkpoint-dir artifacts/run_YYYYMMDD_HHMMSS_01 --word king
python embed.py --checkpoint-dir artifacts/run_YYYYMMDD_HHMMSS_01 --words king queen man woman
python embed.py --checkpoint-dir artifacts/run_YYYYMMDD_HHMMSS_01 --analogy man king woman
python embed.py --checkpoint-dir artifacts/run_YYYYMMDD_HHMMSS_01 --export artifacts/embeddings.tsv
```

By default, each training run creates a new subdirectory under `artifacts/` and saves:

- `model_epoch_XXX.npz` and `config_epoch_XXX.json` after each epoch
- `model.npz` and `config.json` as the final checkpoint for that run

## Project Structure

- `train.py`: training entry point, batching logic, learning-rate schedule, epoch loop, and checkpoint saving.
- `embed.py`: checkpoint loader for querying word vectors, nearest neighbors, and exporting embeddings.
- `word2vec.py`: SGNS parameter initialization, forward pass, loss, gradients, and embedding updates.
- `data.py`: corpus reading, tokenization, vocabulary construction, subsampling, training-pair generation, and negative-sampling utilities.
- `utils.py`: numerical helpers such as sigmoid/log-sigmoid, cosine similarity, nearest-neighbor lookup, and seed control.
- `tests/`: unit tests for data processing, training utilities, numerical helpers, and Word2Vec core logic.
- `artifacts/`: saved training runs and checkpoints.

## References

- Mikolov, T., Chen, K., Corrado, G., and Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. <https://arxiv.org/abs/1301.3781>
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., and Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS 2013. <https://proceedings.neurips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf>
