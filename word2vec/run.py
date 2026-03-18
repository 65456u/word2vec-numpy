from __future__ import annotations

import argparse

from word2vec.architectures import CBOWArchitecture, SkipGramArchitecture
from word2vec.dataset import generate_cbow_examples, generate_skipgram_pairs
from word2vec.eval import most_similar
from word2vec.model import Word2Vec
from word2vec.objectives import (
    HierarchicalSoftmaxObjective,
    NCEObjective,
    NegativeSamplingObjective,
)
from word2vec.preprocessing import build_vocab, encode_tokens, read_text8, split_tokens
from word2vec.trainer import Word2VecTrainer, Word2VecTrainingConfig


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a NumPy Word2Vec model.")
    parser.add_argument("--path", required=True, help="Path to the training corpus.")
    parser.add_argument("--architecture", choices=("skipgram", "cbow"), default="skipgram")
    parser.add_argument("--objective", choices=("neg", "hs", "nce"), default="neg")
    parser.add_argument("--embedding-dim", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--num-negative", type=int, default=5)
    parser.add_argument("--query-word", type=str, default=None)
    return parser


def create_architecture(name: str):
    if name == "skipgram":
        return SkipGramArchitecture(), generate_skipgram_pairs
    return CBOWArchitecture(), generate_cbow_examples


def create_objective(name: str, counts, num_negative: int):
    if name == "neg":
        return NegativeSamplingObjective(counts=counts, num_negative=num_negative)
    if name == "hs":
        return HierarchicalSoftmaxObjective(counts=counts)
    return NCEObjective(counts=counts, num_noise=num_negative)


def main() -> None:
    args = build_argument_parser().parse_args()

    corpus = read_text8(args.path)
    tokens = split_tokens(corpus)
    word_to_index, index_to_word, _, counts = build_vocab(tokens, min_count=args.min_count)
    token_ids = encode_tokens(tokens, word_to_index)

    architecture, example_builder = create_architecture(args.architecture)
    examples = example_builder(token_ids, args.window_size)

    model = Word2Vec(vocab_size=len(word_to_index), embedding_dim=args.embedding_dim)
    objective = create_objective(args.objective, counts, args.num_negative)
    trainer = Word2VecTrainer(model=model, architecture=architecture, objective=objective)

    config = Word2VecTrainingConfig(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
    )
    losses = trainer.fit(examples, config)

    print("Epoch losses:", losses)

    if args.query_word is not None and args.query_word in word_to_index:
        query_id = word_to_index[args.query_word]
        neighbors = most_similar(model, query_id)
        print(f"Most similar to '{args.query_word}':")
        for neighbor_id, score in neighbors:
            print(index_to_word[neighbor_id], score)


if __name__ == "__main__":
    main()
