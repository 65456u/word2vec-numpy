import argparse
import json
from pathlib import Path

import numpy as np

from data import (
    build_negative_sampling_distribution,
    build_vocab,
    encode_tokens,
    generate_training_pairs,
    read_text,
    sample_negative_ids,
    tokenize,
)
from utils import nearest_neighbors
from word2vec import init_parameters, train_batch


def create_batches(pairs, batch_size, shuffle=True, rng=None):
    indices = np.arange(len(pairs))
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        batch_pairs = [pairs[idx] for idx in batch_indices]
        center_ids = np.array([pair[0] for pair in batch_pairs], dtype=np.int64)
        context_ids = np.array([pair[1] for pair in batch_pairs], dtype=np.int64)
        yield center_ids, context_ids


def sample_negative_matrix(rng, neg_probs, batch_context_ids, num_negatives):
    negative_ids = []

    for context_id in batch_context_ids:
        neg_ids = sample_negative_ids(
            rng, neg_probs, k=num_negatives, forbidden_indices={int(context_id)}
        )
        negative_ids.append(neg_ids)

    return np.array(negative_ids, dtype=np.int64)


def train_epoch(pairs, w_in, w_out, neg_probs, batch_size, num_negatives, lr, rng):
    total_loss = 0.0
    num_batches = 0

    for center_ids, context_ids in create_batches(
        pairs, batch_size, shuffle=True, rng=rng
    ):
        negative_ids = sample_negative_matrix(
            rng, neg_probs, context_ids, num_negatives
        )

        loss = train_batch(center_ids, context_ids, negative_ids, w_in, w_out, lr)
        total_loss += loss
        num_batches += 1

    return total_loss / max(num_batches, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train skip-gram word2vec with negative sampling in NumPy"
    )
    parser.add_argument("--data_path", type=str, default="data/text8")
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--embed_dim", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_negatives", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="artifacts")
    return parser.parse_args()


def save_checkpoint(save_dir, w_in, w_out, word_to_id, id_to_word, config):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model_path = save_path / "model.npz"
    config_path = save_path / "config.json"

    np.savez(model_path, w_in=w_in, w_out=w_out)

    vocab = [id_to_word[i] for i in range(len(id_to_word))]
    payload = {
        "vocab": vocab,
        "word_to_id": word_to_id,
        "config": config,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    text = read_text(args.data_path)
    tokens = tokenize(text)
    word_to_id, id_to_word, counts = build_vocab(tokens, min_count=args.min_count)
    token_ids = encode_tokens(tokens, word_to_id)
    pairs = generate_training_pairs(token_ids, window_size=args.window_size)
    neg_probs = build_negative_sampling_distribution(counts)

    vocab_size = len(word_to_id)
    w_in, w_out = init_parameters(vocab_size, args.embed_dim, rng)

    for epoch in range(args.epochs):
        avg_loss = train_epoch(
            pairs,
            w_in,
            w_out,
            neg_probs,
            args.batch_size,
            args.num_negatives,
            args.lr,
            rng,
        )
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}")

    embeddings = w_in
    for word in ["king", "queen", "man", "woman"]:
        if word in word_to_id:
            print(
                word,
                nearest_neighbors(
                    word,
                    word_to_id,
                    id_to_word,
                    embeddings,
                    top_k=5,
                ),
            )

    config = {
        "data_path": args.data_path,
        "min_count": args.min_count,
        "window_size": args.window_size,
        "embed_dim": args.embed_dim,
        "batch_size": args.batch_size,
        "num_negatives": args.num_negatives,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "tokenizer": {
            "lower": True,
            "method": "whitespace_split",
        },
    }
    save_checkpoint(args.save_dir, w_in, w_out, word_to_id, id_to_word, config)


if __name__ == "__main__":
    main()
