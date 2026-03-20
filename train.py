import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data import (
    build_negative_sampling_cdf,
    build_vocab,
    encode_tokens,
    generate_training_pairs_array,
    read_text,
    subsample_token_ids,
    tokenize,
)
from utils import nearest_neighbors
from word2vec import init_parameters, train_batch


def create_batches(pairs, batch_size, shuffle=True, rng=None):
    pairs = np.asarray(pairs)
    num_pairs = len(pairs)
    if not shuffle:
        for i in range(0, num_pairs, batch_size):
            batch_pairs = pairs[i : i + batch_size]
            yield batch_pairs[:, 0], batch_pairs[:, 1]
        return

    indices = np.arange(num_pairs, dtype=np.int32)
    if rng is None:
        rng = np.random.default_rng()
    rng.shuffle(indices)

    for i in range(0, num_pairs, batch_size):
        batch_indices = indices[i : i + batch_size]
        batch_pairs = pairs[batch_indices]
        yield batch_pairs[:, 0], batch_pairs[:, 1]


def sample_negative_matrix(rng, neg_probs, batch_context_ids, num_negatives):
    neg_probs = np.asarray(neg_probs, dtype=np.float64)
    if neg_probs.size == 0:
        return np.empty((batch_context_ids.shape[0], num_negatives), dtype=np.int64)

    is_cdf = np.isclose(neg_probs[-1], 1.0) and np.all(np.diff(neg_probs) >= 0.0)
    neg_cdf = neg_probs if is_cdf else np.cumsum(neg_probs)
    neg_cdf[-1] = 1.0

    negative_ids = np.searchsorted(
        neg_cdf,
        rng.random((batch_context_ids.shape[0], num_negatives)),
        side="right",
    ).astype(np.int64, copy=False)

    invalid_mask = negative_ids == batch_context_ids[:, None]
    while invalid_mask.any():
        negative_ids[invalid_mask] = np.searchsorted(
            neg_cdf,
            rng.random(invalid_mask.sum()),
            side="right",
        ).astype(np.int64, copy=False)
        invalid_mask = negative_ids == batch_context_ids[:, None]

    return negative_ids


def train_epoch(pairs, w_in, w_out, neg_probs, batch_size, num_negatives, lr, rng):
    total_loss = 0.0
    num_batches = 0
    total_steps = (len(pairs) + batch_size - 1) // batch_size

    progress_bar = tqdm(
        create_batches(pairs, batch_size, shuffle=True, rng=rng),
        total=total_steps,
        desc="Training",
        unit="batch",
        leave=False,
    )

    for center_ids, context_ids in progress_bar:
        negative_ids = sample_negative_matrix(
            rng, neg_probs, context_ids, num_negatives
        )

        loss = train_batch(center_ids, context_ids, negative_ids, w_in, w_out, lr)
        total_loss += loss
        num_batches += 1

        progress_bar.set_postfix(
            loss=f"{loss:.4f}",
            avg_loss=f"{(total_loss / num_batches):.4f}",
        )

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
    parser.add_argument("--subsample_t", type=float, default=1e-5)
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
    token_ids = np.asarray(encode_tokens(tokens, word_to_id), dtype=np.int32)
    token_ids = subsample_token_ids(token_ids, counts, args.subsample_t, rng)
    pairs = generate_training_pairs_array(token_ids, window_size=args.window_size)
    neg_cdf = build_negative_sampling_cdf(counts)

    vocab_size = len(word_to_id)
    w_in, w_out = init_parameters(vocab_size, args.embed_dim, rng)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        avg_loss = train_epoch(
            pairs,
            w_in,
            w_out,
            neg_cdf,
            args.batch_size,
            args.num_negatives,
            args.lr,
            rng,
        )
        print(f"Epoch {epoch + 1}/{args.epochs}: avg_loss={avg_loss:.4f}")

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
        "subsample_t": args.subsample_t,
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
