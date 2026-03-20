import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data import (
    build_negative_sampling_cdf,
    build_vocab,
    count_training_pairs,
    encode_tokens,
    read_text,
    stream_training_pair_chunks,
    subsample_token_ids,
    tokenize,
)
from utils import nearest_neighbors, set_seed
from word2vec import init_parameters, train_batch

WORD2VEC_MIN_LR_RATIO = 1e-4


@dataclass(frozen=True)
class TrainConfig:
    data_path: str = "data/text8"
    min_count: int = 5
    window_size: int = 5
    embed_dim: int = 300
    batch_size: int = 448
    num_negatives: int = 5
    negative_sampling_power: float = 0.75
    subsample_t: float = 1e-5
    lr: float = 0.01
    epochs: int = 5
    seed: int = 42
    save_dir: str = "artifacts"
    init_range_scale: float = 0.5
    cosine_eps: float = 1e-12


def _yield_batches_from_buffer(buffer, batch_size):
    for i in range(0, len(buffer), batch_size):
        batch_pairs = buffer[i : i + batch_size]
        yield batch_pairs[:, 0], batch_pairs[:, 1]


def create_batches(
    token_ids,
    window_size,
    batch_size,
    shuffle=True,
    rng=None,
    shuffle_buffer_size=None,
):
    token_ids = np.asarray(token_ids, dtype=np.int32)
    if token_ids.size == 0 or window_size < 1:
        return

    if rng is None:
        rng = np.random.default_rng()

    if shuffle_buffer_size is None:
        shuffle_buffer_size = max(batch_size * 32, batch_size)
    shuffle_buffer_size = max(int(shuffle_buffer_size), batch_size)

    if not shuffle:
        pending_chunks = []
        pending_pairs = 0
        for chunk in stream_training_pair_chunks(
            token_ids, window_size, chunk_size=batch_size
        ):
            pending_chunks.append(chunk)
            pending_pairs += len(chunk)

            while pending_pairs >= batch_size:
                buffer = np.concatenate(pending_chunks, axis=0)
                limit = (len(buffer) // batch_size) * batch_size
                for batch in _yield_batches_from_buffer(buffer[:limit], batch_size):
                    yield batch

                remainder = buffer[limit:]
                pending_chunks = [remainder.copy()] if len(remainder) else []
                pending_pairs = len(remainder)

        if pending_pairs > 0:
            buffer = np.concatenate(pending_chunks, axis=0)
            for batch in _yield_batches_from_buffer(buffer, batch_size):
                yield batch
        return

    pending_chunks = []
    pending_pairs = 0
    chunk_size = max(shuffle_buffer_size, batch_size)

    for chunk in stream_training_pair_chunks(
        token_ids, window_size, chunk_size=chunk_size
    ):
        pending_chunks.append(chunk)
        pending_pairs += len(chunk)

        if pending_pairs < shuffle_buffer_size:
            continue

        buffer = np.concatenate(pending_chunks, axis=0)
        rng.shuffle(buffer)

        limit = (len(buffer) // batch_size) * batch_size
        for batch in _yield_batches_from_buffer(buffer[:limit], batch_size):
            yield batch

        remainder = buffer[limit:]
        pending_chunks = [remainder.copy()] if len(remainder) else []
        pending_pairs = len(remainder)

    if pending_pairs > 0:
        buffer = np.concatenate(pending_chunks, axis=0)
        rng.shuffle(buffer)
        for batch in _yield_batches_from_buffer(buffer, batch_size):
            yield batch


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


def compute_decayed_lr(
    starting_lr,
    processed_pairs,
    total_training_pairs,
    min_lr_ratio=WORD2VEC_MIN_LR_RATIO,
):
    if starting_lr <= 0.0:
        return 0.0

    min_lr = starting_lr * min_lr_ratio
    if total_training_pairs <= 0:
        return min_lr

    progress = max(processed_pairs, 0)
    lr = starting_lr * (1.0 - (progress / (total_training_pairs + 1)))
    return max(lr, min_lr)


def train_epoch(
    token_ids,
    window_size,
    w_in,
    w_out,
    neg_probs,
    batch_size,
    num_negatives,
    lr,
    rng,
    shuffle_buffer_size=None,
    total_training_pairs=None,
    pairs_processed_before_epoch=0,
):
    total_loss = 0.0
    num_batches = 0
    total_pairs = count_training_pairs(len(token_ids), window_size)
    if total_pairs == 0:
        return 0.0
    if total_training_pairs is None:
        total_training_pairs = total_pairs

    total_steps = (total_pairs + batch_size - 1) // batch_size
    pairs_processed = 0

    progress_bar = tqdm(
        create_batches(
            token_ids,
            window_size,
            batch_size,
            shuffle=True,
            rng=rng,
            shuffle_buffer_size=shuffle_buffer_size,
        ),
        total=total_steps,
        desc="Training",
        unit="batch",
        leave=False,
    )

    for center_ids, context_ids in progress_bar:
        current_lr = compute_decayed_lr(
            lr,
            pairs_processed_before_epoch + pairs_processed,
            total_training_pairs,
        )
        negative_ids = sample_negative_matrix(
            rng, neg_probs, context_ids, num_negatives
        )

        loss = train_batch(
            center_ids, context_ids, negative_ids, w_in, w_out, current_lr
        )
        total_loss += loss
        num_batches += 1
        pairs_processed += center_ids.shape[0]

        progress_bar.set_postfix(
            lr=f"{current_lr:.6f}",
            loss=f"{loss:.4f}",
            avg_loss=f"{(total_loss / num_batches):.4f}",
        )

    return total_loss / max(num_batches, 1)


def parse_args(defaults: TrainConfig | None = None):
    defaults = defaults or TrainConfig()
    parser = argparse.ArgumentParser(
        description="Train skip-gram word2vec with negative sampling in NumPy"
    )
    parser.add_argument("--data_path", type=str, default=defaults.data_path)
    parser.add_argument("--min_count", type=int, default=defaults.min_count)
    parser.add_argument("--window_size", type=int, default=defaults.window_size)
    parser.add_argument("--embed_dim", type=int, default=defaults.embed_dim)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--num_negatives", type=int, default=defaults.num_negatives)
    parser.add_argument(
        "--negative_sampling_power",
        type=float,
        default=defaults.negative_sampling_power,
    )
    parser.add_argument("--subsample_t", type=float, default=defaults.subsample_t)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--save_dir", type=str, default=defaults.save_dir)
    parser.add_argument(
        "--init_range_scale", type=float, default=defaults.init_range_scale
    )
    parser.add_argument("--cosine_eps", type=float, default=defaults.cosine_eps)
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
    config = TrainConfig(
        data_path=args.data_path,
        min_count=args.min_count,
        window_size=args.window_size,
        embed_dim=args.embed_dim,
        batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        negative_sampling_power=args.negative_sampling_power,
        subsample_t=args.subsample_t,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        save_dir=args.save_dir,
        init_range_scale=args.init_range_scale,
        cosine_eps=args.cosine_eps,
    )
    
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    text = read_text(config.data_path)
    tokens = tokenize(text)
    word_to_id, id_to_word, counts = build_vocab(tokens, min_count=config.min_count)
    token_ids = np.asarray(encode_tokens(tokens, word_to_id), dtype=np.int32)
    token_ids = subsample_token_ids(token_ids, counts, config.subsample_t, rng)
    neg_cdf = build_negative_sampling_cdf(
        counts, power=config.negative_sampling_power
    )

    vocab_size = len(word_to_id)
    w_in, w_out = init_parameters(
        vocab_size,
        config.embed_dim,
        rng,
        init_range_scale=config.init_range_scale,
    )
    total_pairs_per_epoch = count_training_pairs(len(token_ids), config.window_size)
    total_training_pairs = total_pairs_per_epoch * config.epochs

    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")
        avg_loss = train_epoch(
            token_ids,
            config.window_size,
            w_in,
            w_out,
            neg_cdf,
            config.batch_size,
            config.num_negatives,
            config.lr,
            rng,
            total_training_pairs=total_training_pairs,
            pairs_processed_before_epoch=epoch * total_pairs_per_epoch,
        )
        print(f"Epoch {epoch + 1}/{config.epochs}: avg_loss={avg_loss:.4f}")

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
                    eps=config.cosine_eps,
                ),
            )

    payload_config = asdict(config)
    payload_config["tokenizer"] = {
        "lower": True,
        "method": "whitespace_split",
    }
    save_checkpoint(
        config.save_dir,
        w_in,
        w_out,
        word_to_id,
        id_to_word,
        payload_config,
    )


if __name__ == "__main__":
    main()
