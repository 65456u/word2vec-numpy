import argparse
import json
from pathlib import Path

import numpy as np

from utils import cosine_similarity, nearest_neighbors


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load trained Word2Vec checkpoints and inspect or export embeddings."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="artifacts",
        help="Directory containing model.npz and config.json.",
    )
    parser.add_argument(
        "--embedding-source",
        choices=("input", "output", "average"),
        default="input",
        help="Which embedding matrix to use for inference/export.",
    )
    parser.add_argument(
        "--word",
        type=str,
        help="Word to inspect.",
    )
    parser.add_argument(
        "--words",
        nargs="+",
        help="Multiple words to inspect in sequence.",
    )
    parser.add_argument(
        "--analogy",
        nargs=3,
        metavar=("A", "B", "C"),
        help="Solve analogies of the form A:B :: C:?.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest neighbors to show for --word.",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Write all embeddings to a TSV file.",
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_dir):
    checkpoint_path = Path(checkpoint_dir)
    model_path = checkpoint_path / "model.npz"
    config_path = checkpoint_path / "config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with np.load(model_path) as model:
        w_in = model["w_in"]
        w_out = model["w_out"]

    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    vocab = payload["vocab"]
    word_to_id = payload.get("word_to_id", {word: idx for idx, word in enumerate(vocab)})
    config = payload.get("config", {})
    return vocab, word_to_id, w_in, w_out, config


def select_embeddings(w_in, w_out, source):
    if source == "input":
        return w_in
    if source == "output":
        return w_out
    if source == "average":
        return (w_in + w_out) / 2.0
    raise ValueError(f"Unsupported embedding source: {source}")


def format_vector(vector):
    return " ".join(f"{value:.6f}" for value in vector)


def export_embeddings(path, vocab, embeddings):
    export_path = Path(path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    with open(export_path, "w", encoding="utf-8") as f:
        for word, vector in zip(vocab, embeddings):
            f.write(f"{word}\t{format_vector(vector)}\n")


def build_id_to_word(vocab):
    return {idx: token for idx, token in enumerate(vocab)}


def inspect_word(word, word_to_id, vocab, embeddings, top_k, eps):
    if word not in word_to_id:
        raise KeyError(f"Word '{word}' is not in the vocabulary.")

    word_id = word_to_id[word]
    vector = embeddings[word_id]
    print(f"word: {word}")
    print(f"index: {word_id}")
    print(f"vector: {format_vector(vector)}")

    neighbors = nearest_neighbors(
        word,
        word_to_id,
        build_id_to_word(vocab),
        embeddings,
        top_k=top_k,
        eps=eps,
    )
    if neighbors:
        print("neighbors:")
        for neighbor, score in neighbors:
            print(f"  {neighbor}\t{score:.6f}")


def inspect_words(words, word_to_id, vocab, embeddings, top_k, eps):
    for index, word in enumerate(words):
        if index > 0:
            print()
        inspect_word(word, word_to_id, vocab, embeddings, top_k, eps)


def nearest_neighbors_for_vector(
    query_vector, vocab, word_to_id, embeddings, top_k=5, eps=1e-12, exclude_words=None
):
    if exclude_words is None:
        exclude_words = set()

    similarities = []
    for word, idx in word_to_id.items():
        if word in exclude_words:
            continue
        sim = cosine_similarity(query_vector, embeddings[idx], eps=eps)
        if np.isnan(sim):
            continue
        similarities.append((word, sim))

    similarities.sort(key=lambda item: item[1], reverse=True)
    return similarities[:top_k]


def solve_analogy(words, word_to_id, vocab, embeddings, top_k, eps):
    a, b, c = words
    missing_words = [word for word in (a, b, c) if word not in word_to_id]
    if missing_words:
        missing = ", ".join(missing_words)
        raise KeyError(f"Word(s) not in vocabulary: {missing}")

    query_vector = (
        embeddings[word_to_id[b]]
        - embeddings[word_to_id[a]]
        + embeddings[word_to_id[c]]
    )
    neighbors = nearest_neighbors_for_vector(
        query_vector,
        vocab,
        word_to_id,
        embeddings,
        top_k=top_k,
        eps=eps,
        exclude_words={a, b, c},
    )
    print(f"analogy: {a}:{b} :: {c}:?")
    for neighbor, score in neighbors:
        print(f"  {neighbor}\t{score:.6f}")


def main():
    args = parse_args()
    vocab, word_to_id, w_in, w_out, config = load_checkpoint(args.checkpoint_dir)
    embeddings = select_embeddings(w_in, w_out, args.embedding_source)
    eps = float(config.get("cosine_eps", 1e-12))

    if args.export:
        export_embeddings(args.export, vocab, embeddings)
        print(f"Exported {len(vocab)} embeddings to {args.export}")

    if args.word:
        inspect_word(args.word, word_to_id, vocab, embeddings, args.top_k, eps)
    if args.words:
        inspect_words(args.words, word_to_id, vocab, embeddings, args.top_k, eps)
    if args.analogy:
        solve_analogy(args.analogy, word_to_id, vocab, embeddings, args.top_k, eps)

    if not args.export and not args.word and not args.words and not args.analogy:
        print(
            f"Loaded {len(vocab)} embeddings with dimension {embeddings.shape[1]} "
            f"from {args.checkpoint_dir}"
        )


if __name__ == "__main__":
    main()
