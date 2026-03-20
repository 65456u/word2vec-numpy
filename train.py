import numpy as np
from data import build_negative_sampling_distribution, build_vocab, encode_tokens, generate_training_pairs, read_text, sample_negative_ids, tokenize
from utils import nearest_neighbors
from word2vec import init_parameters, train_batch

def create_batches(pairs, batch_size, shuffle = True, rng = None):
    indices = np.arange(len(pairs))
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        barch_pairs = [pairs[idx] for idx in batch_indices]
        center_ids = np.array([pair[0] for pair in barch_pairs], dtype=np.int64)
        context_ids = np.array([pair[1] for pair in barch_pairs], dtype=np.int64)
        yield center_ids, context_ids

def sample_negative_matrix(rng, neg_probs, batch_context_ids, num_negatives):
    batch_size = len(batch_context_ids)
    negative_ids = []

    for context_id in batch_context_ids:
        neg_ids = sample_negative_ids(
            rng,
            neg_probs,
            k=num_negatives,
            forbidden_indices={int(context_id)}
        )
        negative_ids.append(neg_ids)

    return np.array(negative_ids, dtype=np.int64)

def train_epoch(pairs, w_in, w_out, neg_probs, batch_size, num_negatives, lr, rng):
    total_loss = 0.0
    num_batches = 0

    for center_ids, context_ids in create_batches(pairs, batch_size, shuffle=True, rng=rng):
        negative_ids = sample_negative_matrix(
            rng, neg_probs, context_ids, num_negatives
        )

        loss = train_batch(center_ids, context_ids, negative_ids, w_in, w_out, lr)
        total_loss += loss
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    rng = np.random.default_rng(42)

    text = read_text("data/text8")
    tokens = tokenize(text)
    word_to_id, id_to_word, counts = build_vocab(tokens, min_count=5)
    token_ids = encode_tokens(tokens, word_to_id)
    pairs = generate_training_pairs(token_ids, window_size=2)
    neg_probs = build_negative_sampling_distribution(counts)

    vocab_size = len(word_to_id)
    embed_dim = 100
    batch_size = 256
    num_negatives = 5
    lr = 0.025
    epochs = 3

    w_in, w_out = init_parameters(vocab_size, embed_dim, rng)

    for epoch in range(epochs):
        avg_loss = train_epoch(
            pairs, w_in, w_out, neg_probs, batch_size, num_negatives, lr, rng
        )
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}")

    embeddings = w_in
    for word in ["king", "queen", "man", "woman"]:
        if word in word_to_id:
            print(word, nearest_neighbors(word, word_to_id, id_to_word, embeddings, top_k=5))