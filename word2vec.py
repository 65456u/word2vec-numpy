import numpy as np

from utils import log_sigmoid, sigmoid


def init_parameters(vocab_size, embed_dim, rng, init_range_scale=0.5):
    init_bound = init_range_scale / embed_dim
    w_in = rng.uniform(
        -init_bound, init_bound, (vocab_size, embed_dim)
    ).astype(np.float32)
    w_out = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    return w_in, w_out


def forward_skipgram_negative_sampling(
    center_ids, context_ids, negative_ids, w_in, w_out
):
    center_embeds = w_in[center_ids]
    context_embeds = w_out[context_ids]
    negative_embeds = w_out[negative_ids]

    pos_scores = np.sum(center_embeds * context_embeds, axis=1)
    neg_scores = np.sum(center_embeds[:, None, :] * negative_embeds, axis=2)

    cache = {
        "center_embeds": center_embeds,
        "context_embeds": context_embeds,
        "negative_embeds": negative_embeds,
        "pos_scores": pos_scores,
        "neg_scores": neg_scores,
    }
    return cache


def compute_sgns_loss(pos_scores, neg_scores):
    loss = -log_sigmoid(pos_scores) - np.sum(log_sigmoid(-neg_scores), axis=1)
    return np.mean(loss)


def backward_skipgram_negative_sampling(
    center_ids, context_ids, negative_ids, w_in, w_out, cache
):
    center_embeds = cache["center_embeds"]
    context_embeds = cache["context_embeds"]
    negative_embeds = cache["negative_embeds"]
    pos_scores = cache["pos_scores"]
    neg_scores = cache["neg_scores"]

    batch_size = center_embeds.shape[0]

    pos_coeff = sigmoid(pos_scores) - 1.0
    neg_coeffs = sigmoid(neg_scores)

    grad_center = (
        pos_coeff[:, None] * context_embeds
        + np.sum(neg_coeffs[:, :, None] * negative_embeds, axis=1)
    ) / batch_size
    grad_pos_out = (pos_coeff[:, None] * center_embeds) / batch_size
    grad_neg_out = (neg_coeffs[:, :, None] * center_embeds[:, None, :]) / batch_size

    grad_w_in = np.zeros_like(w_in)
    grad_w_out = np.zeros_like(w_out)

    np.add.at(grad_w_in, center_ids, grad_center)
    np.add.at(grad_w_out, context_ids, grad_pos_out)
    np.add.at(grad_w_out, negative_ids, grad_neg_out)

    return grad_w_in, grad_w_out


def sgd_update(w_in, w_out, grad_w_in, grad_w_out, lr):
    w_in -= lr * grad_w_in
    w_out -= lr * grad_w_out


def apply_sparse_updates(matrix, ids, updates):
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        return

    updates = np.asarray(updates).reshape(ids.size, matrix.shape[1])
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    sorted_updates = updates[order]

    unique_mask = np.empty(sorted_ids.shape[0], dtype=bool)
    unique_mask[0] = True
    unique_mask[1:] = sorted_ids[1:] != sorted_ids[:-1]

    unique_ids = sorted_ids[unique_mask]
    segment_starts = np.flatnonzero(unique_mask)
    aggregated_updates = np.add.reduceat(sorted_updates, segment_starts, axis=0)

    matrix[unique_ids] += aggregated_updates.astype(matrix.dtype, copy=False)


def apply_negative_sparse_updates(matrix, negative_ids, coeffs, base_vectors, scale):
    negative_ids = np.asarray(negative_ids, dtype=np.int64)
    if negative_ids.size == 0:
        return

    flat_ids = negative_ids.reshape(-1)
    flat_coeffs = np.asarray(coeffs).reshape(-1)
    order = np.argsort(flat_ids, kind="stable")
    sorted_ids = flat_ids[order]

    unique_mask = np.empty(sorted_ids.shape[0], dtype=bool)
    unique_mask[0] = True
    unique_mask[1:] = sorted_ids[1:] != sorted_ids[:-1]

    segment_starts = np.flatnonzero(unique_mask)
    unique_ids = sorted_ids[unique_mask]
    num_negatives = negative_ids.shape[1]
    row_indices = order // num_negatives

    sorted_updates = (
        (scale * flat_coeffs[order])[:, None] * base_vectors[row_indices]
    )
    aggregated_updates = np.add.reduceat(sorted_updates, segment_starts, axis=0)
    matrix[unique_ids] += aggregated_updates.astype(matrix.dtype, copy=False)


def train_batch(center_ids, context_ids, negative_ids, w_in, w_out, lr):
    cache = forward_skipgram_negative_sampling(
        center_ids, context_ids, negative_ids, w_in, w_out
    )
    loss = compute_sgns_loss(cache["pos_scores"], cache["neg_scores"])

    center_embeds = cache["center_embeds"]
    context_embeds = cache["context_embeds"]
    negative_embeds = cache["negative_embeds"]
    pos_scores = cache["pos_scores"]
    neg_scores = cache["neg_scores"]
    batch_size = center_embeds.shape[0]

    pos_coeff = sigmoid(pos_scores) - 1.0
    neg_coeffs = sigmoid(neg_scores)

    grad_center = (
        pos_coeff[:, None] * context_embeds
        + np.sum(neg_coeffs[:, :, None] * negative_embeds, axis=1)
    ) / batch_size
    grad_pos_out = (pos_coeff[:, None] * center_embeds) / batch_size

    apply_sparse_updates(w_in, center_ids, -lr * grad_center)
    apply_sparse_updates(w_out, context_ids, -lr * grad_pos_out)
    apply_negative_sparse_updates(
        w_out,
        negative_ids,
        neg_coeffs,
        center_embeds,
        scale=-lr / batch_size,
    )
    return loss
