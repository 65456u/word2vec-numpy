import numpy as np
import random

def sigmoid(x):
    x = np.asarray(x)
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )

def log_sigmoid(x):
    x = np.asarray(x)
    return -np.logaddexp(0.0, -x)

def cosine_similarity(a, b, eps=1e-12):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / max(norm_a * norm_b, eps)

def nearest_neighbors(query_word, word_to_id, id_to_word, embeddings, top_k=5):
    if query_word not in word_to_id:
        return []

    query_id = word_to_id[query_word]
    query_embedding = embeddings[query_id]
    similarities = []

    for word, idx in word_to_id.items():
        if word == query_word:
            continue
        embedding = embeddings[idx]
        sim = cosine_similarity(query_embedding, embedding)
        if np.isnan(sim):
            continue
        similarities.append((word, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)