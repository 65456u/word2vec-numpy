from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from word2vec.model import Word2Vec
from word2vec.objectives.base import Word2VecObjective
from word2vec.trees import HuffmanTree, build_huffman_tree
from word2vec.utils import sigmoid


@dataclass(slots=True)
class HierarchicalSoftmaxObjective(Word2VecObjective):
    counts: np.ndarray
    tree: HuffmanTree | None = None

    def initialize(self, model: Word2Vec) -> None:
        if self.tree is None:
            self.tree = build_huffman_tree(self.counts)
        model.ensure_output_capacity(self.tree.num_internal_nodes)

    def train_step(
        self,
        model: Word2Vec,
        hidden: np.ndarray,
        target_id: int,
        learning_rate: float,
    ) -> tuple[float, np.ndarray]:
        if self.tree is None:
            raise RuntimeError("HierarchicalSoftmaxObjective must be initialized before training.")

        path = self.tree.paths[target_id]
        codes = self.tree.codes[target_id]

        if path.size == 0:
            return 0.0, np.zeros_like(hidden)

        node_vectors = model.output_embeddings[path].copy()
        raw_scores = node_vectors @ hidden
        probabilities = np.asarray(sigmoid(raw_scores))
        labels = codes.astype(np.float64)

        output_gradients = (probabilities - labels)[:, None] * hidden[None, :]
        hidden_gradient = np.sum(
            (probabilities - labels)[:, None] * node_vectors,
            axis=0,
        )

        model.output_embeddings[path] -= learning_rate * output_gradients

        epsilon = 1e-12
        loss = -np.sum(
            labels * np.log(probabilities + epsilon)
            + (1.0 - labels) * np.log(1.0 - probabilities + epsilon)
        )
        return float(loss), hidden_gradient
