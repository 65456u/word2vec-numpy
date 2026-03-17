from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class HuffmanNode:
    index: int
    frequency: int
    left: "HuffmanNode | None" = None
    right: "HuffmanNode | None" = None


@dataclass(slots=True)
class HuffmanTree:
    root: HuffmanNode
    paths: list[np.ndarray]
    codes: list[np.ndarray]


def build_huffman_tree(counts: np.ndarray) -> HuffmanTree:
    """
    Placeholder for hierarchical softmax tree construction.
    """
    raise NotImplementedError("Hierarchical softmax tree construction is not implemented yet.")
