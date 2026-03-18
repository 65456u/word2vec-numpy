from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np


@dataclass(slots=True)
class HuffmanNode:
    index: int | None
    frequency: int
    left: "HuffmanNode | None" = None
    right: "HuffmanNode | None" = None
    internal_index: int | None = None


@dataclass(slots=True)
class HuffmanTree:
    root: HuffmanNode
    paths: list[np.ndarray]
    codes: list[np.ndarray]
    num_internal_nodes: int


def _build_paths(
    node: HuffmanNode,
    path: list[int],
    code: list[int],
    paths: list[np.ndarray],
    codes: list[np.ndarray],
) -> None:
    if node.index is not None:
        paths[node.index] = np.array(path, dtype=np.int64)
        codes[node.index] = np.array(code, dtype=np.int64)
        return

    if node.internal_index is None:
        raise ValueError("Internal Huffman nodes must have an internal_index.")

    if node.left is not None:
        _build_paths(
            node.left,
            [*path, node.internal_index],
            [*code, 0],
            paths,
            codes,
        )

    if node.right is not None:
        _build_paths(
            node.right,
            [*path, node.internal_index],
            [*code, 1],
            paths,
            codes,
        )


def build_huffman_tree(counts: np.ndarray) -> HuffmanTree:
    """
    Builds the Huffman tree used by hierarchical softmax.
    """
    if counts.ndim != 1:
        raise ValueError("counts must be a 1D array")
    if counts.size == 0:
        raise ValueError("counts must not be empty")

    if counts.size == 1:
        root = HuffmanNode(index=0, frequency=int(counts[0]))
        return HuffmanTree(
            root=root,
            paths=[np.array([], dtype=np.int64)],
            codes=[np.array([], dtype=np.int64)],
            num_internal_nodes=0,
        )

    heap: list[tuple[int, int, HuffmanNode]] = []
    order = 0
    for index, frequency in enumerate(counts):
        heapq.heappush(
            heap,
            (int(frequency), order, HuffmanNode(index=index, frequency=int(frequency))),
        )
        order += 1

    internal_index = 0
    while len(heap) > 1:
        left_frequency, _, left = heapq.heappop(heap)
        right_frequency, _, right = heapq.heappop(heap)

        parent = HuffmanNode(
            index=None,
            frequency=left_frequency + right_frequency,
            left=left,
            right=right,
            internal_index=internal_index,
        )
        heapq.heappush(heap, (parent.frequency, order, parent))
        order += 1
        internal_index += 1

    root = heap[0][2]
    paths: list[np.ndarray] = [np.array([], dtype=np.int64) for _ in range(counts.size)]
    codes: list[np.ndarray] = [np.array([], dtype=np.int64) for _ in range(counts.size)]
    _build_paths(root, [], [], paths, codes)

    return HuffmanTree(
        root=root,
        paths=paths,
        codes=codes,
        num_internal_nodes=internal_index,
    )
