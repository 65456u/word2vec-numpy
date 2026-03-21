import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from embed import (
    export_embeddings,
    load_checkpoint,
    nearest_neighbors_for_vector,
    select_embeddings,
)


class EmbedModuleTests(unittest.TestCase):
    def test_load_checkpoint_reads_vocab_and_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            np.savez(
                checkpoint_dir / "model.npz",
                w_in=np.array([[1.0, 2.0]], dtype=np.float32),
                w_out=np.array([[3.0, 4.0]], dtype=np.float32),
            )
            with open(checkpoint_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "vocab": ["king"],
                        "word_to_id": {"king": 0},
                        "config": {"cosine_eps": 1e-9},
                    },
                    f,
                )

            vocab, word_to_id, w_in, w_out, config = load_checkpoint(checkpoint_dir)

            self.assertEqual(vocab, ["king"])
            self.assertEqual(word_to_id, {"king": 0})
            assert_allclose(w_in, np.array([[1.0, 2.0]], dtype=np.float32))
            assert_allclose(w_out, np.array([[3.0, 4.0]], dtype=np.float32))
            self.assertEqual(config["cosine_eps"], 1e-9)

    def test_select_embeddings_supports_average(self):
        w_in = np.array([[1.0, 3.0]], dtype=np.float32)
        w_out = np.array([[5.0, 7.0]], dtype=np.float32)

        embeddings = select_embeddings(w_in, w_out, "average")

        assert_allclose(embeddings, np.array([[3.0, 5.0]], dtype=np.float32))

    def test_export_embeddings_writes_tsv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "embeddings.tsv"

            export_embeddings(
                export_path,
                ["king", "queen"],
                np.array([[1.0, 2.0], [3.5, 4.5]], dtype=np.float32),
            )

            content = export_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(content[0], "king\t1.000000 2.000000")
            self.assertEqual(content[1], "queen\t3.500000 4.500000")

    def test_nearest_neighbors_for_vector_excludes_prompt_words(self):
        vocab = ["man", "king", "woman", "queen", "apple"]
        word_to_id = {word: idx for idx, word in enumerate(vocab)}
        embeddings = np.array(
            [
                [1.0, 0.0],
                [2.0, 1.0],
                [0.0, 1.0],
                [1.0, 2.0],
                [-1.0, -1.0],
            ],
            dtype=np.float32,
        )
        query_vector = embeddings[word_to_id["king"]] - embeddings[word_to_id["man"]] + embeddings[word_to_id["woman"]]

        neighbors = nearest_neighbors_for_vector(
            query_vector,
            vocab,
            word_to_id,
            embeddings,
            top_k=2,
            exclude_words={"man", "king", "woman"},
        )

        self.assertEqual(neighbors[0][0], "queen")


if __name__ == "__main__":
    unittest.main()
