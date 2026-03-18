import unittest

import numpy as np

from word2vec.architectures import CBOWArchitecture, SkipGramArchitecture
from word2vec.dataset import generate_cbow_examples, generate_skipgram_pairs
from word2vec.model import Word2Vec
from word2vec.objectives import (
    HierarchicalSoftmaxObjective,
    NCEObjective,
    NegativeSamplingObjective,
)
from word2vec.preprocessing import build_vocab, encode_tokens, split_tokens
from word2vec.trainer import Word2VecTrainer, Word2VecTrainingConfig


def build_toy_corpus():
    text = "the quick brown fox jumps over the lazy dog the quick fox"
    tokens = split_tokens(text)
    word_to_index, _, _, counts = build_vocab(tokens, min_count=1)
    token_ids = encode_tokens(tokens, word_to_index)
    return token_ids, counts, len(word_to_index)


class TrainingTests(unittest.TestCase):
    def test_negative_sampling_trains_skipgram(self) -> None:
        token_ids, counts, vocab_size = build_toy_corpus()
        examples = generate_skipgram_pairs(token_ids, window_size=2)
        model = Word2Vec(vocab_size=vocab_size, embedding_dim=8, seed=0)
        objective = NegativeSamplingObjective(counts=counts, num_negative=3, seed=1)
        trainer = Word2VecTrainer(model, SkipGramArchitecture(), objective)

        losses = trainer.fit(examples, Word2VecTrainingConfig(learning_rate=0.05, epochs=2))

        self.assertEqual(len(losses), 2)
        self.assertTrue(all(np.isfinite(loss) for loss in losses))

    def test_negative_sampling_trains_cbow(self) -> None:
        token_ids, counts, vocab_size = build_toy_corpus()
        examples = generate_cbow_examples(token_ids, window_size=2)
        model = Word2Vec(vocab_size=vocab_size, embedding_dim=8, seed=0)
        objective = NegativeSamplingObjective(counts=counts, num_negative=3, seed=2)
        trainer = Word2VecTrainer(model, CBOWArchitecture(), objective)

        losses = trainer.fit(examples, Word2VecTrainingConfig(learning_rate=0.05, epochs=2))

        self.assertEqual(len(losses), 2)
        self.assertTrue(all(np.isfinite(loss) for loss in losses))

    def test_hierarchical_softmax_trains_both_architectures(self) -> None:
        token_ids, counts, vocab_size = build_toy_corpus()
        config = Word2VecTrainingConfig(learning_rate=0.05, epochs=2)

        for architecture, examples in (
            (SkipGramArchitecture(), generate_skipgram_pairs(token_ids, window_size=2)),
            (CBOWArchitecture(), generate_cbow_examples(token_ids, window_size=2)),
        ):
            model = Word2Vec(vocab_size=vocab_size, embedding_dim=8, seed=0)
            objective = HierarchicalSoftmaxObjective(counts=counts)
            trainer = Word2VecTrainer(model, architecture, objective)

            losses = trainer.fit(examples, config)

            self.assertEqual(len(losses), 2)
            self.assertTrue(all(np.isfinite(loss) for loss in losses))

    def test_nce_trains_both_architectures(self) -> None:
        token_ids, counts, vocab_size = build_toy_corpus()
        config = Word2VecTrainingConfig(learning_rate=0.05, epochs=2)

        for architecture, examples, seed in (
            (SkipGramArchitecture(), generate_skipgram_pairs(token_ids, window_size=2), 3),
            (CBOWArchitecture(), generate_cbow_examples(token_ids, window_size=2), 4),
        ):
            model = Word2Vec(vocab_size=vocab_size, embedding_dim=8, seed=0)
            objective = NCEObjective(counts=counts, num_noise=3, seed=seed)
            trainer = Word2VecTrainer(model, architecture, objective)

            losses = trainer.fit(examples, config)

            self.assertEqual(len(losses), 2)
            self.assertTrue(all(np.isfinite(loss) for loss in losses))


if __name__ == "__main__":
    unittest.main()
