from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence

import numpy as np

from word2vec.architectures.base import Word2VecArchitecture
from word2vec.model import Word2Vec


TrainingExample = Any


@dataclass(slots=True)
class Word2VecTrainingConfig:
    learning_rate: float = 0.025
    epochs: int = 1


class Objective(Protocol):
    def initialize(self, model: Word2Vec) -> None:
        """
        Prepare any objective-specific state before training starts.
        """
        raise NotImplementedError("Objective.initialize must be implemented by subclasses.")

    def train_step(
        self,
        model: Word2Vec,
        hidden: np.ndarray,
        target_id: int,
        learning_rate: float,
    ) -> tuple[float, np.ndarray]:
        """
        Execute a single SGD update and return loss plus hidden-state gradient.
        """
        raise NotImplementedError("Objective.train_step must be implemented by subclasses.")


class Word2VecTrainer:
    def __init__(
        self,
        model: Word2Vec,
        architecture: Word2VecArchitecture[Any, Any],
        objective: Objective,
    ):
        self.model = model
        self.architecture = architecture
        self.objective = objective

    def fit(
        self,
        examples: Sequence[TrainingExample] | Iterable[TrainingExample],
        config: Word2VecTrainingConfig,
    ) -> list[float]:
        """
        Runs the objective-specific SGD loop and returns average loss per epoch.
        """
        self.objective.initialize(self.model)
        materialized_examples = examples if isinstance(examples, Sequence) else list(examples)
        epoch_losses: list[float] = []

        for _ in range(config.epochs):
            total_loss = 0.0
            num_examples = 0

            for example in materialized_examples:
                hidden, target_id, cache = self.architecture.forward(self.model, example)
                loss, hidden_gradient = self.objective.train_step(
                    self.model,
                    hidden,
                    target_id,
                    config.learning_rate,
                )
                self.architecture.backward(
                    self.model,
                    cache,
                    hidden_gradient,
                    config.learning_rate,
                )
                total_loss += loss
                num_examples += 1

            average_loss = total_loss / max(1, num_examples)
            epoch_losses.append(average_loss)

        return epoch_losses
