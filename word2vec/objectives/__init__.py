from word2vec.objectives.base import Word2VecObjective
from word2vec.objectives.hierarchical_softmax import HierarchicalSoftmaxObjective
from word2vec.objectives.negative_sampling import NegativeSamplingObjective
from word2vec.objectives.noise_contrastive_estimation import NCEObjective

__all__ = [
    "Word2VecObjective",
    "NegativeSamplingObjective",
    "HierarchicalSoftmaxObjective",
    "NCEObjective",
]
