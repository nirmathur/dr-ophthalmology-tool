import numpy as np
import tensorflow as tf
import random


def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
