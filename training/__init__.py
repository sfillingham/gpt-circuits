"""
Training helpers
"""

import math

from config import TrainingConfigBase


def get_lr(step, config: TrainingConfigBase):
    """
    Get the learning rate for a given step. Assumes that step starts at 0 and ends at max_steps - 1.
    """
    # 1) linear warmup for warmup_iters steps
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    # 2) if not decaying, return the learning rate
    if not config.decay_lr:
        return config.learning_rate
    # 3) if it > max_steps, return min learning rate
    if step > config.max_steps:
        return config.min_lr
    # 4) in between, use cosine decay down to min learning rate
    decay_ratio = ((step + 1) - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)
