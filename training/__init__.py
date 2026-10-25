"""
Training Module for Heart Sound Classification

This package contains training utilities and classes for the dual-path
contrastive learning framework.
"""

from .trainer import DualPathTrainer, add_training_methods_to_model, HeartSoundTrainer



add_training_methods_to_model()

__all__ = [
    'DualPathTrainer',
    'HeartSoundTrainer',
    'add_training_methods_to_model'
]
