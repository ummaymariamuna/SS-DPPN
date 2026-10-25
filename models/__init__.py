"""
Neural Network Models for Heart Sound Classification

This package contains all neural network models and components used in the
dual-path contrastive learning framework.
"""

from .tcn_encoder import (
    DilatedTemporalConvEncoder,
    ResidualDilatedBlock,
    DilatedConvEncoder,
    DilatedBlock,  # Legacy
    Encoder
)

from .resnet_encoder import (
    ResNet50SpectrogramEncoder,
    SpectrogramEncoder,
    LightweightSpectrogramEncoder
)

from .heads import (
    ProjectionHead,
    ClassifierHead,
    PrototypicalHead,
    FusionHead
)

from .dual_path_model import (
    DualPathContrastiveLearning,
    DualPathHeartSoundModel
)

__all__ = [ 
    # TCN Encoder
    'DilatedTemporalConvEncoder',
    'ResidualDilatedBlock',
    'DilatedConvEncoder',
    'DilatedBlock',
    'Encoder',
    #ResNet Encoder
    'ResNet50SpectrogramEncoder',
    'SpectrogramEncoder',
    'LightweightSpectrogramEncoder',
    #Heads
    'ProjectionHead',
    'ClassifierHead',
    'PrototypicalHead',
    'FusionHead',

    'DualPathContrastiveLearning',
    'DualPathHeartSoundModel'
]
