"""
Heart Sound Classification with Dual-Path Self-Supervised Learning

A comprehensive framework for heart sound classification using dual-path contrastive learning
with Temporal Convolutional Networks (TCN) and ResNet-based spectrogram analysis.

Authors: Research Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from .models import *
from .preprocessing import *
from .training import *
from .evaluation import *
from .utils import *
__all__ = [ 
    'DualPathContrastiveLearning',
    'DilatedTemporalConvEncoder',
    'ResNet50SpectrogramEncoder',
    'MurmurCenteredPreprocessor',
    'CalibrationMetrics',
    'StatisticalTests',
    'VisualizationUtils'
]
