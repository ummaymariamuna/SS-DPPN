"""
SS-DPPN Careful Rollover Preprocessing Package

This package provides comprehensive tools for careful rollover preprocessing
of heart sound data for murmur detection. The preprocessing strategy focuses
on maintaining clinical context while ensuring robust data quality.

Key Features:
- Careful rollover windowing with artifact detection and prevention
- Patient-aware data splitting with cross-validation support
- Cardiac-specific data augmentation techniques
- Quality-based filtering with comprehensive metadata tracking
- Enhanced data loading with flexible transformations
- Support for both balanced training and imbalanced evaluation

Modules:
- rollover_preprocessor: Careful rollover preprocessing (2000-sample windows)
- murmur_preprocessor: Murmur-centered preprocessing (4000-sample windows)
- data_loader: Data loading and PyTorch dataset utilities

Usage Example:
--------------
```python
from preprocessing import CarefulRolloverPreprocessor, EnhancedHeartSoundDataLoader

# Initialize preprocessor (PhysioNet 2022 training_data folder + csv)
preprocessor = CarefulRolloverPreprocessor(
    data_path='physionet2022/training_data',
    csv_path='physionet2022/training_data.csv',
    sampling_rate=4000
)

# Process data with patient-aware splitting (70/15/15)
preprocessor.process_dataset(
    output_dir='processed_data/',
    balance_train=True
)

# Load processed data with quality filtering
loader = EnhancedHeartSoundDataLoader(
    data_path='processed_data/train_careful_rollover.h5',
    batch_size=32,
    min_quality_threshold=0.7  # drops the padded (0.6) windows
)

train_loader, val_loader, class_info = loader.create_data_loaders(
    val_split=0.2,
    augment_train=True,  # Training-time augmentation
    stratify=True
)
```

Quality Assurance:
-----------------
This preprocessing package implements multiple quality assurance mechanisms:

1. **Artifact Detection**: Statistical outlier detection and amplitude-based filtering
2. **Crossfade Concatenation**: Smooth transitions between adjacent windows
3. **Phase-aware Alignment**: Cardiac cycle boundary preservation
4. **Quality Scoring**: Comprehensive window quality assessment
5. **Patient Stratification**: Ensuring robust train/validation/test splits
6. **Metadata Tracking**: Complete provenance and quality information

Clinical Alignment:
------------------
The preprocessing strategy is designed to maintain clinical relevance:

- Window length optimized for cardiac cycles (0.5s = typical S1-S2 duration)
- Rollover technique preserves temporal context and murmur characteristics
- Patient-aware splitting prevents data leakage
- Augmentation techniques specific to cardiac acoustics
- Quality metrics aligned with clinical audio standards

Technical Implementation:
------------------------
- Built on numpy, scipy, and h5py for efficient array processing
- PyTorch integration for seamless deep learning workflows
- Comprehensive error handling and data validation
- Memory-efficient processing for large datasets
- Extensive logging and quality reporting

For detailed documentation, see individual module docstrings.
"""

__version__ = "1.0.0"
__author__ = "MEMS-Project Research Team"
__license__ = "MIT"

# Import core preprocessing functionality
from .rollover_preprocessor import CarefulRolloverPreprocessor
from .murmur_preprocessor import MurmurCenteredPreprocessor, preprocess_segments
from .data_loader import (
    EnhancedHeartSoundDataset, 
    EnhancedHeartSoundDataLoader,
    load_careful_rollover_data,
    load_data,
    split_data,
    load_prebalanced_data,
    create_quality_filtered_loader,
    HeartSoundDataset,
    HeartSoundDataLoader
)
# Legacy imports for backward compatibility (now handled by data_loader)
# All legacy functions are available through data_loader module

__all__ = [
    #Core preprocessor classes
    'CarefulRolloverPreprocessor',
    'MurmurCenteredPreprocessor',

    'preprocess_segments',


    
    'EnhancedHeartSoundDataset',
    'EnhancedHeartSoundDataLoader',
    'load_careful_rollover_data',
    'create_quality_filtered_loader',
    
    'load_data',
    'split_data', 
    'load_prebalanced_data',
    'HeartSoundDataset',
    'HeartSoundDataLoader'
]


# Package metadata
PACKAGE_INFO = { 
    'name': 'SS-DPPN Careful Rollover Preprocessing',
    'version': __version__,
    'description': 'Clinically-aligned heart sound preprocessing with careful rollover windowing',
    'features': [
        'Careful rollover windowing with artifact prevention',
        'Patient-aware data splitting and stratification',
        'Cardiac-specific augmentation techniques',
        'Quality-based filtering and comprehensive metadata',
        'Enhanced PyTorch dataset integration',
        'Memory-efficient processing for large datasets'
    ],
    'supported_formats': ['WAV', 'FLAC', 'MP3', 'H5', 'NPY'],
    'target_sampling_rates': [4000, 8000, 16000, 22050, 44100],
    'recommended_window_lengths': [0.5, 1.0, 2.0],  # seconds
    'clinical_applications': [
        'Heart murmur detection',
        'Cardiac pathology classification', 
        'Heart sound segmentation',
        'Valve disorder assessment'
    ]
}


def get_package_info():
    """
    Get comprehensive package information
    
    Returns:
    --------
    dict
        Package metadata and capabilities
    """
    return PACKAGE_INFO.copy()


def verify_installation():
    """
    Verify that all required dependencies are available
    
    Returns:
    --------
    bool
        True if installation is complete and functional
    """
    import sys
    #import names, not pip names (scikit-learn imports as sklearn)
    required_packages = [
        'numpy', 'scipy', 'h5py', 'sklearn',
        'torch', 'tqdm', 'librosa'
    ]
    
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError: 
            missing_packages.append(package)

    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✓ All required packages are available")
    return True


def print_welcome():
    """Print welcome message with package information"""
    print("=" * 70)
    print("SS-DPPN Careful Rollover Preprocessing Package")
    print("=" * 70)
    print(f"Version: {__version__}")
    print(f"Purpose: {PACKAGE_INFO['description']}")
    print()
    print("Key Features:")
    for feature in PACKAGE_INFO['features']:
        print(f"  • {feature}")
    print()
    print("Clinical Applications:")
    for app in PACKAGE_INFO['clinical_applications']:
        print(f"  • {app}")
    print()
    print("Quick Start:")
    print("  from preprocessing import CarefulRolloverPreprocessor")
    print("  preprocessor = CarefulRolloverPreprocessor(data_path, csv_path)")
    print("  preprocessor.process_dataset('output/')")
    print("=" * 70)

# Print welcome message when package is imported
if __name__ != "__main__":
    # Only print welcome in interactive environments
    import sys
    if hasattr(sys, 'ps1'):  # Interactive interpreter
        print_welcome()

# If run as script, demonstrate usage
if __name__ == "__main__":
    print_welcome()
    print("\nRunning installation verification...")
    if not verify_installation():
        print("Please install missing dependencies before using this package.")
