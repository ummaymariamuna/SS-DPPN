"""
Setup script for heart_sound_classification package
"""

from setuptools import setup, find_packages 
import os

#Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]


setup(
    name="heart-sound-classification",
    version="1.0.0",
    author="Research Team",
    author_email="your.email@university.edu",
    description="Dual-Path Self-Supervised Contrastive Learning for Heart Sound Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/heart-sound-classification",

    package_dir={"heart_sound_classification": "."},
    packages=["heart_sound_classification"] +
             ["heart_sound_classification." + p for p in find_packages(exclude=["venv*", "build*", "dist*"])],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "full": [
            "geomloss>=0.2.4",
            "umap-learn>=0.5.3",
            "statsmodels>=0.13.0",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.900",
        ],
    }, 
    # no console_scripts for now, the examples/ scripts they pointed to don't exist yet
    project_urls={
        "Bug Reports": "https://github.com/your-username/heart-sound-classification/issues",
        "Source": "https://github.com/your-username/heart-sound-classification",
        "Documentation": "https://github.com/your-username/heart-sound-classification/wiki",
    },
    keywords="heart sound classification, contrastive learning, deep learning, medical AI, temporal convolutional networks",
    include_package_data=True,
    zip_safe=False,
)
