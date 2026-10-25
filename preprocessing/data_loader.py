#!/usr/bin/env python3
"""
Enhanced Data Loading and Preprocessing Utilities

This module contains utility functions for loading heart sound data from H5 files
with support for careful rollover preprocessing and comprehensive data handling.

Supports both legacy data formats and new careful rollover preprocessing outputs
with enhanced metadata tracking and quality control features.
"""


import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple,Optional, Callable, Dict, List


class EnhancedHeartSoundDataset(Dataset):
    """
    Enhanced PyTorch Dataset for heart sound data with comprehensive features
    
    Supports:
    - Careful rollover preprocessing data with quality scores
    - Training-time augmentation
    - Metadata tracking (patient IDs, valve locations, quality scores)
    - Flexible data transformations
    
    Args:
        audio_data (np.ndarray): Audio waveforms (N, samples)
        labels (np.ndarray): Labels (N,)
        patient_ids (np.ndarray): Patient identifiers (optional)
        valves (np.ndarray): Valve locations (optional) 
        quality_scores (np.ndarray): Quality scores for rollover windows (optional)
        artifacts_counts (np.ndarray): Number of artifacts per window (optional)
        transform (callable, optional): Data transformation function
        augment (bool): Whether to apply light augmentation during training
    """
    
    def __init__(self, audio_data, labels, patient_ids=None, valves=None,  
                 quality_scores=None,artifacts_counts=None,transform=None, augment=False):
        self.audio_data = audio_data
        self.labels = labels
        self.patient_ids=patient_ids if patient_ids is not None else np.arange(len(audio_data))
        self.valves = valves if valves is not None else np.array(['Unknown'] * len(audio_data))
        self.quality_scores = quality_scores if quality_scores is not None else np.ones(len(audio_data))
        self.artifacts_counts = artifacts_counts if artifacts_counts is not None else np.zeros(len(audio_data))
        self.transform = transform
        self.augment = augment

        
        print(f"Dataset initialized: {len(self.audio_data)} samples")
        print(f"  Augmentation: {'ON' if augment else 'OFF'}")
        if quality_scores is not None:
            print(f"  Mean quality score: {np.mean(quality_scores):.3f}")
    
    def __len__(self):
        return len(self.audio_data)

    
    def __getitem__(self,idx):
        audio = self.audio_data[idx].astype(np.float32)
        label = self.labels[idx]

        # Convert string labels to numeric if needed
        if isinstance(label, (str, bytes)):
            if isinstance(label, bytes):
                label = label.decode('utf-8')
            label_numeric = 1 if label == 'Present' else 0
        else:
            label_numeric = int(label)

        #Apply transforms if specified
        if self.transform:
            audio = self.transform(audio)

        if self.augment:
            audio = self._apply_training_augmentation(audio) 
        return {
            'audio': torch.FloatTensor(audio),
            'label': torch.LongTensor([label_numeric]).squeeze(),
            'patient_id': str(self.patient_ids[idx]),
            'valve': str(self.valves[idx]),
            'quality_score': float(self.quality_scores[idx]),
            'artifacts_count': int(self.artifacts_counts[idx])
        }
    def _apply_training_augmentation(self, audio):
        """Apply light augmentation during training to improve robustness"""
        if np.random.random() < 0.2:
            noise_factor = np.random.uniform(0.001, 0.005)
            noise = np.random.normal(0,noise_factor,len(audio))
            audio = audio + noise

        
        if np.random.random() < 0.2:
            scale = np.random.uniform(0.95, 1.05) 
            audio = audio * scale

        return audio.astype(np.float32)
class EnhancedHeartSoundDataLoader:
    """
    Enhanced data loader wrapper for heart sound classification with rollover support
    
    Features:
    - Quality-based filtering for rollover windows
    - Weighted sampling for imbalanced data
    - Patient-aware validation splits
    - Comprehensive metadata handling
    - Support for both single-file and multi-file loading
    
    Args:
        data_path (str): Path to H5 data file (optional)
        batch_size (int): Batch size for training
        shuffle (bool): Whether to shuffle data
        num_workers (int): Number of worker processes
        use_weighted_sampling (bool): Whether to use weighted sampling for imbalanced data
        min_quality_threshold (float): Minimum quality score for filtering
    """

    def __init__(self, data_path=None, batch_size=32, shuffle=True, num_workers=0, 
                 use_weighted_sampling=False, min_quality_threshold=0.0):
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers 
        self.use_weighted_sampling = use_weighted_sampling
        self.min_quality_threshold = min_quality_threshold
        
        self.audio_data=None
        self.labels = None
        self.patient_ids=None
        self.valves = None
        self.quality_scores=None
        self.artifacts_counts=None


        if data_path:
            self.load_single_file(data_path) 
    
    def load_single_file(self, file_path): 
        """Load data from a single H5 file"""
        print(f"Loading data from: {file_path}")
        data = load_careful_rollover_data(file_path)

        self.audio_data = data['audio']
        self.labels = data['labels']
        self.patient_ids = data.get('patient_ids', None)
        self.valves = data.get('valves', None)
        self.quality_scores = data.get('quality_scores', None)
        self.artifacts_counts = data.get('artifacts_counts', None)
        
        self._apply_quality_filtering( )

    def load_prebalanced_files(self, train_path, val_path, test_path):
        """Load pre-balanced data from multiple careful rollover files"""
        print("Loading pre-balanced careful rollover datasets...")
        train_data = load_careful_rollover_data(train_path)
        val_data = load_careful_rollover_data(val_path)
        test_data = load_careful_rollover_data(test_path)

        return train_data, val_data, test_data

    def _apply_quality_filtering(self):
        """Apply quality-based filtering to the loaded data"""
        if self.quality_scores is None or self.min_quality_threshold <= 0:
            return
        
        #Create quality mask
        quality_mask=self.quality_scores >= self.min_quality_threshold

        if not np.any(quality_mask):
            #used to silently keep everything here, which hides a wrong threshold
            raise ValueError(f"No samples meet quality threshold {self.min_quality_threshold} "
                             f"(max quality score is {np.max(self.quality_scores):.2f})")

        # Apply filtering
        original_count = len(self.audio_data)
        self.audio_data=self.audio_data[quality_mask]
        self.labels = self.labels[quality_mask]
        
        if self.patient_ids is not None:
            self.patient_ids = self.patient_ids[quality_mask]
        if self.valves is not None:
            self.valves=self.valves[quality_mask]
        if self.quality_scores is not None:
            self.quality_scores = self.quality_scores[quality_mask]
        if self.artifacts_counts is not None:
            self.artifacts_counts = self.artifacts_counts[quality_mask]
        
        filtered_count = len(self.audio_data)
        print(f"Quality filtering: {filtered_count}/{original_count} samples kept "
              f"(threshold ≥ {self.min_quality_threshold:.2f})")

    def create_data_loaders(self, val_split=0.2, test_split=0.0, random_state=42,
                           augment_train=True, stratify=True):
        """
        Create train/val/test data loaders with enhanced features
        
        Args:
            val_split (float): Fraction of data for validation
            test_split (float): Fraction of data for testing
            random_state (int): Random seed
            augment_train (bool): Whether to augment training data
            stratify (bool): Whether to stratify splits
        
        Returns:
            tuple: Data loaders and class information
        """
        if self.audio_data is None:
            raise ValueError("No data loaded. Call load_single_file() first.")

        numeric_labels = self._convert_labels_to_numeric(self.labels)
        all_idx = np.arange(len(self.audio_data))

        if test_split > 0:
            idx_temp, idx_test = self._patient_split(
                all_idx, numeric_labels, test_split, random_state, stratify
            )
            idx_train,idx_val = self._patient_split( 
                idx_temp, numeric_labels, val_split / (1 - test_split), random_state, stratify
            )

            X_train, y_train=self.audio_data[idx_train], numeric_labels[idx_train]
            X_val, y_val = self.audio_data[idx_val], numeric_labels[idx_val]
            X_test, y_test=self.audio_data[idx_test], numeric_labels[idx_test]

            # Create datasets with metadata
            train_dataset = self._create_dataset_with_metadata(X_train, y_train, idx_train, augment_train)
            val_dataset = self._create_dataset_with_metadata(X_val, y_val, idx_val, False)
            test_dataset = self._create_dataset_with_metadata(X_test, y_test, idx_test, False)

            #Create data loaders
            train_loader = self._create_train_loader(train_dataset, y_train)
            val_loader=DataLoader(val_dataset, batch_size=self.batch_size,
                                   shuffle=False,num_workers=self.num_workers)
            test_loader = DataLoader(test_dataset, batch_size = self.batch_size,
                                    shuffle=False, num_workers=self.num_workers)
            class_info = self._compute_class_info(y_train, y_val, y_test)

            
            return train_loader, val_loader, test_loader, class_info
        else:
            # Two-way split (by patient if we have patient ids)
            idx_train,idx_val = self._patient_split(
                all_idx,numeric_labels,val_split,random_state, stratify 
            )
            X_train, y_train = self.audio_data[idx_train], numeric_labels[idx_train]
            X_val,y_val = self.audio_data[idx_val],numeric_labels[idx_val]

            train_dataset = self._create_dataset_with_metadata( X_train, y_train, idx_train, augment_train)
            val_dataset = self._create_dataset_with_metadata(X_val, y_val, idx_val, False)

            
            train_loader = self._create_train_loader(train_dataset, y_train)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, 
                                   shuffle=False, num_workers=self.num_workers)

            class_info = self._compute_class_info(y_train, y_val)

            return train_loader, val_loader, class_info

    def _patient_split(self, indices, numeric_labels, test_size, random_state, stratify):
        """Split window indices so the same patient never ends up on both sides"""
        labels = numeric_labels[indices]

        if self.patient_ids is None:
            # no patient info -> window level split (can leak patients!)
            print("Warning: no patient_ids, splitting by window")
            return train_test_split(indices, test_size=test_size, random_state=random_state,
                                    stratify=labels if stratify else None)

        patients = self.patient_ids[indices]
        unique_patients = np.unique(patients)
        patient_labels = np.array([labels[patients == p].max() for p in unique_patients])

        train_patients,test_patients = train_test_split(
            unique_patients,test_size=test_size, random_state=random_state,
            stratify = patient_labels if stratify else None
        )
        in_test=np.isin(patients, test_patients)
        return indices[~in_test],indices[in_test]

    def _convert_labels_to_numeric(self, labels):
        """Convert string labels to numeric format"""
        if isinstance(labels[0],(str,bytes)):
            numeric_labels=[]
            for label in labels:
                if isinstance(label,bytes):
                    label = label.decode('utf-8')
                numeric_labels.append(1 if label == 'Present' else 0)
            return np.array(numeric_labels)
        return labels.astype(int) 
    def _create_dataset_with_metadata(self, audio_data, labels, indices, augment):
        """Create dataset with full metadata"""
        # Extract metadata for selected indices
        patient_ids = self.patient_ids[indices] if self.patient_ids is not None else None
        valves = self.valves[indices] if self.valves is not None else None
        quality_scores=self.quality_scores[indices] if self.quality_scores is not None else None
        artifacts_counts=self.artifacts_counts[indices] if self.artifacts_counts is not None else None
        
        return EnhancedHeartSoundDataset(
            audio_data, labels, patient_ids, valves, 
            quality_scores,artifacts_counts, augment=augment
        )
    
    def _create_train_loader(self, dataset, labels):
        """Create training data loader with optional weighted sampling"""
        if self.use_weighted_sampling:
            # Compute class weights
            unique_labels = np.unique(labels)
            class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
            weight_map = dict(zip(unique_labels, class_weights))
            sample_weights = [weight_map[label] for label in labels]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            
            return DataLoader(dataset,batch_size=self.batch_size,
                            sampler=sampler, num_workers=self.num_workers)
        else:
            return DataLoader(dataset,batch_size = self.batch_size,
                            shuffle = self.shuffle,num_workers=self.num_workers)

    def _compute_class_info(self, *label_arrays):
        """Compute comprehensive class distribution information"""
        info = {}
        split_names = ['train', 'val', 'test'][:len(label_arrays)]

        for i, labels in enumerate(label_arrays):
            unique, counts=np.unique(labels, return_counts = True)
            info[split_names[i]] = {
                'counts': dict(zip(unique, counts)),
                'total': len(labels),
                'percentages': dict(zip(unique, counts / len(labels) * 100))
            }

        #Overall class weights
        all_labels = np.concatenate(label_arrays)
        unique_all=np.unique(all_labels)
        class_weights = compute_class_weight('balanced', classes=unique_all, y=all_labels)
        info['class_weights'] = dict(zip(unique_all, class_weights))
        
        return info

def load_careful_rollover_data( file_path: str) -> Dict[str,np.ndarray]:
    """
    Load careful rollover processed heart sound data from H5 file
    
    Supports both legacy formats and new careful rollover preprocessing outputs.
    Automatically handles string/numeric label conversion and metadata extraction.
    
    Parameters:
    -----------
    file_path : str
        Path to the H5 file
        
    Returns:
    --------
    dict
        Dictionary containing audio, labels, and comprehensive metadata
    """
    data={}
    metadata = {}
    
    print(f"Loading careful rollover data from: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        # Display available keys
        available_keys=list(f.keys())
        print(f"Available datasets: {available_keys}")

        # Load audio data (murmur preprocessor files used to call it 'windows')
        audio_key = 'audio' if 'audio' in f else 'windows'
        data['audio'] = f[audio_key][:]

        # Load and process labels
        labels_raw = f['labels'][:]
        if labels_raw.dtype.kind == 'S':
            labels_str = [label.decode('utf-8') if isinstance(label, bytes) else str(label) 
                         for label in labels_raw]
            data['labels'] = np.array(labels_str)
            data['labels_numeric'] = np.array([1 if label == 'Present' else 0 for label in labels_str])
            label_type = "String (Present/Absent)"
        else:
            data['labels'] = labels_raw.astype(int)
            data['labels_numeric'] = labels_raw.astype(int)
            label_type = "Numeric (0/1)"

        # Load metadata
        metadata_keys = ['patient_ids', 'valves', 'quality_scores', 'artifacts_counts']
        for key in metadata_keys:
            if key in f:
                if key in ['patient_ids', 'valves'] and f[key].dtype.kind == 'S':
                    data[key] = np.array([item.decode('utf-8') if isinstance(item, bytes) 
                                        else str(item) for item in f[key][:]])
                else:
                    data[key] = f[key][:]
                metadata[key] = f"Loaded {key}"
            else:
                metadata[key] = f"Not available"

        # Load file attributes (preprocessing metadata)
        attrs = dict(f.attrs)
        metadata['file_attributes'] = attrs
    
    # Print comprehensive dataset information
    print(f"\nDataset Information:")
    print(f"Total samples: {len(data['audio'])}")
    print(f"Audio shape: {data['audio'].shape}")

    
    if 'window_length' in metadata.get('file_attributes', {}):
        window_length = metadata['file_attributes']['window_length']
        sampling_rate = metadata['file_attributes'].get('sampling_rate', 4000)
        duration=window_length / sampling_rate
        print(f"Window: {window_length} samples ({duration:.2f}s at {sampling_rate}Hz)")

    preprocessing_method = metadata.get('file_attributes', {}).get('preprocessing_method', 'Unknown')
    print(f"Preprocessing method: {preprocessing_method}")
    print(f"Label format: {label_type}")
    
    # Class distribution
    present_count = np.sum(data['labels_numeric'] == 1)
    absent_count = np.sum(data['labels_numeric'] == 0)
    print(f"\nClass Distribution:")
    print(f"Murmur Present: {present_count} samples ({present_count/len(data['labels'])*100:.2f}%)")
    print(f"Murmur Absent: {absent_count} samples ({absent_count/len(data['labels'])*100:.2f}%)")


    if 'quality_scores' in data:
        quality_scores = data['quality_scores']
        print(f"\nQuality Information:")
        print(f"Mean quality score: {np.mean(quality_scores):.3f} ± {np.std(quality_scores):.3f}")
        print(f"Quality range: [{np.min(quality_scores):.3f}, {np.max(quality_scores):.3f}]")

        
        #Quality distribution 
        high_quality = np.sum(quality_scores >= 0.8)
        medium_quality = np.sum((quality_scores >= 0.6) & (quality_scores < 0.8))
        low_quality = np.sum(quality_scores < 0.6)
        print(f"High quality (≥0.8): {high_quality} samples")
        print(f"Medium quality (0.6-0.8): {medium_quality} samples")
        print(f"Low quality (<0.6): {low_quality} samples")

    if 'artifacts_counts' in data:
        artifacts = data['artifacts_counts']
        print(f"Mean artifacts per window: {np.mean(artifacts):.2f}")
        print(f"Windows with no artifacts: {np.sum(artifacts == 0)}")

    # Metadata summary
    print(f"\nMetadata Summary:")
    for key, status in metadata.items():
        if key != 'file_attributes':
            print(f"  {key}: {status}")
    return data


def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legacy function for backward compatibility
    Load heart sound segments and binary labels from H5 file
    """
    data = load_careful_rollover_data(file_path)
    return data['audio'], data['labels_numeric']



def split_data(segments: np.ndarray, labels: np.ndarray,  
               test_size: float = 0.2, val_size: float = 0.1) -> Tuple[np.ndarray, ...]:
    """
    Split data into training, validation, and test sets with stratification
    
    Parameters:
    -----------
    segments : numpy.ndarray
        Heart sound segments
    labels : numpy.ndarray
        Labels
    test_size : float
        Proportion of data for testing
    val_size : float
        Proportion of data for validation
        
    Returns:
    --------
    tuple
        (x_train, x_val, x_test, y_train, y_val, y_test)
    """
    if isinstance(labels[0], (str, bytes)):
        numeric_labels = []
        for label in labels:
            if isinstance(label, bytes):
                label = label.decode('utf-8')
            numeric_labels.append(1 if label == 'Present' else 0)
        labels = np.array(numeric_labels)
    
    #with patient_ids if windows from the same patient must not leak across splits
    # First split off test set (train_test_split doesn't accept test_size=0)
    if test_size > 0:
        x_train_val,x_test,y_train_val,y_test = train_test_split(
            segments,labels,test_size=test_size, stratify=labels, random_state=42
        )
    else:
        x_train_val, y_train_val = segments, labels 
        x_test, y_test = segments[:0], labels[:0]
    
    val_size_adjusted = val_size / (1 - test_size)
    x_train,x_val, y_train, y_val = train_test_split(
        x_train_val,y_train_val, test_size=val_size_adjusted, 
        stratify=y_train_val, random_state=42
    )

    print(f"Data split:")
    print(f"  Training: {len(x_train)} samples")
    print(f"  Validation: {len(x_val)} samples")
    print(f"  Test: {len(x_test)} samples")
    
    return x_train, x_val, x_test, y_train, y_val, y_test


def load_prebalanced_data(train_path: str, val_path: str, test_path: str) -> Tuple[np.ndarray, ...]:
    """
    Load pre-balanced training, validation, and test data from careful rollover H5 files
    
    Parameters:
    -----------
    train_path : str
        Path to pre-balanced training data
    val_path : str
        Path to validation data  
    test_path : str
        Path to test data
        
    Returns:
    --------
    tuple
        (x_train, x_val, x_test, y_train, y_val, y_test)
    """
    print("Loading pre-balanced careful rollover datasets...")

    train_data = load_careful_rollover_data(train_path)
    x_train, y_train = train_data['audio'], train_data['labels_numeric']
    
    print("\nLoading validation data...")
    val_data = load_careful_rollover_data(val_path)
    x_val, y_val = val_data['audio'], val_data['labels_numeric']

    print("\nLoading test data...")
    test_data = load_careful_rollover_data(test_path)
    x_test, y_test = test_data['audio'], test_data['labels_numeric']

    return x_train, x_val, x_test, y_train, y_val, y_test


def create_quality_filtered_loader(file_path: str, min_quality: float = 0.8, 
                                  batch_size: int = 32,augment: bool = False) -> DataLoader:
    """
    Create a data loader with quality filtering for careful rollover windows
    
    Parameters:
    -----------
    file_path : str
        Path to H5 file with quality scores
    min_quality : float
        Minimum quality score threshold
    batch_size : int
        Batch size
    augment : bool
        Whether to apply augmentation
        
    Returns:
    --------
    DataLoader
        Quality-filtered data loader with comprehensive metadata
    """
    print(f"Creating quality-filtered loader (threshold ≥ {min_quality:.2f})")

    loader = EnhancedHeartSoundDataLoader(
        data_path = file_path, 
        batch_size = batch_size,
        min_quality_threshold=min_quality
    )
    
    if loader.audio_data is not None and len(loader.audio_data) > 0:
        dataset = EnhancedHeartSoundDataset(
            loader.audio_data,  
            loader.labels,
            loader.patient_ids,
            loader.valves,
            loader.quality_scores,
            loader.artifacts_counts,
            augment=augment
        )
        return DataLoader( dataset,batch_size=batch_size,shuffle=False)
    else:
        print("No data available after quality filtering")
        return None


class HeartSoundDataset( Dataset):
    """Legacy Dataset class for backward compatibility"""
    def __init__( self, audio_data, labels, transform=None):
        self.audio_data = audio_data
        self.labels=labels
        self.transform = transform
    def __len__(self):
        return len(self.audio_data)
    
    def __getitem__(self,idx):
        audio = self.audio_data[idx]
        label = self.labels[idx]

        if self.transform:
            audio = self.transform(audio)

        return torch.FloatTensor(audio), torch.LongTensor([label]).squeeze()


# Legacy data loader class  
class HeartSoundDataLoader:
    """Legacy DataLoader class for backward compatibility"""
    
    def __init__(self, data_path, batch_size = 32, shuffle=True, num_workers=0):
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers=num_workers

        self.audio_data,self.labels=load_data(data_path)

    def create_data_loaders(self,val_split=0.2,test_split=0.0,random_state=42):
        """Create data loaders using legacy interface"""
        if test_split > 0:
            x_train, x_val, x_test, y_train, y_val, y_test = split_data(
                self.audio_data,self.labels,test_split, val_split 
            )
            
            train_dataset = HeartSoundDataset(x_train, y_train)
            val_dataset = HeartSoundDataset(x_val, y_val)
            test_dataset = HeartSoundDataset(x_test, y_test)

            train_loader = DataLoader( train_dataset,batch_size=self.batch_size,
                                    shuffle = self.shuffle, num_workers=self.num_workers) 
            val_loader = DataLoader(val_dataset,batch_size=self.batch_size,
                                   shuffle=False,num_workers=self.num_workers)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                    shuffle = False, num_workers=self.num_workers)

            return train_loader,val_loader,test_loader
        else:
            x_train, x_val, _, y_train, y_val, _ = split_data(
                self.audio_data, self.labels, 0.0, val_split
            )
            
            train_dataset = HeartSoundDataset( x_train, y_train)
            val_dataset=HeartSoundDataset( x_val,y_val)

            train_loader = DataLoader(train_dataset, batch_size = self.batch_size,
                                    shuffle=self.shuffle, num_workers=self.num_workers)
            val_loader=DataLoader( val_dataset, batch_size=self.batch_size,
                                   shuffle = False, num_workers=self.num_workers)

            return train_loader, val_loader