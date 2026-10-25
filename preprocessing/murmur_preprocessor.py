"""
Murmur-Centered Data Preprocessing Module

This module contains advanced murmur-centered preprocessing with 4,000-sample windows
for heart sound classification with sophisticated training and evaluation strategies.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
import librosa
import glob
from tqdm import tqdm
import pickle
from typing import Dict, List, Tuple, Union, Optional
import h5py
from sklearn.model_selection import train_test_split
import random


class MurmurCenteredPreprocessor:
    """
    Advanced murmur-centered preprocessing with 4,000-sample window construction:
    
    For Training:
    - 4,000-sample windows with rollover procedure
    - 2,000-sample "murmur core" from concatenated systolic segments
    - 1,000 samples before and after the core
    - Cross-fade joins (20-50ms) and exclusion margins
    - Minimum 200ms murmur overlap required
    - Soft labels based on overlap fraction
    
    For Development/Test:
    - Contiguous, non-concatenated windows only
    """

    def __init__(self, data_path: str, csv_path: str, sampling_rate: int=4000):
        self.data_path = data_path
        self.csv_path = csv_path
        self.sampling_rate = sampling_rate
        
        # Window configuration (4,000 samples = 1 second at 4kHz)
        self.total_window_samples = 4000
        self.core_samples = 2000
        self.flank_samples = 1000
        
        # Cross-fade configuration
        self.min_crossfade_ms = 20        #20ms minimum crossfade
        self.max_crossfade_ms=50        # 50ms maximum crossfade 
        self.min_crossfade_samples=int(self.min_crossfade_ms * self.sampling_rate / 1000)
        self.max_crossfade_samples = int(self.max_crossfade_ms * self.sampling_rate / 1000)
        
        self.min_murmur_overlap_ms = 200 
        self.min_murmur_overlap_samples = int(self.min_murmur_overlap_ms * self.sampling_rate / 1000)
        self.exclusion_margin_ratio = 0.5  # Exclusion margin ≥ 0.5 × flank length
        self.exclusion_margin_samples = int(self.flank_samples * self.exclusion_margin_ratio)

        self.window_length=self.total_window_samples / self.sampling_rate  # seconds
        self.window_samples = self.total_window_samples
        self.min_murmur_overlap = self.min_murmur_overlap_ms / 1000.0  # seconds 
        self.augment_margin = 0.05  # +-50ms random nudge of the window center

        self.patient_info = self.load_patient_info()

        print(f"Preprocessor Configuration:")
        print(f"  Total window: {self.total_window_samples} samples (1.0s)")
        print(f"  Murmur core: {self.core_samples} samples (0.5s)")
        print(f"  Flanks: {self.flank_samples} samples each (0.25s)")
        print(f"  Cross-fade: {self.min_crossfade_ms}-{self.max_crossfade_ms}ms")
        print(f"  Min overlap: {self.min_murmur_overlap_ms}ms")
        print(f"  Exclusion margin: {self.exclusion_margin_samples} samples")
        
    def load_patient_info(self) -> pd.DataFrame:
        """Load and process patient information from CSV"""
        df=pd.read_csv(self.csv_path)
        print(f"Loaded information for {len(df)} patients")
        
        #Print murmur distribution
        murmur_counts = df['Murmur'].value_counts()
        print("Murmur distribution:")
        for status, count in murmur_counts.items():
            print(f"  {status}: {count} patients")
            
        return df

    def get_split_recordings(self, split_type: str, max_patients: int = None) -> pd.DataFrame: 
        """Get recordings for a specific data split (train/dev/test)."""
        patient_df = self.patient_info
        if max_patients:
            patient_df = patient_df.head(max_patients)

        # drop Unknown murmur patients, they were ending up as negatives before
        patient_df = patient_df[patient_df['Murmur'].isin(['Present', 'Absent'])]
        all_patients = patient_df['Patient ID'].values
        murmur = patient_df['Murmur'].values

        # Patient-aware stratified split: 70% train, 15% dev, 15% test
        train_patients, temp_patients, _, temp_murmur = train_test_split(
            all_patients,murmur,test_size=0.3,random_state=42, stratify=murmur
        )
        dev_patients,test_patients = train_test_split( 
            temp_patients, test_size=0.5, random_state=42, stratify=temp_murmur
        )

        if split_type == 'train':
            patients = train_patients
        elif split_type == 'dev':
            patients = dev_patients
        elif split_type == 'test':
            patients = test_patients
        else:
            raise ValueError(f"Invalid split_type: {split_type}")
        
        recordings = []
        valve_types = ['AV', 'MV', 'PV', 'TV']

        for patient_id in patients:
            patient_info = self.patient_info[self.patient_info['Patient ID'] == patient_id]
            if len(patient_info) > 0:
                patient_row = patient_info.iloc[0]
                murmur_status = patient_row['Murmur']
                murmur_locations = str(patient_row['Murmur locations']).split('+') if pd.notna(patient_row['Murmur locations']) else []

                for valve in valve_types:
                    #label per valve: for murmur patients only the valves where the murmur
                    if murmur_status == 'Present' and valve not in murmur_locations:
                        continue

                    wav_path = os.path.join(self.data_path, f"{patient_id}_{valve}.wav")
                    if os.path.exists(wav_path):
                        recordings.append({
                            'patient_id': patient_id,
                            'recording': f"{patient_id}_{valve}",
                            'murmur': murmur_status,
                            'valve': valve
                        })

        
        return pd.DataFrame(recordings)
    
    def load_recording_data(self, recording_name: str) -> Tuple[Optional[np.ndarray], Optional[List[Dict]]]:
        """Load audio and annotation data for a recording."""
        wav_path = os.path.join(self.data_path, f"{recording_name}.wav")
        tsv_path = os.path.join(self.data_path, f"{recording_name}.tsv")
        
        if not os.path.exists(wav_path) or not os.path.exists(tsv_path):
            return None,None
            
        try:
            audio, sr = librosa.load( wav_path, sr=self.sampling_rate)
            
            #Load annotations
            annotations = []
            with open(tsv_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        start_time = float(parts[0])
                        end_time = float(parts[1])
                        label = int(parts[2])
                        annotations.append({
                            'start': start_time,
                            'end': end_time,
                            'label': label,
                            'duration': end_time - start_time
                        }) 
            
            return audio,annotations

        except Exception as e:
            print(f"Error loading {recording_name}: {e}")
            return None, None

    def extract_systolic_intervals(self, annotations: List[Dict]) -> List[Dict]:
        """Extract systolic intervals from annotations."""
        systolic_intervals = [ann for ann in annotations if ann['label'] == 2]
        return systolic_intervals
    
    def balance_classes(self, *data_arrays, method: str = 'undersample') -> List:
        """Balance classes using the specified method."""
        windows, labels = data_arrays[0], data_arrays[1] 

        if method == 'undersample':
            unique_labels, counts=np.unique(labels, return_counts=True)
            min_count = min(counts)
            
            balanced_indices=[]
            for label in unique_labels:
                label_indices = np.where(np.array(labels) == label)[0]
                selected_indices = np.random.choice(label_indices, min_count, replace=False)
                balanced_indices.extend(selected_indices)

            np.random.shuffle(balanced_indices)
            
            balanced_data = []
            for data_array in data_arrays:
                if isinstance(data_array,list): 
                    balanced_data.append([data_array[i] for i in balanced_indices])
                else: 
                    balanced_data.append(np.array(data_array)[balanced_indices])
            
            return balanced_data
        
        elif method == 'none':
            return list( data_arrays)

        else:
            raise ValueError(f"Unsupported balance method: {method}")

            
        return list(data_arrays)

    def load_audio_and_annotations(self, patient_id: str, valve: str) -> Tuple[np.ndarray, List[Dict], int]: 
        """Load audio file and corresponding annotations"""
        wav_path = os.path.join(self.data_path, f"{patient_id}_{valve}.wav")
        tsv_path = os.path.join(self.data_path, f"{patient_id}_{valve}.tsv")

        if not os.path.exists(wav_path) or not os.path.exists(tsv_path):
            return None,None,None
            
        try:
            audio, sr = librosa.load(wav_path, sr=self.sampling_rate)
        except:
            return None,None,None

        annotations = []
        with open(tsv_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    start_time = float(parts[0])
                    end_time = float(parts[1]) 
                    label = int(parts[2])
                    annotations.append({
                        'start': start_time,
                        'end': end_time,
                        'label': label,
                        'duration': end_time - start_time
                    })
        
        return audio, annotations, sr

    
    def get_systolic_intervals(self, annotations: List[Dict]) -> List[Dict]:
        """Extract systolic intervals (label = 2) from annotations"""
        systolic_intervals = []
        for ann in annotations:
            if ann['label'] == 2:
                systolic_intervals.append(ann) 
        return systolic_intervals

    def extract_positive_windows(self, audio: np.ndarray, patient_id: str, valve: str, 
                                systolic_intervals: List[Dict], is_training: bool = False) -> List[Dict]:
        """
        Extract positive windows centered on murmur annotations.
        For patients with murmurs, center 1s windows on systolic intervals from murmur-present valves.
        """
        positive_windows = []
        
        # Getting patient murmur info
        patient_row = self.patient_info[self.patient_info['Patient ID'] == int(patient_id)]
        if len(patient_row) == 0:
            return positive_windows

        patient_row = patient_row.iloc[0]
        murmur_status = patient_row['Murmur']

        print(f"Processing positive windows for patient {patient_id}, valve {valve}, murmur_status: {murmur_status}")


        if murmur_status != 'Present':
            return positive_windows

        murmur_locations = str(patient_row['Murmur locations']).split('+') if pd.notna(patient_row['Murmur locations']) else []
        print(f"  Murmur locations: {murmur_locations}")
        
        if valve not in murmur_locations:
            return positive_windows

        print(f"  Valve {valve} has murmur, extracting positive samples from {len(systolic_intervals)} systolic intervals")
        # Extracting windows from each systolic interval
        for i, interval in enumerate(systolic_intervals):
            start_time = interval['start']
            end_time = interval['end']
            duration=end_time - start_time
            print(f"    Interval {i}: duration={duration:.3f}s")
            
            # Calculating window center (midpoint of systolic interval)
            center_time=(start_time + end_time) / 2.0
            
            if is_training:
                nudge = np.random.uniform(-self.augment_margin, self.augment_margin)
                center_time += nudge
            
            # Calculating window boundaries
            window_start = center_time - self.window_length / 2.0
            window_end = center_time + self.window_length / 2.0
            
            overlap_start = max(window_start, start_time)
            overlap_end = min(window_end,end_time)
            overlap_duration = max(0, overlap_end - overlap_start)

            # Ensuring minimum overlap with systolic interval
            if overlap_duration >= self.min_murmur_overlap:
                start_sample=int(window_start * self.sampling_rate)
                end_sample=int(window_end * self.sampling_rate)

                if start_sample >= 0 and end_sample <= len(audio):
                    window_audio = audio[start_sample:end_sample]
                    
                    if len(window_audio) == self.window_samples: 
                        print(f"      Extracted positive window from interval {i}")
                        positive_windows.append({
                            'audio': window_audio,
                            'label': 'Present',
                            'patient_id': patient_id,
                            'valve': valve,
                            'window_start': window_start,
                            'window_end': window_end,
                            'murmur_overlap': overlap_duration,
                            'systolic_start': start_time,
                            'systolic_end': end_time
                        })
                    else:
                        print(f"      Wrong audio length: {len(window_audio)} != {self.window_samples}")
                else:
                    print(f"      Window out of bounds: start_sample={start_sample}, end_sample={end_sample}")
            else:
                print(f"      Insufficient overlap: {overlap_duration:.3f}s < {self.min_murmur_overlap:.3f}s")
        print(f"  Generated {len(positive_windows)} positive windows")
        return positive_windows

    def extract_negative_windows(self,audio: np.ndarray, patient_id: str, valve: str, 
                                systolic_intervals: List[Dict], positive_windows: List[Dict],
                                is_training: bool = False) -> List[Dict]:
        """
        Extract negative windows from systolic intervals of murmur-absent patients
        or from systolic intervals that don't contain murmurs.
        """
        negative_windows = []
        # Get patient murmur info
        patient_row = self.patient_info[self.patient_info['Patient ID'] == int(patient_id)]
        if len(patient_row) == 0:
            return negative_windows
        patient_row = patient_row.iloc[0]
        murmur_status = patient_row['Murmur']
        print(f"Processing negative windows for patient {patient_id}, valve {valve}, murmur_status: {murmur_status}")

        
        if murmur_status == 'Absent':
            print(f"  Processing murmur-absent patient, {len(systolic_intervals)} systolic intervals")
            for i, interval in enumerate( systolic_intervals):
                start_time = interval['start']
                end_time = interval['end']
                duration = end_time - start_time
                print(f"    Interval {i}: duration={duration:.3f}s")

                
                center_time=(start_time + end_time) / 2.0
                window_start=center_time - self.window_length / 2.0
                window_end = center_time + self.window_length / 2.0
                start_sample = int(window_start * self.sampling_rate)
                end_sample = int(window_end * self.sampling_rate)

                if start_sample >= 0 and end_sample <= len(audio):
                    window_audio = audio[start_sample:end_sample]

                    
                    if len( window_audio) == self.window_samples:
                        print(f"      Extracted window from interval {i}")
                        negative_windows.append({
                            'audio': window_audio,
                            'label': 'Absent',
                            'patient_id': patient_id,
                            'valve': valve,
                            'window_start': window_start,
                            'window_end': window_end,
                            'systolic_start': start_time,
                            'systolic_end': end_time
                        })
                    else:
                        print(f"      Wrong audio length: {len(window_audio)} != {self.window_samples}")
                else:
                    print(f"      Window out of bounds: start_sample={start_sample}, end_sample={end_sample}, audio_len={len(audio)}")

        elif murmur_status == 'Present':
            murmur_locations = str(patient_row['Murmur locations']).split('+') if pd.notna(patient_row['Murmur locations']) else []
            print(f"  Processing murmur-present patient, murmur locations: {murmur_locations}")

            #If this valve doesn't have murmur, use its systolic intervals as negative
            if valve not in murmur_locations:
                print(f"    Valve {valve} not in murmur locations, extracting negative samples")
                for i,interval in enumerate(systolic_intervals):
                    start_time = interval['start']
                    end_time = interval['end']
                    duration=end_time - start_time
                    
                    center_time = (start_time + end_time) / 2.0
                    window_start = center_time - self.window_length / 2.0
                    window_end = center_time + self.window_length / 2.0

                    start_sample = int(window_start * self.sampling_rate)
                    end_sample = int(window_end * self.sampling_rate)
                    if start_sample >= 0 and end_sample <= len( audio):
                        window_audio = audio[start_sample:end_sample]

                        
                        if len(window_audio) == self.window_samples:
                            negative_windows.append({ 
                                'audio': window_audio,
                                'label': 'Absent',
                                'patient_id': patient_id,
                                'valve': valve,
                                'window_start': window_start,
                                'window_end': window_end,
                                'systolic_start': start_time,
                                'systolic_end': end_time
                            })
            else:
                print(f"    Valve {valve} has murmur, skipping negative sample extraction for now")

        print(f"  Generated {len(negative_windows)} negative windows")
        return negative_windows

    def process_patient_recording(self,patient_id: str,valve: str,is_training: bool = False) -> List[Dict]:
        """Process a single patient recording and extract windows"""
        windows = []
        
        audio, annotations, sr = self.load_audio_and_annotations(patient_id, valve)
        if audio is None:
            return windows 

        systolic_intervals = self.get_systolic_intervals( annotations)
        if not systolic_intervals:
            return windows 
            
        positive_windows = self.extract_positive_windows( 
            audio,patient_id,valve,systolic_intervals, is_training
        )
        
        negative_windows = self.extract_negative_windows( 
            audio, patient_id, valve, systolic_intervals, positive_windows, is_training
        )

        
        windows.extend(positive_windows)
        windows.extend(negative_windows)

        return windows
    def patient_aware_split(self, test_size: float = 0.2, val_size: float = 0.2, 
                           random_state: int=42) -> Tuple[List, List, List]:
        """
        Perform patient-aware stratified split to prevent data leakage
        """
        # Get unique patients and their murmur status
        patients_df = self.patient_info[['Patient ID', 'Murmur']].copy()
        patients_df = patients_df[patients_df['Murmur'].isin(['Present', 'Absent'])]  #Exclude Unknown
        
        print(f"Total patients for splitting: {len(patients_df)}")
        print("Patient distribution:")
        print(patients_df['Murmur'].value_counts())
        train_patients, temp_patients = train_test_split(
            patients_df['Patient ID'].values,
            test_size=(test_size + val_size),
            stratify=patients_df['Murmur'].values,
            random_state=random_state
        )


        temp_df = patients_df[patients_df['Patient ID'].isin(temp_patients)]
        val_patients, test_patients=train_test_split(
            temp_df['Patient ID'].values,
            test_size=(test_size / (test_size + val_size)),
            stratify=temp_df['Murmur'].values,
            random_state = random_state
        )

        print(f"\nPatient split:")
        print(f"Train: {len(train_patients)} patients")
        print(f"Val: {len(val_patients)} patients") 
        print(f"Test: {len(test_patients)} patients")
        return train_patients.tolist(), val_patients.tolist(), test_patients.tolist()
    
    def apply_augmentation_balancing(self, audio_data: np.ndarray, labels: np.ndarray, 
                                   all_windows: List[Dict],augment_factor: int = 3) -> Tuple[np.ndarray,np.ndarray,List[Dict]]:
        """
        Balance classes through augmentation of minority class
        """
        #Count samples per class 
        unique_labels, counts = np.unique(labels, return_counts=True) 
        print(f"Before balancing: {dict(zip(unique_labels, counts))}")

        if len(unique_labels) != 2:
            print("Warning: Expected 2 classes for balancing")
            return audio_data, labels, all_windows

        minority_idx = np.argmin(counts)
        majority_idx = np.argmax(counts)
        minority_class = unique_labels[minority_idx]
        majority_class = unique_labels[majority_idx]
        minority_count=counts[minority_idx]
        majority_count=counts[majority_idx]

        print(f"Minority class: {minority_class} ({minority_count} samples)")
        print(f"Majority class: {majority_class} ({majority_count} samples)")

        #Calculated how many augmented samples needed
        target_count = majority_count
        additional_needed = target_count - minority_count
        
        if additional_needed <= 0:
            print("Classes already balanced or minority is larger")
            return audio_data,labels,all_windows

        print(f"Generating {additional_needed} additional {minority_class} samples")
        
        # Get minority class samples
        minority_mask = labels == minority_class 
        minority_audio = audio_data[minority_mask]
        minority_windows = [w for i,w in enumerate(all_windows) if minority_mask[i]]

        augmented_audio = []
        augmented_labels = []
        augmented_windows=[] 

        augmented_audio.extend( audio_data)
        augmented_labels.extend(labels)
        augmented_windows.extend(all_windows)

        # Generating augmented samples
        for _ in range(additional_needed): 
            # Randomly select a minority sample
            idx = random.randint(0, len(minority_audio) - 1)
            original_audio = minority_audio[idx].copy()
            original_window = minority_windows[idx].copy( )
            
            augmentation_type = random.choice(['noise', 'time_shift', 'amplitude', 'filter'])
            
            if augmentation_type == 'noise':
                # Add Gaussian noise (0.001 to 0.01)
                noise_factor = random.uniform( 0.001, 0.01)
                noise = np.random.normal(0, noise_factor, original_audio.shape)
                augmented = original_audio + noise

                
            elif augmentation_type == 'time_shift':
                # Circular shift by ±5% of length
                shift_samples = random.randint(-len(original_audio)//20, len(original_audio)//20)
                augmented = np.roll(original_audio,shift_samples)

            elif augmentation_type == 'amplitude':
                #Amplitude scaling ±20%
                scale_factor = random.uniform(0.8, 1.2)
                augmented = original_audio * scale_factor
                
            elif augmentation_type == 'filter':
                try:
                    low_freq = random.uniform(50, 100)
                    high_freq=random.uniform(1000,2000)
                    nyquist=self.sampling_rate / 2
                    low = low_freq / nyquist
                    high = high_freq / nyquist
                    b, a = butter(2, [low, high], btype='band')
                    augmented = filtfilt(b, a, original_audio)
                except:
                    augmented=original_audio 
            
            #Ensuring amplitude range is preserved
            augmented = np.clip(augmented,-1.0, 1.0)

            #Now adding to augmented dataset 
            augmented_audio.append( augmented)
            augmented_labels.append(minority_class)
            aug_window = original_window.copy()
            aug_window['augmented'] = True
            aug_window['augmentation_type'] = augmentation_type
            augmented_windows.append(aug_window)
        
        final_audio = np.array(augmented_audio)
        final_labels = np.array(augmented_labels)
        
        shuffle_indices = np.random.permutation(len(final_audio))
        final_audio = final_audio[shuffle_indices]
        final_labels = final_labels[shuffle_indices]
        final_windows = [augmented_windows[i] for i in shuffle_indices] 
        
        final_unique, final_counts = np.unique(final_labels, return_counts = True)
        print(f"After balancing: {dict(zip(final_unique, final_counts))}")

        return final_audio, final_labels, final_windows
    def process_recording_rollover(self,patient_id: str,recording_name: str, 
                                 murmur_label: str, augment: bool=True) -> Dict:
        """
        Process a single recording using rollover windowing strategy for training.
        
        This implements the advanced 4,000-sample window construction:
        - Creates 2,000-sample "murmur core" from concatenated systolic segments
        - Adds 1,000 samples before and after the core (flanks)
        - Uses cross-fade joins and exclusion margins
        - Calculates soft labels based on murmur overlap
        """
        try:
            # Load audio and annotations
            audio, annotations = self.load_recording_data(recording_name)
            if audio is None or annotations is None:
                return {'windows': [], 'labels': [], 'metadata': [], 'soft_labels': []}

            # Extract systolic intervals
            systolic_intervals = self.extract_systolic_intervals( annotations) 
            if not systolic_intervals:
                return {'windows': [], 'labels': [], 'metadata': [], 'soft_labels': []}

            
            windows = []
            labels=[]
            soft_labels=[]
            metadata = []

            has_murmur = (murmur_label == 'Present')

            if has_murmur:
                murmur_windows=self.create_murmur_core_windows( 
                    audio, systolic_intervals, patient_id, recording_name
                )
                windows.extend([w['audio'] for w in murmur_windows])
                labels.extend([1] * len(murmur_windows))  #Binary label
                soft_labels.extend([w['soft_label'] for w in murmur_windows])
                metadata.extend([w['metadata'] for w in murmur_windows])
            else:
                # For non-murmur recordings build the windows the exact same way (systolic core + flanks) 
                background_windows = self.create_murmur_core_windows(
                    audio,systolic_intervals,patient_id,recording_name
                )
                for w in background_windows:
                    w['metadata']['type'] = 'normal_core'
                windows.extend([w['audio'] for w in background_windows])
                labels.extend([0] * len(background_windows))  # Binary label
                soft_labels.extend([0.0] * len(background_windows))
                metadata.extend([w['metadata'] for w in background_windows])
            
            return {
                'windows': windows,
                'labels': labels,
                'soft_labels': soft_labels,
                'metadata': metadata
            }

        except Exception as e:
            print(f"Error processing {recording_name}: {e}")
            return {'windows': [], 'labels': [], 'metadata': [], 'soft_labels': []}

    def process_recording_contiguous(self, patient_id: str, recording_name: str, 
                                   murmur_label: str) -> Dict:
        """
        Process a single recording using contiguous windowing for dev/test.
        
        Uses non-overlapping, contiguous windows without concatenation.
        """
        try:
            audio, annotations = self.load_recording_data(recording_name)
            if audio is None or annotations is None:
                return {'windows': [], 'labels': [], 'metadata': []}
            
            systolic_intervals = self.extract_systolic_intervals(annotations)
            if not systolic_intervals:
                return {'windows': [], 'labels': [], 'metadata': []}


            windows = [] 
            labels=[]
            metadata = []
            
            # Create contiguous windows
            contiguous_windows = self.create_contiguous_windows(
                audio, systolic_intervals, patient_id, recording_name, murmur_label
            )

            windows.extend([w['audio'] for w in contiguous_windows])
            labels.extend([w['label'] for w in contiguous_windows])
            metadata.extend([w['metadata'] for w in contiguous_windows])

            return {
                'windows': windows,
                'labels': labels,
                'metadata': metadata
            } 
        except Exception as e:
            print(f"Error processing {recording_name}: {e}")
            return {'windows': [], 'labels': [], 'metadata': []}

    def create_murmur_core_windows(self, audio: np.ndarray, systolic_intervals: List[Dict], 
                                 patient_id: str, recording_name: str) -> List[Dict]:
        """
        Create training windows with murmur cores from concatenated systolic segments.
        """
        windows = []
        # Concatenate all systolic segments to create murmur material 
        systolic_segments = []
        for interval in systolic_intervals:
            start_sample = int(interval['start'] * self.sampling_rate)
            end_sample = int(interval['end'] * self.sampling_rate)
            
            if start_sample < len(audio) and end_sample <= len(audio):
                segment = audio[start_sample:end_sample]
                systolic_segments.append(segment)

        if not systolic_segments:
            return windows

        # Concatenate with cross-fades
        concatenated_audio = self.concatenate_with_crossfade(systolic_segments)

        if len(concatenated_audio) < self.core_samples:
            return windows 


        max_start = len(concatenated_audio) - self.core_samples
        num_windows=min(10, max_start // 200 + 1)  # Up to 10 windows, spaced by 200 samples

        for i in range(num_windows):
            core_start = i * (max_start // max(num_windows - 1, 1))
            core_end = core_start + self.core_samples

            murmur_core = concatenated_audio[core_start:core_end]
            window_audio, soft_label = self.add_flanks_to_core(
                murmur_core,audio, systolic_intervals
            )
            
            if window_audio is not None:
                windows.append({
                    'audio': window_audio,
                    'soft_label': soft_label,
                    'metadata': {
                        'patient_id': patient_id,
                        'recording': recording_name,
                        'type': 'murmur_core',
                        'core_start': core_start,
                        'core_end': core_end,
                        'num_systolic_segments': len(systolic_segments)
                    }
                })
        
        return windows

    def create_background_windows( self, audio: np.ndarray, systolic_intervals: List[Dict],
                                patient_id: str, recording_name: str) -> List[Dict]:
        """
        Create background windows from non-murmur recordings.
        """
        windows=[]

        # Find gaps between systolic intervals for background material
        background_segments = []
        audio_duration=len(audio) / self.sampling_rate
        
        #Add segment before first systolic interval
        if systolic_intervals[0]['start'] > 0.5:
            start_sample = 0
            end_sample = int((systolic_intervals[0]['start'] - 0.1) * self.sampling_rate)
            background_segments.append(audio[start_sample:end_sample])

        # Add segments between systolic intervals 
        for i in range(len(systolic_intervals) - 1):
            gap_start = systolic_intervals[i]['end'] + 0.1
            gap_end = systolic_intervals[i + 1]['start'] - 0.1  # 0.1s before start

            if gap_end - gap_start > 0.5:
                start_sample = int(gap_start * self.sampling_rate)
                end_sample = int(gap_end * self.sampling_rate)
                background_segments.append( audio[start_sample:end_sample])

        # Add segment after last systolic interval
        if audio_duration - systolic_intervals[-1]['end'] > 0.5:
            start_sample = int((systolic_intervals[-1]['end'] + 0.1) * self.sampling_rate)
            end_sample = len(audio)
            background_segments.append(audio[start_sample:end_sample])
        
        # Create windows from background segments
        for segment in background_segments: 
            if len(segment) >= self.total_window_samples:
                num_windows = len(segment) // self.total_window_samples
                for i in range(min(num_windows, 3)):
                    start_idx = i * self.total_window_samples
                    end_idx = start_idx + self.total_window_samples
                    window_audio = segment[start_idx:end_idx]

                    windows.append({
                        'audio': window_audio,
                        'metadata': {
                            'patient_id': patient_id,
                            'recording': recording_name,
                            'type': 'background',
                            'segment_index': len(windows)
                        }
                    })

        return windows
    
    def create_contiguous_windows(self, audio: np.ndarray, systolic_intervals: List[Dict],
                                patient_id: str,recording_name: str, murmur_label: str) -> List[Dict]:
        """
        Create contiguous windows for dev/test sets.
        """
        windows = []
        label = 1 if murmur_label == 'Present' else 0
        
        total_samples=len(audio) 
        
        for start_sample in range( 0, total_samples - self.total_window_samples + 1, 
                                self.total_window_samples):
            end_sample = start_sample + self.total_window_samples
            window_audio=audio[start_sample:end_sample]

            windows.append({ 
                'audio': window_audio,
                'label': label,
                'metadata': {
                    'patient_id': patient_id,
                    'recording': recording_name,
                    'type': 'contiguous',
                    'start_time': start_sample / self.sampling_rate,
                    'end_time': end_sample / self.sampling_rate
                } 
            })

        
        return windows
    def concatenate_with_crossfade(self, segments: List[np.ndarray]) -> np.ndarray:
        """
        Concatenate audio segments with cross-fade transitions.
        """
        if len(segments) == 1:
            return segments[0]

        
        concatenated = segments[0].copy()


        for segment in segments[1:]:
            # Determine cross-fade length (20-50ms)
            crossfade_samples = np.random.randint(
                self.min_crossfade_samples, 
                self.max_crossfade_samples + 1
            )
            crossfade_samples = min(crossfade_samples,  
                                  min(len(concatenated), len(segment)) // 2) 

            if crossfade_samples > 0:
                # Create cross-fade
                fade_out = np.linspace(1, 0, crossfade_samples)
                fade_in = np.linspace(0, 1, crossfade_samples)
                
                # Apply cross-fade
                concatenated[-crossfade_samples:] *= fade_out
                concatenated[-crossfade_samples:] += segment[:crossfade_samples] * fade_in

                #Append remaining part of segment
                concatenated = np.concatenate([concatenated, segment[crossfade_samples:]]) 
            else:
                concatenated = np.concatenate([concatenated,segment])
        return concatenated

    def add_flanks_to_core(self, murmur_core: np.ndarray, audio: np.ndarray, 
                          systolic_intervals: List[Dict]) -> tuple:
        """
        Add flanks before and after murmur core, calculate soft label.
        """
        #Find suitable flank regions with exclusion margins
        suitable_regions = []
        
        for i, interval in enumerate(systolic_intervals):
            start_time = interval['start']
            end_time = interval['end']
            
            # Before systolic interval (with exclusion margin)
            before_end=start_time - (self.exclusion_margin_samples / self.sampling_rate)
            before_start = before_end - (self.flank_samples / self.sampling_rate)

            if before_start >= 0:
                suitable_regions.append(('before', before_start, before_end))
            
            # After systolic interval (with exclusion margin)
            after_start = end_time + (self.exclusion_margin_samples / self.sampling_rate)
            after_end = after_start + (self.flank_samples / self.sampling_rate)
            if after_end <= len(audio) / self.sampling_rate:
                suitable_regions.append(('after', after_start, after_end))

        if len(suitable_regions) < 2:
            return None, 0.0
        
        #(has to actually be a 'before' region and an 'after' region, the type was ignored before)
        np.random.shuffle(suitable_regions)
        before_region = None
        after_region = None

        for region_type, start_time, end_time in suitable_regions:
            if region_type == 'before' and before_region is None:
                before_region = (start_time, end_time)
            elif region_type == 'after' and after_region is None:
                after_region = (start_time, end_time)
            if before_region is not None and after_region is not None:
                break

        if before_region is None or after_region is None:
            return None, 0.0

        # Extract flank audio
        before_start_sample = int(before_region[0] * self.sampling_rate)
        before_end_sample = int(before_region[1] * self.sampling_rate)
        before_flank = audio[before_start_sample:before_end_sample]

        
        after_start_sample=int(after_region[0] * self.sampling_rate)
        after_end_sample = int(after_region[1] * self.sampling_rate)
        after_flank = audio[after_start_sample:after_end_sample]
        before_flank=before_flank[:self.flank_samples]
        after_flank = after_flank[:self.flank_samples]
        
        if len(before_flank) < self.flank_samples:
            before_flank = np.pad(before_flank, (0, self.flank_samples - len(before_flank)))

        if len(after_flank) < self.flank_samples:
            after_flank = np.pad(after_flank, (0, self.flank_samples - len(after_flank)))
        
        # Combine flanks and core
        window_audio = np.concatenate([before_flank, murmur_core, after_flank])
        # Calculate soft label (fraction of window that contains murmur)
        murmur_overlap_samples = len(murmur_core)
        soft_label = murmur_overlap_samples / self.total_window_samples

        min_overlap_fraction = self.min_murmur_overlap_samples / self.total_window_samples
        if soft_label < min_overlap_fraction:
            return None, 0.0

        return window_audio, soft_label
    def process_all_recordings(self, split_type: str = 'train', 
                             augment: bool = True, 
                             balance_method: str = 'undersample',
                             save_processed: bool = True,
                             output_dir: str = None,
                             max_patients: int = None) -> Dict:
        """
        Process all recordings for the specified data split using advanced windowing.
        
        Args:
            split_type: 'train', 'dev', or 'test'
            augment: Whether to apply data augmentation (train only)
            balance_method: Method for class balancing ('undersample', 'oversample', 'weighted')
            save_processed: Whether to save processed data to H5 file
            
        Returns:
            Dictionary containing processed data and metadata
        """
        if split_type not in ['train', 'dev', 'test']:
            raise ValueError(f"Invalid split_type: {split_type}")

        #Get recordings for this split
        recordings_info=self.get_split_recordings(split_type,max_patients=max_patients)

        print(f"\nProcessing {split_type} split with {'rollover' if split_type == 'train' else 'contiguous'} windowing:")
        print(f"Found {len(recordings_info)} recordings")
        
        all_windows = []
        all_labels = []
        all_soft_labels = []
        all_patient_ids=[]
        all_metadata = []
        
        #Progress bar for recordings
        for _,row in tqdm(recordings_info.iterrows(), 
                          total=len(recordings_info),
                          desc=f"Processing {split_type} recordings"):
            
            patient_id = str(row['patient_id'])
            recording_name = row['recording']
            murmur_label = row['murmur']
            
            # Process this recording with appropriate strategy
            if split_type == 'train':
                recording_data = self.process_recording_rollover(
                    patient_id=patient_id,
                    recording_name = recording_name,
                    murmur_label=murmur_label,
                    augment=augment 
                )
            else:
                recording_data = self.process_recording_contiguous(
                    patient_id = patient_id,
                    recording_name=recording_name,
                    murmur_label = murmur_label
                )


            if recording_data['windows']:
                all_windows.extend(recording_data['windows'])
                all_labels.extend(recording_data['labels'])
                all_patient_ids.extend([patient_id] * len(recording_data['windows']))
                all_metadata.extend(recording_data['metadata'])

                
                #Handle soft labels for training
                if split_type == 'train' and 'soft_labels' in recording_data:
                    all_soft_labels.extend(recording_data['soft_labels'])
        print(f"\nExtracted {len(all_windows)} windows")
        print(f"Label distribution: {np.bincount(all_labels)}")

        if split_type == 'train' and balance_method != 'none':
            balance_data = [all_windows, all_labels, all_patient_ids, all_metadata]
            if all_soft_labels:
                balance_data.append(all_soft_labels)
            
            balanced_data = self.balance_classes( *balance_data, method=balance_method)

            all_windows = balanced_data[0]
            all_labels = balanced_data[1]
            all_patient_ids = balanced_data[2]
            all_metadata=balanced_data[3]
            if all_soft_labels:
                all_soft_labels = balanced_data[4]

            print(f"After balancing: {len(all_windows)} windows")
            print(f"Balanced label distribution: {np.bincount(all_labels)}")

        processed_data = {
            'windows': np.array(all_windows),
            'labels': np.array(all_labels),
            'patient_ids': np.array(all_patient_ids),
            'metadata': all_metadata,
            'sampling_rate': self.sampling_rate,
            'window_samples': self.total_window_samples,
            'split_type': split_type
        }
        
        if split_type == 'train' and all_soft_labels:
            processed_data['soft_labels'] = np.array(all_soft_labels)
        
        # Save to file if requested
        if save_processed:
            filename = f"murmur_processed_{split_type}_4000samples.h5"
            if output_dir is None:
                output_dir = os.path.join(self.data_path, 'processed')
            save_path = os.path.join(output_dir,filename)
            os.makedirs(os.path.dirname(save_path), exist_ok = True)
            
            with h5py.File(save_path, 'w') as f:
                # 'audio' key so load_careful_rollover_data can read it
                f.create_dataset('audio', data=processed_data['windows'], compression='gzip')
                f.create_dataset('labels', data=processed_data['labels'])
                f.create_dataset('patient_ids', data=processed_data['patient_ids'].astype('S'))
                f.attrs['sampling_rate'] = self.sampling_rate
                f.attrs['window_samples'] = self.total_window_samples
                f.attrs['split_type'] = split_type
                
                if 'soft_labels' in processed_data:
                    f.create_dataset('soft_labels', data=processed_data['soft_labels'])

                # as a dataset, attributes are limited to 64KB and this gets way bigger than that 
                metadata_json = json.dumps(processed_data['metadata'])
                f.create_dataset('metadata', data=metadata_json)
            print(f"Saved processed data to: {save_path}")

        
        return processed_data


    def process_dataset(self, output_dir: str, max_patients: int = None, balance_train: bool = True, 
                       balance_val: bool=False, augment_factor: int = 3):
        """
        Main processing pipeline with advanced 4,000-sample windowing strategy.
        
        Now uses the new rollover windowing for training and contiguous windowing for dev/test.
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"Advanced Preprocessing Configuration:")
        print(f"  Training: Rollover windowing with 4,000-sample windows")
        print(f"  Dev/Test: Contiguous windowing")
        print(f"  Balance training set: {balance_train}")
        print(f"  Balance validation set: {balance_val}")
        
        for split_type in ['train', 'dev', 'test']:
            print(f"\n{'='*50}")
            print(f"Processing {split_type.upper()} split")
            print(f"{'='*50}")
            
            processed_data=self.process_all_recordings(
                split_type = split_type,
                augment=(split_type == 'train'),
                balance_method='undersample' if (split_type == 'train' and balance_train) or 
                               (split_type == 'dev' and balance_val) else 'none',
                save_processed=True,
                output_dir = output_dir,
                max_patients=max_patients
            )

            print(f"\n{split_type.upper()} Results:")
            print(f"  Total windows: {len(processed_data['windows'])}")
            print(f"  Label distribution: {np.bincount(processed_data['labels'])}")
            if 'soft_labels' in processed_data:
                print(f"  Soft label range: {np.min(processed_data['soft_labels']):.3f} - {np.max(processed_data['soft_labels']):.3f}")
        print(f"\nAdvanced preprocessing completed!")
        print(f"Processed files saved to: {output_dir}/")
        
        return True
def preprocess_segments(segments, lowcut = 20, highcut=500, sampling_rate=4000, visualize=True, labels=None):
    """
    Apply bandpass filtering and standardization to heart sound segments
    
    Parameters:
    -----------
    segments : numpy.ndarray
        Heart sound segments
    lowcut : int
        Low cutoff frequency for bandpass filter (Hz)
    highcut : int
        High cutoff frequency for bandpass filter (Hz)
    sampling_rate : int
        Sampling rate of the audio (Hz)
    visualize : bool
        Whether to visualize original vs processed segments
    labels : numpy.ndarray
        Labels for visualization (if visualize=True)
        
    Returns:
    --------
    numpy.ndarray
        Preprocessed segments
    """
    print(f"\nApplying bandpass filtering ({lowcut}-{highcut} Hz) and standardization...")
    
    # Define bandpass filter parameters 
    nyquist = 0.5 * sampling_rate
    low=lowcut / nyquist
    high = highcut / nyquist

    order=4
    b, a = butter(order, [low, high], btype='band')

    processed_segments=np.zeros_like(segments)
    for i in range(len(segments)):
        filtered_segment = filtfilt(b,a,segments[i])
        
        mean_val = np.mean(filtered_segment)
        std_val = np.std(filtered_segment)
        if std_val > 0:
            processed_segments[i] = (filtered_segment - mean_val) / std_val
        else:
            processed_segments[i] = filtered_segment - mean_val

    return processed_segments
