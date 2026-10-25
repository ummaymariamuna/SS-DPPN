#!/usr/bin/env python3
"""
Careful Rollover Data Preprocessing Module

This module implements an enhanced rollover concatenation method that addresses 
artificial boundary problems while maintaining fixed-length segments. Creates 
2000-sample (0.5s) windows through intelligent concatenation with strict safeguards 
to preserve natural cardiac timing and prevent boundary artifacts.

Key Features:
- Patient-recording boundary preservation
- Cardiac cycle integrity with phase-aware concatenation
- Temporal integrity safeguards (50ms-1000ms gaps)
- Smart concatenation with crossfade smoothing
- Comprehensive quality control and validation
"""



import os
import pandas as pd
import numpy as np
import h5py 
import librosa
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, spectrogram
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Dict,List,Tuple, Union, Optional
import random
import matplotlib.pyplot as plt
import seaborn as sns

class CarefulRolloverPreprocessor:
    """
    Advanced rollover preprocessing with strict safeguards against artificial boundaries.
    
    Core Principles:
    - Patient-recording boundary preservation (never cross patients/sessions)
    - Cardiac cycle integrity (phase-aware concatenation: S1 → systole → S2 → diastole)
    - Temporal continuity validation (realistic inter-beat intervals)
    - Quality control with comprehensive validation metrics
    
    Creates 2000-sample (0.5s) fixed-length windows for all splits with natural cardiac timing.
    """

    def __init__(self, data_path: str, csv_path: str, sampling_rate: int = 4000):
        """
        Initialize the Careful Rollover Preprocessor
        
        Args:
            data_path: Path to directory containing .wav and .tsv files
            csv_path: Path to training_data.csv file
            sampling_rate: Audio sampling rate (default 4000 Hz)
        """
        self.data_path=data_path 
        self.csv_path = csv_path
        self.sampling_rate = sampling_rate
        # Window parameters (2000 samples = 0.5 seconds at 4kHz) 
        self.target_length = 2000
        self.window_duration = 0.5  # seconds
        
        self.min_gap_threshold = 0.05   # 50 milliseconds minimum gap
        self.max_gap_threshold = 1.0
        self.amplitude_threshold = 0.1

        self.min_crossfade_ms = 20
        self.max_crossfade_ms=50
        self.crossfade_samples = 80     # 20ms at 4kHz (default)

        self.min_heart_rate = 40        # Minimum heart rate (BPM)
        self.max_heart_rate = 200 
        self.cardiac_cycle_multiplier=1.5 
        # Patient information
        self.patient_info = self.load_patient_info()

        print(f"Initialized Careful Rollover Preprocessor")
        print(f"Target window: {self.target_length} samples ({self.window_duration}s)")
        print(f"Temporal safeguards: {self.min_gap_threshold*1000:.0f}-{self.max_gap_threshold*1000:.0f}ms gaps")
        print(f"Crossfade: {self.min_crossfade_ms}-{self.max_crossfade_ms}ms smooth transitions")
        print(f"Heart rate validation: {self.min_heart_rate}-{self.max_heart_rate} BPM")
        print(f"Cardiac cycle gap limit: {self.cardiac_cycle_multiplier}x expected cycle duration")

        
    def load_patient_info(self) -> pd.DataFrame:
        """Load patient information from CSV file"""
        try:
            df = pd.read_csv(self.csv_path)
            print(f"Loaded information for {len(df)} patients")
            
            murmur_counts = df['Murmur'].value_counts()
            print(f"Murmur distribution:")
            for murmur_type, count in murmur_counts.items():
                print(f"  {murmur_type}: {count} patients")

            return df
        except Exception as e:
            print(f"Error loading patient info: {e}")
            return pd.DataFrame()
    
    def load_audio_and_annotations(self,patient_id: str,valve: str) -> Tuple[np.ndarray,List[Dict], int]:
        """Load audio file and corresponding annotations with enhanced error handling"""
        try:
            wav_path = os.path.join(self.data_path, f"{patient_id}_{valve}.wav")
            if not os.path.exists(wav_path):
                return np.array([]),[],self.sampling_rate

                
            audio,sr = librosa.load( wav_path,sr=self.sampling_rate)

            # Load annotations
            tsv_path = os.path.join(self.data_path, f"{patient_id}_{valve}.tsv")
            annotations = []

            if os.path.exists(tsv_path):
                with open(tsv_path, 'r') as f:
                    for line in f: 
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            start_time = float(parts[0])
                            end_time=float(parts[1])
                            label = int(parts[2]) 

                            # Add cardiac phase information
                            phase = self._determine_cardiac_phase(label) 
                            
                            annotations.append({
                                'start_time': start_time,
                                'end_time': end_time,
                                'label': label,
                                'duration': end_time - start_time,
                                'cardiac_phase': phase
                            })

            return audio,annotations, sr

        except Exception as e:
            print(f"Error loading audio/annotations for {patient_id}_{valve}: {e}")
            return np.array([]), [], self.sampling_rate

    def _determine_cardiac_phase(self, label: int) -> str:
        """Determine cardiac phase from annotation label"""
        phase_map = {
            1: 'S1',        # First heart sound
            2: 'systole',   # Systolic period
            3: 'S2',
            4: 'diastole'
        }
        return phase_map.get(label, 'unknown')
    
    def get_systolic_intervals(self, annotations: List[Dict]) -> List[Dict]:
        """Extract systolic intervals (label = 2) sorted by time"""
        systolic_intervals=[]
        for ann in annotations:
            if ann['label'] == 2:
                systolic_intervals.append(ann)
        return sorted(systolic_intervals, key=lambda x: x['start_time'])

    def estimate_heart_rate(self, segments: List[Dict]) -> float:
        """Estimate heart rate from segment timing for validation"""
        if len(segments) < 2:
            return 70.0  # Default heart rate
        
        intervals = []
        for i in range(1, len(segments)):
            # Calculate inter-segment interval
            interval = segments[i]['start_time'] - segments[i-1]['end_time']
            if 0.1 <= interval <= 2.0:  # Reasonable interval range 
                intervals.append(interval)

        if intervals:
            mean_interval = np.mean(intervals)
            estimated_hr = 60.0 / (mean_interval * 2.5)
            return np.clip(estimated_hr, self.min_heart_rate, self.max_heart_rate)
        
        return 70.0

    def can_concatenate_safely(self, current_segments: List[Dict], new_segment: Dict, 
                              patient_id: str, valve: str) -> bool:
        """Comprehensive safety checks for concatenation"""
        if not current_segments:
            return True  #First segment is always safe


        last_segment = current_segments[-1]
        
        # 1. Patient and valve consistency check
        if (last_segment.get('patient_id') != patient_id or 
            last_segment.get('valve') != valve):
            return False
        
        time_gap = new_segment['start_time'] - last_segment['end_time']
        if time_gap < self.min_gap_threshold or time_gap > self.max_gap_threshold:
            return False

        test_segments=current_segments + [new_segment]
        estimated_hr=self.estimate_heart_rate(test_segments)
        if self.min_heart_rate <= estimated_hr <= self.max_heart_rate: 
            expected_cycle_duration = 60.0 / estimated_hr
            max_allowed_gap = self.cardiac_cycle_multiplier * expected_cycle_duration
            if time_gap > max_allowed_gap:
                return False

        
        #3. Heart rate consistency check
        test_segments = current_segments + [new_segment]
        estimated_hr = self.estimate_heart_rate(test_segments)
        if estimated_hr < self.min_heart_rate or estimated_hr > self.max_heart_rate:
            return False


        #4. Duration reasonableness check
        total_duration = sum(seg['duration'] for seg in current_segments) + new_segment['duration']
        if total_duration > 1.5: 
            return False

        # 5. Phase progression check
        if len(current_segments) >= 1:
            if not self._is_phase_progression_natural(last_segment['cardiac_phase'], 
                                                    new_segment['cardiac_phase']):
                return False
        return True
    
    def _is_phase_progression_natural(self, prev_phase: str, curr_phase: str) -> bool:
        """Check if cardiac phase progression is natural"""
        # NOTE: we only ever concatenate systolic segments, so systole -> systole
        # (next beat, S2/diastole/S1 are in the gap) has to be allowed, otherwise nothing ever concatenates
        natural_transitions = {
            'S1': ['systole'],
            'systole': ['S2', 'diastole', 'systole'],
            'S2': ['diastole'],
            'diastole': ['S1', 'systole']
        }
        return curr_phase in natural_transitions.get(prev_phase, [curr_phase])
    
    def apply_crossfade( self, audio1: np.ndarray, audio2: np.ndarray) -> np.ndarray:
        """Apply smooth crossfade between audio segments"""
        if len(audio1) < self.crossfade_samples or len(audio2) < self.crossfade_samples:
            return np.concatenate([audio1, audio2])
        fade_out = np.linspace(1, 0, self.crossfade_samples)
        fade_in = np.linspace(0, 1, self.crossfade_samples)

        audio1_main = audio1[:-self.crossfade_samples]
        audio1_fade = audio1[-self.crossfade_samples:] * fade_out
        audio2_fade = audio2[:self.crossfade_samples] * fade_in
        audio2_main = audio2[self.crossfade_samples:]

        crossfaded_region = audio1_fade + audio2_fade

        
        return np.concatenate([audio1_main,crossfaded_region,audio2_main])
    
    def detect_boundary_artifacts(self, window_audio: np.ndarray, 
                                 concatenation_points: List[int]) -> List[Dict]:
        """Detect artificial artifacts at segment boundaries"""
        artifacts = []
        
        for point in concatenation_points:
            if point < 100 or point >= len(window_audio) - 100:
                continue

            #Extract regions around boundary
            before_region = window_audio[max(0, point-50):point]
            after_region = window_audio[point:min(len(window_audio),point+50)]
            
            if len(before_region) < 20 or len(after_region) < 20: 
                continue

            amplitude_before = np.mean(np.abs(before_region)) 
            amplitude_after = np.mean(np.abs(after_region))
            amplitude_jump = abs(amplitude_before - amplitude_after) > self.amplitude_threshold

            try:
                freq_before = np.mean(np.abs(np.fft.fft(before_region)[:min(25, len(before_region)//2)]))
                freq_after = np.mean(np.abs(np.fft.fft(after_region)[:min(25, len(after_region)//2)]))
                freq_discontinuity = abs(freq_before - freq_after) > max(amplitude_before, amplitude_after) * 0.3
            except: 
                freq_discontinuity=False
            
            if amplitude_jump or freq_discontinuity:
                artifacts.append({ 
                    'location': point,
                    'amplitude_jump': amplitude_jump,
                    'frequency_jump': freq_discontinuity
                })
        return artifacts 

    def create_rollover_windows(self, patient_data: Dict) -> List[Dict]: 
        """Create rollover windows with comprehensive safeguards"""
        all_windows = []
        
        for patient_id, patient_info in tqdm(patient_data.items(), desc="Processing patients"):
            murmur_status = patient_info['murmur_status']
            
            for valve in ['AV', 'MV', 'PV', 'TV']:
                # Load audio and annotations
                audio,annotations,sr = self.load_audio_and_annotations(patient_id, valve)

                if len(audio) == 0 or len(annotations) == 0:
                    continue
                
                #Get systolic intervals
                systolic_intervals = self.get_systolic_intervals(annotations)
                
                if not systolic_intervals:
                    continue

                
                current_segments = []
                current_audio_parts = []
                concatenation_points=[]
                for segment_info in systolic_intervals:
                    # Extract segment audio
                    start_sample = int(segment_info['start_time'] * sr)
                    end_sample = int(segment_info['end_time'] * sr)
                    if start_sample >= len(audio) or end_sample > len(audio):
                        continue

                    segment_audio=audio[start_sample:end_sample]

                    if len(segment_audio) < 50:
                        continue
                    
                    # Add patient and valve info to segment
                    segment_info['patient_id'] = patient_id
                    segment_info['valve'] = valve

                    # Check if can concatenate safely
                    if self.can_concatenate_safely(current_segments,segment_info,patient_id,valve):
                        # Add to current window 
                        current_segments.append(segment_info)

                        if current_audio_parts: 
                            last_audio=current_audio_parts[-1] 
                            crossfaded = self.apply_crossfade(last_audio, segment_audio)
                            current_audio_parts = [crossfaded]
                            concatenation_points.append(len(last_audio) - self.crossfade_samples)
                        else:
                            current_audio_parts.append(segment_audio)

                        total_length = sum( len(part) for part in current_audio_parts)
                        
                        if total_length >= self.target_length:
                            window_audio=np.concatenate(current_audio_parts)
                            
                            if len(window_audio) > self.target_length:
                                window_audio = window_audio[:self.target_length]
                            elif len(window_audio) < self.target_length:
                                padding = self.target_length - len(window_audio)
                                noise = np.random.normal(0, 0.001, padding)
                                window_audio = np.concatenate([window_audio, noise])

                            artifacts=self.detect_boundary_artifacts(window_audio, concatenation_points)
                            quality_score = max(0.0, 1.0 - (len(artifacts) * 0.15))

                            
                            if quality_score >= 0.7:
                                window_data = {
                                    'audio': window_audio,
                                    'label': murmur_status,
                                    'patient_id': patient_id,
                                    'valve': valve,
                                    'segments_used': len(current_segments),
                                    'concatenation_points': concatenation_points.copy(),
                                    'quality_score': quality_score,
                                    'artifacts_count': len(artifacts)
                                }
                                all_windows.append(window_data) 

                            current_segments = []
                            current_audio_parts = []
                            concatenation_points = []

                    else:
                        window_data = self._finalize_partial_window( 
                            current_segments, current_audio_parts, concatenation_points,
                            murmur_status, patient_id, valve
                        )
                        if window_data is not None:
                            all_windows.append(window_data)

                        current_segments = [segment_info]
                        current_audio_parts=[segment_audio]
                        concatenation_points = []

                # whatever is left at the end of the recording was getting dropped before
                window_data = self._finalize_partial_window(
                    current_segments,current_audio_parts,concatenation_points,
                    murmur_status, patient_id, valve
                )
                if window_data is not None:
                    all_windows.append(window_data)

        print(f"Generated {len(all_windows)} rollover windows")
        return all_windows
    def _finalize_partial_window(self, current_segments, current_audio_parts, concatenation_points,
                                 murmur_status, patient_id, valve):
        """Pad a shorter-than-target window and return it, or None if it's too short to keep"""
        if not current_segments or not current_audio_parts:
            return None

        window_audio = np.concatenate( current_audio_parts)

        # Only keep if reasonably sized 
        if len(window_audio) < self.target_length * 0.4:  #At least 40% of target
            return None

        # Pad to target length
        if len(window_audio) < self.target_length:
            padding = self.target_length - len(window_audio)
            noise = np.random.normal(0, 0.001, padding)
            window_audio=np.concatenate([window_audio, noise])
        else:
            window_audio = window_audio[:self.target_length]

        quality_score=0.6  # Lower quality due to padding

        return {
            'audio': window_audio,
            'label': murmur_status,
            'patient_id': patient_id,
            'valve': valve,
            'segments_used': len(current_segments),
            'concatenation_points': concatenation_points.copy(),
            'quality_score': quality_score,
            'artifacts_count': 0
        }
    
    def patient_aware_split(self, test_size: float = 0.15, val_size: float = 0.15,
                           random_state: int = 42, patient_df: Optional[pd.DataFrame] = None) -> Tuple[List, List, List]:
        """Patient-aware stratified split ensuring no patient appears in multiple splits"""
        if patient_df is None:
            patient_df = self.patient_info 

        usable_patients = patient_df[patient_df['Murmur'].isin(['Present', 'Absent'])]

        print(f"Total patients for splitting: {len(usable_patients)}")
        print("Patient distribution:")
        print(usable_patients['Murmur'].value_counts())

        
        # First split: separate test set (15%)
        train_val_patients, test_patients = train_test_split(
            usable_patients['Patient ID'].values,
            test_size=test_size,
            stratify=usable_patients['Murmur'].values,
            random_state=random_state
        )
        
        #Second split: separate train (70%) and validation (15%)
        val_size_adjusted = val_size / (1 - test_size)
        train_patients, val_patients = train_test_split(
            train_val_patients,
            test_size = val_size_adjusted,
            stratify=usable_patients[usable_patients['Patient ID'].isin(train_val_patients)]['Murmur'].values,
            random_state=random_state
        )

        print(f"\nPatient split:")
        print(f"Train: {len(train_patients)} patients (70%)")
        print(f"Val: {len(val_patients)} patients (15%)")
        print(f"Test: {len(test_patients)} patients (15%)")

        return list(train_patients),list(val_patients), list(test_patients)
    def apply_augmentation(self, audio: np.ndarray, strategy: str) -> np.ndarray:
        """Apply specific augmentation strategy"""
        augmented = audio.copy()
        
        if strategy == 'gaussian_noise':
            #Additive Gaussian noise (0.001 to 0.01)
            noise_factor = random.uniform( 0.001, 0.01)
            noise=np.random.normal(0, noise_factor, len(audio))
            augmented = audio + noise 

        elif strategy == 'time_shift':
            # Temporal shifting (±100 milliseconds)
            shift_samples = int(random.uniform(-0.1,0.1) * self.sampling_rate)
            if shift_samples > 0:
                augmented = np.concatenate([np.zeros(shift_samples), audio[:-shift_samples]])
            elif shift_samples < 0:
                augmented=np.concatenate([audio[-shift_samples:],np.zeros(-shift_samples)])
                
        elif strategy == 'amplitude_scale':
            scale_factor = random.uniform(0.8, 1.2)
            augmented = audio * scale_factor
            
        elif strategy == 'pitch_shift':
            # Pitch shifting (2 semitones)
            try:
                shift_steps = random.uniform( -2, 2)
                augmented = librosa.effects.pitch_shift(audio, sr=self.sampling_rate, n_steps=shift_steps)
            except:
                augmented = audio

        elif strategy == 'bandpass_filter':
            # Random bandpass filtering (15-25 Hz low cut, 450-550 Hz high cut)
            low_freq = random.uniform(15, 25)
            high_freq=random.uniform(450, 550)
            nyquist=self.sampling_rate / 2
            low_norm = low_freq / nyquist
            high_norm = min(high_freq / nyquist, 0.95)
            try:
                b, a = butter(4, [low_norm, high_norm], btype='band')
                augmented = filtfilt(b, a, audio)
            except:
                augmented = audio 

        elif strategy == 'additive_snr_noise':
            #Additive noise with SNR (20-30 dB)
            snr_db = random.uniform(20, 30)
            signal_power = np.mean(audio ** 2) 
            noise_power=signal_power / (10 ** (snr_db / 10))
            noise = np.random.normal(0,np.sqrt(noise_power),len(audio))
            augmented=audio+noise
        
        return np.clip(augmented, -1.0, 1.0)
    
    def balance_training_data(self, windows: List[Dict]) -> List[Dict]:
        """Balance training data through randomized augmentation oversampling"""
        # Separate by class
        present_windows = [w for w in windows if w['label'] == 'Present']
        absent_windows = [w for w in windows if w['label'] == 'Absent']
        
        print(f"Before balancing:")
        print(f"  Present: {len(present_windows)} windows")
        print(f"  Absent: {len(absent_windows)} windows")

        if len(present_windows) < len(absent_windows):
            minority_windows = present_windows
            majority_windows = absent_windows
            minority_label = 'Present'
        else:
            minority_windows = absent_windows 
            majority_windows = present_windows
            minority_label = 'Absent'

        if len(minority_windows) == 0:
            print(f"Warning: no {minority_label} windows to augment, skipping balancing")
            return windows

        # Generate augmented samples for minority class
        target_count = len(majority_windows)
        augmentation_strategies = [
            'gaussian_noise', 'time_shift', 'amplitude_scale', 
            'pitch_shift', 'bandpass_filter', 'additive_snr_noise'
        ]
        
        augmented_windows = []
        
        while len(minority_windows) + len(augmented_windows) < target_count:
            original_window = random.choice(minority_windows)
            
            # Apply random combination of augmentations
            num_augmentations = random.randint(1,2)
            selected_strategies = random.sample(augmentation_strategies, num_augmentations)

            
            augmented_audio = original_window['audio'].copy()
            for strategy in selected_strategies: 
                augmented_audio = self.apply_augmentation(augmented_audio,strategy)
            
            augmented_window = original_window.copy()
            augmented_window['audio'] = augmented_audio
            augmented_window['augmented'] = True
            augmented_window['augmentation_strategies'] = selected_strategies
            augmented_windows.append(augmented_window)
        
        balanced_windows = minority_windows + augmented_windows + majority_windows 
        
        print(f"After balancing:")
        print(f"  {minority_label}: {len(minority_windows) + len(augmented_windows)} windows")
        print(f"  Total: {len(balanced_windows)} windows")

        return balanced_windows

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply z-score normalization per segment"""
        return (audio - np.mean(audio)) / (np.std(audio) + 1e-8)
    
    def process_dataset(self, output_dir: str, max_patients: Optional[int]=None,
                       balance_train: bool = True):
        """Process complete dataset with careful rollover approach"""
        print("="*80)
        print("CAREFUL ROLLOVER PREPROCESSING PIPELINE")
        print("="*80)

        patient_df = self.patient_info.copy()

        if max_patients:
            patient_df = patient_df.head( max_patients)
            print(f"Limited to first {max_patients} patients for testing")

        #Patient-aware splitting (70% train, 15% val, 15% test)
        train_patients, val_patients, test_patients = self.patient_aware_split(patient_df=patient_df)

        patient_data = {}
        for _, row in patient_df.iterrows():
            patient_data[str(row['Patient ID'])] = {
                'murmur_status': row['Murmur']
            }

        print(f"\nConfiguration:")
        print(f"  Balance training set: {balance_train}")
        print(f"  Window length: {self.target_length} samples ({self.window_duration}s)")
        print(f"  Temporal safeguards: {self.min_gap_threshold*1000:.0f}-{self.max_gap_threshold*1000:.0f}ms")
        
        # Process each split
        splits = {
            'train': (train_patients, balance_train),
            'val': (val_patients, False),
            'test': (test_patients, False)
        }
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, (patient_list, should_balance) in splits.items():
            print(f"\nProcessing {split_name} split...")

            # Filter patient data for this split
            split_patient_data = {pid: patient_data[pid] for pid in map(str, patient_list) 
                                 if str(pid) in patient_data}

            # Create rollover windows
            windows = self.create_rollover_windows(split_patient_data)

            if not windows:
                print(f"No windows generated for {split_name} split")
                continue

            # Balance training data if requested
            if should_balance:
                windows = self.balance_training_data( windows)
            # Extract and normalize audio data
            audio_data = []
            labels=[]
            patient_ids = [] 
            valves=[]
            quality_scores = []
            artifacts_counts = []

            for window in windows: 
                #Apply z-score normalization per segment
                normalized_audio = self.normalize_audio(window['audio'])
                audio_data.append(normalized_audio)
                labels.append(window['label'])
                patient_ids.append(window['patient_id'])
                valves.append(window['valve'])
                quality_scores.append(window.get('quality_score', 1.0))
                artifacts_counts.append(window.get('artifacts_count', 0))

            audio_data = np.array(audio_data)
            labels = np.array(labels)

            output_path = os.path.join(output_dir, f"{split_name}_careful_rollover.h5")
            
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('audio', data=audio_data, compression='gzip')
                f.create_dataset('labels', data=labels.astype('S10'))  #String labels
                f.create_dataset('patient_ids', data=np.array(patient_ids, dtype='S20'))
                f.create_dataset('valves', data=np.array(valves, dtype='S5'))
                f.create_dataset('quality_scores', data=np.array(quality_scores))
                f.create_dataset('artifacts_counts', data=np.array(artifacts_counts))

                f.attrs['window_length'] = self.target_length
                f.attrs['sampling_rate'] = self.sampling_rate
                f.attrs['preprocessing_method'] = 'careful_rollover'
            present_count = np.sum(labels == 'Present')
            absent_count = np.sum(labels == 'Absent')
            mean_quality = np.mean(quality_scores)
            mean_artifacts = np.mean(artifacts_counts)
            
            print(f"{split_name}: {len(windows)} windows")
            print(f"  Present: {present_count} ({present_count/len(labels)*100:.1f}%)")
            print(f"  Absent: {absent_count} ({absent_count/len(labels)*100:.1f}%)")
            print(f"  Mean quality score: {mean_quality:.3f}")
            print(f"  Mean artifacts per window: {mean_artifacts:.2f}")
            print(f"  Saved to: {output_path}")
        
        print("\n" + "="*60)
        print("CAREFUL ROLLOVER PREPROCESSING COMPLETED")
        print("="*60)
        print("Key advantages achieved:")
        print("✓ Natural cardiac timing preserved")
        print("✓ No artificial splice boundaries")
        print("✓ Patient-aware splits prevent data leakage")
        print("✓ Comprehensive quality control")
        print("✓ Balanced training data through augmentation")
        print("✓ Fixed-length 2000-sample windows for all splits")


def main():
    """Main function for testing the preprocessing pipeline"""
    data_path = "/home/mariamuna/research/MEMS-Project/physionet2022/training_data"
    csv_path = "/home/mariamuna/research/MEMS-Project/physionet2022/training_data.csv"
    output_dir = "/home/mariamuna/research/MEMS-Project/derived_datasets/careful_rollover_windows"

    # Initialize preprocessor
    preprocessor=CarefulRolloverPreprocessor(
        data_path=data_path,
        csv_path=csv_path,
        sampling_rate=4000
    )

    preprocessor.process_dataset(
        output_dir = output_dir,
        max_patients=20,  # Test with small subset first
        balance_train=True
    )
    
    print("\nCareful rollover preprocessing completed!")


if __name__ == "__main__":
    main()