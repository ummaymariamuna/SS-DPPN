"""
Dual-Path Contrastive Learning Model for Heart Sound Classification

This module implements the main dual-path contrastive learning framework that
combines TCN-based audio processing with ResNet-based spectrogram analysis.
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


from .tcn_encoder import DilatedTemporalConvEncoder
from .resnet_encoder import ResNet50SpectrogramEncoder
from .heads import ProjectionHead, ClassifierHead, PrototypicalHead, FusionHead


# Check if geomloss is available for efficient Wasserstein distance
try:
    import geomloss
    HAS_GEOMLOSS = True
except ImportError:
    HAS_GEOMLOSS = False 


class DualPathContrastiveLearning(nn.Module):
    """
    Dual-Path Contrastive Learning Framework for Heart Sound Classification
    
    This class implements a sophisticated multi-modal learning approach that:
    1. Processes raw audio waveforms using a Temporal Convolutional Network (TCN)
    2. Converts audio to spectrograms and processes them using ResNet-50
    3. Learns representations through multi-modal contrastive learning
    4. Supports various training strategies (SSL, supervised, few-shot)
    5. Includes comprehensive evaluation and calibration analysis
    
    Key Features:
    - Dual-path processing (time-domain + frequency-domain)
    - Advanced augmentation strategies
    - Multiple contrastive loss functions
    - Prototypical networks for few-shot learning
    - Statistical evaluation with confidence intervals
    
    Args:
        input_shape (tuple): Shape of input audio (samples, channels)
        projection_dim (int): Dimension of contrastive projections
        temperature (float): Temperature parameter for contrastive loss
        sample_rate (int): Audio sampling rate
        n_mels (int): Number of mel-frequency bins
        spec_time_dim (int): Time dimension for spectrograms
    """
    
    def __init__(self, input_shape=(4000, 1), projection_dim=128, temperature=0.1, 
                 sample_rate=4000, n_mels=64, spec_time_dim=64, n_fft=256, hop_length=32):
        
        super(DualPathContrastiveLearning, self).__init__()

        self.input_shape = input_shape
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.spec_time_dim = spec_time_dim
        self.n_fft=n_fft
        self.hop_length=hop_length

        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.audio_encoder=DilatedTemporalConvEncoder(input_shape).to(self.device)
        self.spec_encoder = ResNet50SpectrogramEncoder(
            input_shape=(n_mels, spec_time_dim), 
            feature_dim=512
        ).to(self.device)

        # Projection heads for contrastive learning
        self.audio_projector = ProjectionHead(512, projection_dim).to(self.device)
        self.spec_projector = ProjectionHead(512,projection_dim).to(self.device)

        self.fusion_head = FusionHead(
            input_dim=512, 
            output_dim=512, 
            fusion_type='attention',
            num_modalities=2
        ).to(self.device)

        # hop 32 -> ~62 frames for 2000 samples, ~125 for 4000 (hop 512 only gave 4-8 frames)
        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate, 
            n_mels = n_mels,
            n_fft=n_fft,
            hop_length=hop_length, 
            win_length=n_fft,
            power=2.0
        ).to(self.device)

        self.amplitude_to_db = AmplitudeToDB().to(self.device)

        self.training_history={}
        
        self.classifier = None
        self.prototypical_head = None

        print(f"Dual-Path Model initialized on device: {self.device}")
        print(f"TCN Encoder receptive field: {self.audio_encoder.get_receptive_field()} samples")
    
    def _waveform_to_spectrogram(self, waveform):
        """
        Convert waveform to mel-spectrogram
        
        Args:
            waveform (torch.Tensor): Input waveform (batch_size, samples)
        
        Returns:
            torch.Tensor: Mel-spectrogram (batch_size, n_mels, time_frames)
        """
        # Ensure waveform is 2D
        if waveform.dim() == 1: 
            waveform=waveform.unsqueeze(0)
        
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        if mel_spec_db.size( -1) != self.spec_time_dim: 
            mel_spec_db = F.interpolate(
                mel_spec_db.unsqueeze(1), 
                size=(self.n_mels, self.spec_time_dim), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)

        return mel_spec_db
    
    def _spec_augmentation(self, spec, strength = 0.1): 
        """Apply spectrogram-specific augmentations"""
        if torch.rand(1) < 0.5:
            time_mask_param = int(spec.size(-1) * strength)
            if time_mask_param > 0:
                t = torch.randint(0, time_mask_param, (1,)).item() 
                t0 = torch.randint(0, spec.size(-1) - t, (1,)).item()
                spec[..., t0:t0+t] = 0
        
        if torch.rand(1) < 0.5:
            freq_mask_param = int(spec.size(-2) * strength)
            if freq_mask_param > 0:
                f = torch.randint(0,freq_mask_param,(1,)).item()
                f0 = torch.randint(0, spec.size(-2) - f, (1,)).item()
                spec[...,f0:f0+f,:] = 0

        return spec

    def _data_augmentation(self, x, strength=0.1):
        """Apply waveform augmentations"""
        augmentations = [
            self._add_gaussian_noise,
            self._time_shift,
            self._amplitude_scale,
        ]
        if torch.rand( 1) < 0.7:  #70% chance of augmentation
            aug_func = np.random.choice(augmentations)
            x = aug_func(x, strength)
        
        return x
    
    def _add_gaussian_noise(self, x, strength=0.1):
        """Add Gaussian noise to waveform"""
        noise_factor = strength * 0.1
        noise = torch.randn_like(x) * noise_factor
        return x + noise
    
    def _time_shift(self, x, strength = 0.1):
        """Apply time shifting"""
        shift_samples = int(x.size(-1) * strength * 0.1)
        if shift_samples > 0:
            shift = torch.randint(-shift_samples,shift_samples + 1, (1,))
            x = torch.roll(x, shift.item(), dims=-1)
        return x
    
    def _amplitude_scale(self,x,strength=0.1):
        """Apply amplitude scaling"""
        scale_factor = 1 + (torch.rand(1).item() - 0.5) * strength * 0.4
        return x * scale_factor

    def _contrastive_loss(self, z1, z2):
        """Compute InfoNCE contrastive loss"""
        batch_size=z1.size(0)

        
        z1 = F.normalize(z1,dim=1)
        z2 = F.normalize(z2,dim=1)
        similarity_matrix=torch.matmul(z1, z2.t()) / self.temperature

        labels = torch.arange(batch_size).to(self.device)

        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels) 
        return loss

    def _wasserstein_loss(self, projections_1, projections_2):
        """Compute Wasserstein distance loss"""
        if HAS_GEOMLOSS:
            # Use efficient geomloss implementation
            wasserstein = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05)
            return wasserstein(projections_1, projections_2)
        else:
            #Fallback to simple implementation
            return self._simple_wasserstein_loss(projections_1, projections_2)

    def _simple_wasserstein_loss( self, x, y):
        """Simple Wasserstein-1 approximation"""
        # Sort along each dimension and compute L1 distance
        x_sorted = torch.sort(x, dim=0)[0]
        y_sorted = torch.sort(y, dim=0)[0]
        return torch.mean( torch.abs(x_sorted - y_sorted))

    def create_classifier(self, num_classes=2):
        """Create classifier head for downstream tasks"""
        self.classifier = ClassifierHead(
            input_dim = 512,
            num_classes=num_classes
        ).to(self.device)

        return self.classifier
    
    def save_model(self, path):
        """Save model state"""
        state_dict = {
            'audio_encoder': self.audio_encoder.state_dict(),
            'spec_encoder': self.spec_encoder.state_dict(),
            'audio_projector': self.audio_projector.state_dict(),
            'spec_projector': self.spec_projector.state_dict(),
            'fusion_head': self.fusion_head.state_dict(),
            'config': {
                'input_shape': self.input_shape,
                'projection_dim': self.projection_dim,
                'temperature': self.temperature,
                'sample_rate': self.sample_rate,
                'n_mels': self.n_mels,
                'spec_time_dim': self.spec_time_dim,
                'n_fft': self.n_fft,
                'hop_length': self.hop_length
            }
        }
        
        if self.classifier is not None:
            state_dict['classifier'] = self.classifier.state_dict()
        if self.prototypical_head is not None:
            state_dict['prototypical_head'] = self.prototypical_head.state_dict()

        torch.save( state_dict, path)
        print(f"Model saved to {path}")
    def load_model(self,path):
        """Load model state"""
        state_dict = torch.load(path, map_location=self.device)
        
        self.audio_encoder.load_state_dict(state_dict['audio_encoder'])
        self.spec_encoder.load_state_dict(state_dict['spec_encoder'])
        self.audio_projector.load_state_dict(state_dict['audio_projector'])
        self.spec_projector.load_state_dict(state_dict['spec_projector'])
        self.fusion_head.load_state_dict(state_dict['fusion_head'])
        
        if 'classifier' in state_dict:
            if self.classifier is None:
                num_classes = state_dict['classifier']['classifier.6.weight'].shape[0]
                self.create_classifier(num_classes=num_classes)
            self.classifier.load_state_dict(state_dict['classifier'])

        if 'prototypical_head' in state_dict:
            if self.prototypical_head is None:
                num_classes, input_dim = state_dict['prototypical_head']['prototypes'].shape
                self.prototypical_head=PrototypicalHead(input_dim=input_dim, num_classes=num_classes).to(self.device)
            self.prototypical_head.load_state_dict(state_dict['prototypical_head'])

        print(f"Model loaded from {path}")
    
    def forward_features(self, x):
        """
        Same as get_features but keeps gradients and doesn't touch train/eval mode,
        use this one inside training loops
        """
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)

        if x.dim() == 1:
            x=x.unsqueeze(0)
        audio_features = self.audio_encoder(x) 
        spec=self._waveform_to_spectrogram(x)
        spec_features=self.spec_encoder(spec)


        # Fusion
        fused_features = self.fusion_head(audio_features,spec_features)

        return audio_features, spec_features, fused_features

    def get_features( self, x):
        """
        Extract features from both audio and spectrogram paths (inference only)

        Args:
            x (torch.Tensor): Input audio waveform

        Returns:
            tuple: (audio_features, spec_features, fused_features)
        """
        self.audio_encoder.eval()
        self.spec_encoder.eval()
        self.fusion_head.eval()

        with torch.no_grad( ):
            return self.forward_features(x)

    def evaluate_probabilities(self,x_test):
        """
        Generate probability predictions for test data
        
        Args:
            x_test (np.ndarray or torch.Tensor): Test data
        
        Returns:
            np.ndarray: Predicted probabilities for positive class
        """
        if self.classifier is None:
            raise ValueError("Classifier not initialized. Please create classifier first.")

        self.audio_encoder.eval()
        self.spec_encoder.eval()
        self.fusion_head.eval()
        self.classifier.eval()
        
        probabilities = []
        
        batch_size = 32
        with torch.no_grad( ):
            for i in range(0, len(x_test), batch_size):
                batch = x_test[i:i+batch_size]
                
                if isinstance(batch, np.ndarray):
                    batch = torch.FloatTensor(batch).to(self.device)
                # Get fused features 
                _,_,fused_features = self.get_features(batch)

                logits = self.classifier(fused_features)
                probs=torch.softmax(logits, dim=1)

                #Extract probability for positive class (class 1)
                batch_probs = probs[:, 1].cpu().numpy()
                probabilities.extend(batch_probs)

        return np.array(probabilities)
    
    def evaluate( self, x_test, y_test):
        """
        Evaluate model performance on test data
        
        Args:
            x_test (np.ndarray): Test data
            y_test (np.ndarray): Test labels
        
        Returns:
            dict: Evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        y_prob = self.evaluate_probabilities(x_test)
        y_pred=(y_prob > 0.5).astype(int)
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='binary', zero_division=0),
        }
        
        try:
            metrics['auroc'] = roc_auc_score(y_test, y_prob)
        except ValueError:  # only one class in y_test
            metrics['auroc'] = 0.5

        
        return metrics


DualPathHeartSoundModel = DualPathContrastiveLearning


