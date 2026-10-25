"""
Training Module for Dual-Path Contrastive Learning

This module contains training logic for self-supervised pre-training,
supervised fine-tuning, and prototypical network training.
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
try: 
    from ..models.dual_path_model import DualPathContrastiveLearning
    from ..models.heads import PrototypicalHead
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from models.dual_path_model import DualPathContrastiveLearning
    from models.heads import PrototypicalHead



class DualPathTrainer:
    """
    Trainer class for dual-path contrastive learning model
    
    This class handles all aspects of training including:
    - Self-supervised pre-training with contrastive loss
    - Supervised fine-tuning with optional SSL regularization
    - Prototypical network training for few-shot learning
    - Learning rate scheduling and optimization
    """
    
    def __init__(self,model: DualPathContrastiveLearning):
        self.model=model
        self.device=model.device

    def train_self_supervised(self,data, epochs=50, batch_size=64, learning_rate=0.001,
                            weight_audio = 0.5, weight_spec=0.5, weight_cross=1.0,
                            loss_type='contrastive', hybrid_alpha=0.5):
        """
        Self-supervised pre-training with dual-path contrastive learning
        
        Args:
            data: Training data (numpy array or tensor)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            weight_audio: Weight for audio-audio contrastive loss
            weight_spec: Weight for spectrogram-spectrogram contrastive loss
            weight_cross: Weight for cross-modal contrastive loss
            loss_type: Type of loss ('contrastive', 'wasserstein', 'hybrid')
            hybrid_alpha: Alpha parameter for hybrid loss
        
        Returns:
            dict: Training history
        """
        print(f"Starting self-supervised training...")
        print(f"Data shape: {data.shape}")
        print(f"Loss type: {loss_type}")

        if isinstance(data, np.ndarray):
            data = torch.FloatTensor(data) 
        
        dataset=TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True, 
                              num_workers = 2, pin_memory = True)

        
        self.model.audio_encoder.train()
        self.model.spec_encoder.train() 
        self.model.audio_projector.train( )
        self.model.spec_projector.train()

        parameters = (list(self.model.audio_encoder.parameters()) + 
                     list(self.model.spec_encoder.parameters()) +
                     list(self.model.audio_projector.parameters()) +
                     list(self.model.spec_projector.parameters()))
        
        optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
        history = {
            'loss': [],
            'loss_audio': [],
            'loss_spec': [],
            'loss_cross': []
        }
        
        for epoch in range(epochs):
            epoch_losses = {'total': 0, 'audio': 0, 'spec': 0, 'cross': 0}
            num_batches = 0 
            
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch_idx,(batch_data,) in enumerate(progress_bar):
                batch_data = batch_data.to(self.device)

                
                #Create augmented versions
                aug1 = self.model._data_augmentation(batch_data.clone(), strength=0.15)
                aug2 = self.model._data_augmentation(batch_data.clone(),strength = 0.15)

                
                audio_features_1 = self.model.audio_encoder(aug1)
                audio_features_2 = self.model.audio_encoder(aug2)
                audio_proj_1 = self.model.audio_projector(audio_features_1)
                audio_proj_2=self.model.audio_projector(audio_features_2)

                spec_1=self.model._waveform_to_spectrogram(aug1)
                spec_2 = self.model._waveform_to_spectrogram(aug2)
                spec_1 = self.model._spec_augmentation(spec_1,strength=0.1)
                spec_2 = self.model._spec_augmentation(spec_2,strength=0.1)
                spec_features_1=self.model.spec_encoder(spec_1)
                spec_features_2 = self.model.spec_encoder( spec_2)
                spec_proj_1 = self.model.spec_projector(spec_features_1)
                spec_proj_2 = self.model.spec_projector(spec_features_2)
                # Compute losses
                if loss_type == 'contrastive':
                    loss_audio = self.model._contrastive_loss(audio_proj_1,audio_proj_2)
                    loss_spec = self.model._contrastive_loss(spec_proj_1,spec_proj_2)
                    loss_cross = (self.model._contrastive_loss(audio_proj_1, spec_proj_1) +
                                self.model._contrastive_loss(audio_proj_2, spec_proj_2)) / 2
                elif loss_type == 'wasserstein':
                    loss_audio = self.model._wasserstein_loss(audio_proj_1, audio_proj_2) 
                    loss_spec = self.model._wasserstein_loss(spec_proj_1, spec_proj_2)
                    loss_cross = (self.model._wasserstein_loss(audio_proj_1, spec_proj_1) +
                                self.model._wasserstein_loss(audio_proj_2, spec_proj_2)) / 2
                elif loss_type == 'hybrid':
                    # Hybrid: contrastive + wasserstein
                    loss_audio = (hybrid_alpha * self.model._contrastive_loss(audio_proj_1, audio_proj_2) +
                                (1 - hybrid_alpha) * self.model._wasserstein_loss(audio_proj_1, audio_proj_2))
                    loss_spec = (hybrid_alpha * self.model._contrastive_loss(spec_proj_1, spec_proj_2) +
                               (1 - hybrid_alpha) * self.model._wasserstein_loss(spec_proj_1, spec_proj_2))
                    loss_cross = hybrid_alpha * ((self.model._contrastive_loss(audio_proj_1,spec_proj_1) +
                                               self.model._contrastive_loss(audio_proj_2, spec_proj_2)) / 2)

                #Combined loss 
                total_loss = (weight_audio * loss_audio +  
                            weight_spec * loss_spec + 
                            weight_cross * loss_cross)
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
                optimizer.step()
                epoch_losses['total'] += total_loss.item()
                epoch_losses['audio'] += loss_audio.item()
                epoch_losses['spec'] += loss_spec.item()
                epoch_losses['cross'] += loss_cross.item()
                num_batches += 1

                progress_bar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Audio': f'{loss_audio.item():.4f}',
                    'Spec': f'{loss_spec.item():.4f}',
                    'Cross': f'{loss_cross.item():.4f}'
                })
            
            scheduler.step()
            
            #Record epoch statistics
            history['loss'].append(epoch_losses['total'] / num_batches)
            history['loss_audio'].append(epoch_losses['audio'] / num_batches)
            history['loss_spec'].append(epoch_losses['spec'] / num_batches)
            history['loss_cross'].append(epoch_losses['cross'] / num_batches)
            print(f"Epoch {epoch+1}: Loss = {history['loss'][-1]:.4f}")
        self.model.training_history['ssl'] = history
        return history

    def fine_tune(self, x_train, y_train, x_val=None, y_val=None, epochs=50, batch_size=32,  
                learning_rate=0.0001, unfreeze_after=10, ssl_weight=0.0):
        """
        Fine-tune the model with supervised learning
        
        Args:
            x_train: Training data
            y_train: Training labels
            x_val: Validation data (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            unfreeze_after: Epoch to unfreeze encoders
            ssl_weight: Weight for SSL regularization
        
        Returns:
            dict: Training history
        """
        print(f"Starting supervised fine-tuning...")
        if self.model.classifier is None:
            self.model.create_classifier(num_classes=len(np.unique(y_train)))
        
        #Prepare data
        if isinstance(x_train, np.ndarray):
            x_train = torch.FloatTensor(x_train)
            y_train = torch.LongTensor(y_train)

        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,
                                num_workers = 2,pin_memory=True)
        
        val_loader = None
        if x_val is not None and y_val is not None:
            if isinstance(x_val, np.ndarray):
                x_val = torch.FloatTensor(x_val)
                y_val = torch.LongTensor(y_val)
            val_dataset = TensorDataset(x_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=2, pin_memory=True)
        
        # Set initial training mode (freeze encoders initially)
        self.model.audio_encoder.eval()
        self.model.spec_encoder.eval( )
        for param in self.model.audio_encoder.parameters():
            param.requires_grad = False
        for param in self.model.spec_encoder.parameters():
            param.requires_grad = False

        self.model.fusion_head.train()
        self.model.classifier.train()
        parameters = (list(self.model.fusion_head.parameters()) +
                     list(self.model.classifier.parameters()))

        optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'ssl_loss': []
        }
        
        for epoch in range(epochs):
            #Unfreeze encoders after specified epoch
            if epoch == unfreeze_after:
                print(f"Unfreezing encoders at epoch {epoch+1}")
                self.model.audio_encoder.train()
                self.model.spec_encoder.train() 
                for param in self.model.audio_encoder.parameters():
                    param.requires_grad = True
                for param in self.model.spec_encoder.parameters():
                    param.requires_grad=True
                
                #Update optimizer to include encoder parameters
                parameters = (list(self.model.audio_encoder.parameters()) +
                            list(self.model.spec_encoder.parameters()) +
                            list(self.model.fusion_head.parameters()) +
                            list(self.model.classifier.parameters()))
                optimizer=optim.Adam(parameters, lr=learning_rate/10, weight_decay=1e-4)

            train_loss, train_acc, ssl_loss=self._train_epoch(
                train_loader,optimizer,criterion,ssl_weight
            )
            
            val_loss, val_acc = 0.0, 0.0
            if val_loader is not None:
                val_loss, val_acc = self._validate_epoch(val_loader, criterion)
            
            #Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['ssl_loss'].append(ssl_loss)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        self.model.training_history['fine_tune'] = history
        return history
    
    def _train_epoch(self, dataloader, optimizer, criterion, ssl_weight):
        """Train for one epoch"""
        total_loss = 0
        correct = 0
        total = 0
        ssl_loss_total = 0 

        for batch_data, batch_labels in tqdm(dataloader, desc="Training"):
            batch_data = batch_data.to(self.device)
            batch_labels=batch_labels.to(self.device)
            
            optimizer.zero_grad()
            audio_features, spec_features, fused_features = self.model.forward_features(batch_data)
            logits = self.model.classifier(fused_features) 

            # Supervised loss
            supervised_loss = criterion(logits,batch_labels)
            total_loss_batch=supervised_loss

            # Optional SSL regularization
            ssl_loss_batch = 0
            if ssl_weight > 0:
                aug_data=self.model._data_augmentation(batch_data.clone())
                with torch.no_grad():
                    _, _, aug_features = self.model.forward_features(aug_data)
                ssl_loss_batch = F.mse_loss(fused_features, aug_features)
                total_loss_batch = total_loss_batch + ssl_weight * ssl_loss_batch

            # Backward pass
            total_loss_batch.backward()
            # clip everything the optimizer is updating, not just the classifier
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group['params'] if p.requires_grad],
                max_norm=1.0
            )
            optimizer.step()

            #Statistics
            total_loss += total_loss_batch.item() 
            ssl_loss_total += ssl_loss_batch if isinstance(ssl_loss_batch, (int, float)) else ssl_loss_batch.item()
            _, predicted = torch.max(logits.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item() 

        return total_loss / len(dataloader), 100 * correct / total, ssl_loss_total / len(dataloader)

    def _validate_epoch(self,dataloader, criterion): 
        """Validate for one epoch"""
        modules = [self.model.audio_encoder, self.model.spec_encoder,
                   self.model.fusion_head,self.model.classifier]
        prev_modes = [m.training for m in modules]
        for m in modules:
            m.eval()
        
        total_loss=0
        correct = 0
        total = 0

        
        with torch.no_grad(): 
            for batch_data, batch_labels in dataloader: 
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                _, _, fused_features = self.model.get_features(batch_data)
                logits=self.model.classifier(fused_features)


                
                # Loss and accuracy
                loss = criterion(logits, batch_labels)
                total_loss += loss.item()

                _,predicted = torch.max(logits.data,1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item() 

        for m, was_training in zip(modules, prev_modes):
            m.train(was_training)


        return total_loss / len(dataloader), 100 * correct / total

    def train_prototypical_classifier(self, x_train, y_train, x_val = None, y_val=None, 
                                    epochs=30, batch_size=32, learning_rate=0.0001):
        """
        Train prototypical network for few-shot learning
        
        Args:
            x_train: Training data
            y_train: Training labels
            x_val: Validation data (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        
        Returns:
            dict: Training history
        """
        print(f"Training prototypical classifier...")
        
        # Create prototypical head
        num_classes = len(np.unique(y_train))
        self.model.prototypical_head = PrototypicalHead(
            input_dim = 512,
            num_classes = num_classes,
            distance_metric='euclidean'
        ).to(self.device)

        if isinstance(x_train, np.ndarray):
            x_train = torch.FloatTensor(x_train)
            y_train = torch.LongTensor(y_train)

        train_dataset = TensorDataset(x_train, y_train)
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=2, pin_memory=True)
        
        val_loader = None
        if x_val is not None and y_val is not None:
            if isinstance(x_val, np.ndarray):
                x_val = torch.FloatTensor(x_val)
                y_val = torch.LongTensor(y_val)
            val_dataset = TensorDataset(x_val, y_val)
            val_loader = DataLoader( val_dataset, batch_size=batch_size, shuffle=False)

        self.model.audio_encoder.eval()
        self.model.spec_encoder.eval()
        self.model.fusion_head.eval()
        
        for param in self.model.audio_encoder.parameters():
            param.requires_grad = False
        for param in self.model.spec_encoder.parameters():
            param.requires_grad=False 
        for param in self.model.fusion_head.parameters():
            param.requires_grad = False 

        #Optimizer for prototypical head only
        optimizer=optim.Adam(self.model.prototypical_head.parameters(), 
                             lr=learning_rate, weight_decay=1e-4)

        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            total_loss=0
            self.model.prototypical_head.train()

            for batch_data, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                with torch.no_grad():
                    _,_, features = self.model.get_features(batch_data)


                log_probs = self.model.prototypical_head(features)
                loss = F.nll_loss(log_probs, batch_labels)
                loss.backward()
                optimizer.step()


                total_loss += loss.item()
            history['train_loss'].append(total_loss / len(train_loader))
            
            if val_loader is not None:
                val_loss = self._validate_prototypical( val_loader)
                history['val_loss'].append(val_loss)
                print(f"Epoch {epoch+1}: Train Loss: {history['train_loss'][-1]:.4f}, "
                      f"Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}: Train Loss: {history['train_loss'][-1]:.4f}")
        
        self.model.training_history['prototypical'] = history
        return history

    def _validate_prototypical(self, dataloader):
        """Validate prototypical network"""
        self.model.prototypical_head.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_data,batch_labels in dataloader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                _, _, features = self.model.get_features(batch_data)
                
                # Prototypical network forward pass 
                log_probs = self.model.prototypical_head(features)
                loss = F.nll_loss(log_probs, batch_labels)
                
                total_loss += loss.item()

        return total_loss / len(dataloader)


def add_training_methods_to_model():
    """Add training methods to DualPathContrastiveLearning class"""
    
    def train_self_supervised( self, data, **kwargs): 
        trainer = DualPathTrainer(self)
        return trainer.train_self_supervised(data,**kwargs)
    
    def fine_tune(self,x_train,y_train,**kwargs):
        trainer = DualPathTrainer( self)
        return trainer.fine_tune(x_train, y_train, **kwargs)

    def train_prototypical_classifier(self, x_train, y_train, **kwargs):
        trainer = DualPathTrainer( self)
        return trainer.train_prototypical_classifier(x_train, y_train, **kwargs)

    
    # Add methods to the class
    DualPathContrastiveLearning.train_self_supervised = train_self_supervised
    DualPathContrastiveLearning.fine_tune = fine_tune
    DualPathContrastiveLearning.train_prototypical_classifier = train_prototypical_classifier

#Initialize training methods
add_training_methods_to_model()


HeartSoundTrainer = DualPathTrainer
