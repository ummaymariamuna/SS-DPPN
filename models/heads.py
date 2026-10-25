"""
Neural Network Heads for Contrastive Learning and Classification

This module contains the projection and classification heads used in the
dual-path contrastive learning framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning
    
    Projects feature representations to a lower-dimensional space for contrastive learning.
    This head is used during self-supervised pre-training to learn discriminative features.
    
    Args:
        input_dim (int): Input feature dimension (default: 512)
        projection_dim (int): Output projection dimension (default: 128)
        hidden_dim (int): Hidden layer dimension (default: 256)
    """
    def __init__(self, input_dim=512, projection_dim=128, hidden_dim=256):
        super(ProjectionHead, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        self._initialize_weights() 

    def _initialize_weights( self):
        """Initialize projection head weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        """
        Forward pass through projection head
        
        Args:
            x (torch.Tensor): Input features of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Projected features of shape (batch_size, projection_dim)
        """
        return F.normalize(self.projection(x), dim=1)


class ClassifierHead(nn.Module):
    """
    Classifier head for downstream classification task
    
    This head is used for the final classification task after self-supervised pre-training.
    It can be trained on top of frozen or fine-tuned feature representations.
    
    Args:
        input_dim (int): Input feature dimension (default: 512)
        num_classes (int): Number of output classes (default: 2)
        hidden_dim (int): Hidden layer dimension (default: 256)
        dropout (float): Dropout probability (default: 0.3)
    """
    
    def __init__(self, input_dim=512, num_classes=2, hidden_dim=256, dropout=0.3):
        super(ClassifierHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Dropout( dropout),
            nn.Linear(hidden_dim,hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2,num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize classifier head weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through classifier head
        
        Args:
            x (torch.Tensor): Input features of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Class logits of shape (batch_size, num_classes)
        """
        return self.classifier(x)


class PrototypicalHead(nn.Module):
    """
    Prototypical network head for few-shot learning
    
    Implements prototypical networks for few-shot classification by computing
    distances to class prototypes in the feature space.
    
    Args:
        input_dim (int): Input feature dimension (default: 512)
        num_classes (int): Number of classes (default: 2)
        distance_metric (str): Distance metric ('euclidean' or 'cosine')
    """
    
    def __init__(self, input_dim=512, num_classes=2, distance_metric='euclidean'):
        super(PrototypicalHead, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.distance_metric = distance_metric


        self.prototypes = nn.Parameter(torch.randn(num_classes, input_dim))
        
        # Optional projection for prototypes
        self.prototype_projection = nn.Linear(input_dim, input_dim)
        
        self._initialize_weights() 
    
    def _initialize_weights(self):
        """Initialize prototypical head weights"""
        nn.init.xavier_uniform_(self.prototypes)
        nn.init.xavier_uniform_(self.prototype_projection.weight)
        nn.init.constant_(self.prototype_projection.bias, 0)

    
    def compute_distances(self,features,prototypes):
        """
        Compute distances between features and prototypes
        
        Args:
            features (torch.Tensor): Feature vectors (batch_size, input_dim)
            prototypes (torch.Tensor): Prototype vectors (num_classes, input_dim)
        
        Returns:
            torch.Tensor: Distance matrix (batch_size, num_classes)
        """
        if self.distance_metric == 'euclidean':
            distances = torch.cdist(features, prototypes, p=2)
        elif self.distance_metric == 'cosine':
            # Cosine distance (1 - cosine similarity)
            features_norm = F.normalize( features, dim=1)
            prototypes_norm = F.normalize( prototypes, dim=1)
            similarities = torch.mm(features_norm, prototypes_norm.t())
            distances = 1 - similarities
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")


        return distances
    
    def forward(self, x, support_features = None, support_labels = None):
        """
        Forward pass through prototypical head
        
        Args:
            x (torch.Tensor): Query features (batch_size, input_dim)
            support_features (torch.Tensor, optional): Support set features
            support_labels (torch.Tensor, optional): Support set labels
        
        Returns:
            torch.Tensor: Negative log probabilities (batch_size, num_classes)
        """
        x = self.prototype_projection(x)

        if support_features is not None and support_labels is not None:
            # Use support set to compute prototypes (few-shot learning)
            #project support too so it lives in the same space as the queries 
            support_features=self.prototype_projection(support_features) 
            prototypes = self._compute_prototypes( support_features, support_labels)
        else:
            #Use learnable prototypes
            prototypes = self.prototypes

        # Compute distances to prototypes
        distances=self.compute_distances(x, prototypes)
        
        log_probs = -distances
        
        return F.log_softmax(log_probs,dim=1)
    
    def _compute_prototypes(self, support_features, support_labels):
        """
        Compute class prototypes from support set
        
        Args:
            support_features (torch.Tensor): Support features (n_support, input_dim)
            support_labels (torch.Tensor): Support labels (n_support,)
        
        Returns:
            torch.Tensor: Class prototypes (num_classes, input_dim)
        """
        unique_labels=torch.unique(support_labels)
        prototypes = torch.zeros(len(unique_labels), self.input_dim,  
                               device=support_features.device)
        
        for i, label in enumerate(unique_labels):
            mask = (support_labels == label)
            class_features=support_features[mask]
            prototypes[i] = class_features.mean(dim = 0)

        return prototypes


class FusionHead(nn.Module):
    """
    Fusion head for combining multi-modal features
    
    Combines features from different modalities (e.g., audio and spectrogram)
    using various fusion strategies.
    
    Args:
        input_dim (int): Input feature dimension for each modality
        output_dim (int): Output feature dimension
        fusion_type (str): Fusion strategy ('concat', 'add', 'attention')
        num_modalities (int): Number of input modalities (default: 2)
    """
    def __init__(self, input_dim=512, output_dim=512, fusion_type='concat', num_modalities=2):
        super( FusionHead, self).__init__()

        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        self.num_modalities = num_modalities
        if fusion_type == 'concat':
            self.fusion_layer = nn.Linear(input_dim * num_modalities, output_dim) 
        elif fusion_type == 'add':
            self.fusion_layer = nn.Linear(input_dim, output_dim)
        elif fusion_type == 'attention':
            self.attention_weights = nn.Linear(input_dim,1)
            self.fusion_layer = nn.Linear(input_dim, output_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize fusion head weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward( self, *features):
        """
        Forward pass through fusion head
        
        Args:
            *features: Variable number of feature tensors, each of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Fused features of shape (batch_size, output_dim)
        """
        if len(features) != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} feature tensors, got {len(features)}")

        
        if self.fusion_type == 'concat':
            fused = torch.cat(features,dim = 1)
            output = self.fusion_layer(fused)
            
        elif self.fusion_type == 'add':
            fused = torch.stack(features, dim = 0).sum(dim=0)
            output = self.fusion_layer(fused)
            
        elif self.fusion_type == 'attention':
            feature_stack = torch.stack(features,dim=1)

            
            attention_scores = self.attention_weights(feature_stack)
            attention_weights = F.softmax(attention_scores, dim=1)
            
            # Weighted combination
            fused = (feature_stack * attention_weights).sum(dim = 1)
            output = self.fusion_layer(fused)

        
        return output
