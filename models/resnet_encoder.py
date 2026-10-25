"""
ResNet-50 based Spectrogram Encoder for Heart Sound Classification

This module implements a ResNet-50 based encoder specifically designed for processing
mel-spectrogram representations of heart sounds.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ResNet50SpectrogramEncoder(nn.Module):
    """
    ResNet-50 based encoder for spectrogram feature extraction
    
    This encoder uses a pre-trained ResNet-50 architecture adapted for spectrogram analysis:
    - Replaces the first 7x7 conv with 3x3 conv for smaller spectrograms
    - Uses pre-trained ImageNet weights (optional)
    - Outputs 512-dimensional feature vectors
    - Handles both single-channel and RGB spectrograms
    
    Args:
        input_shape (tuple): Shape of input spectrogram (height, width)
        pretrained (bool): Whether to use ImageNet pre-trained weights
        feature_dim (int): Dimension of output features (default: 512)
    """
    
    def __init__(self,input_shape = (64,64), pretrained=True, feature_dim=512):
        super(ResNet50SpectrogramEncoder, self).__init__()
        
        self.input_height = input_shape[0]
        self.input_width = input_shape[1]
        self.feature_dim = feature_dim
        
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # Modify first convolution for spectrogram input
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Removing the original classifier
        self.backbone.fc = nn.Identity()

        # Getting the feature dimension from ResNet-50 (2048)
        resnet_feature_dim = 2048
        
        # Adding custom projection head
        self.projection=nn.Sequential(
            nn.Linear(resnet_feature_dim, 1024),
            nn.ReLU(inplace = True),
            nn.Dropout(0.2),
            nn.Linear(1024, feature_dim)
        )
        self._initialize_custom_layers() 
    
    def _initialize_custom_layers(self): 
        """Initialize custom layers (conv1 and projection)"""
        nn.init.xavier_uniform_(self.backbone.conv1.weight)

        # Initializing projection layers
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the ResNet encoder
        
        Args:
            x (torch.Tensor): Input spectrogram tensor of shape 
                            (batch_size, height, width) or (batch_size, 1, height, width)
        
        Returns:
            torch.Tensor: Encoded features of shape (batch_size, feature_dim)
        """
        #Handling input shape: ensure (batch_size, 1, height, width)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        if x.size( 2) != self.input_height or x.size(3) != self.input_width: 
            x = F.interpolate(x,size = (self.input_height,self.input_width), 
                            mode='bilinear', align_corners=False)

        #Passing through ResNet backbone
        features=self.backbone(x)  #(batch_size, 2048)
        
        # Applying projection
        output = self.projection(features)

        return output

    def freeze_backbone(self):
        """Freeze ResNet backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False

        # We will keep conv1 trainable since we modified it
        for param in self.backbone.conv1.parameters():
            param.requires_grad=True
    def unfreeze_backbone(self):
        """Unfreeze ResNet backbone parameters"""
        for param in self.backbone.parameters(): 
            param.requires_grad = True

    
    def get_feature_maps(self, x, layer_name='layer4'):
        """
        Extract intermediate feature maps for visualization
        
        Args:
            x (torch.Tensor): Input spectrogram
            layer_name (str): Name of the layer to extract features from
        
        Returns:
            torch.Tensor: Feature maps from the specified layer
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        feature_maps={}
        
        def hook_fn(module, input, output): 
            feature_maps[layer_name] = output

        # Having the target layer
        target_layer = getattr( self.backbone, layer_name)
        handle=target_layer.register_forward_hook(hook_fn) 

        
        _=self.backbone(x)
        
        handle.remove()

        return feature_maps[layer_name]

class SpectrogramEncoder(ResNet50SpectrogramEncoder):
    """Alias for backward compatibility"""
    pass


class LightweightSpectrogramEncoder(nn.Module):
    """
    Lightweight CNN encoder for spectrograms (alternative to ResNet-50)
    
    This is a smaller, faster alternative for when computational resources are limited.
    """


    
    def __init__(self, input_shape=(64, 64), feature_dim=512):
        super(LightweightSpectrogramEncoder,self).__init__()
        
        self.input_height = input_shape[0]
        self.input_width = input_shape[1]
        self.feature_dim = feature_dim

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size = 3, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128,kernel_size = 3,padding = 1), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256,128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, feature_dim)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the lightweight encoder"""
        if x.dim() == 3: 
            x = x.unsqueeze(1)

        
        if x.size(2) != self.input_height or x.size(3) != self.input_width:
            x = F.interpolate(x,size=(self.input_height, self.input_width), 
                            mode='bilinear', align_corners=False)
        
        x = self.features(x)
        x=x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
