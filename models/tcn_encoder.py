"""
Temporal Convolutional Network (TCN) Encoder for 1D Audio Signal Processing

This module implements a dilated temporal convolutional network specifically designed
for processing raw audio waveforms in heart sound classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualDilatedBlock(nn.Module):
    """
    Residual Dilated Block for TCN with pre-activation normalization
    
    Implements the residual dilated convolution block with:
    - Pre-activation normalization (BatchNorm → ReLU → Conv)
    - Dilated convolution for expanded receptive field
    - Identity skip connections for gradient flow
    
    Args:
        channels (int): Number of input/output channels
        dilation_rate (int): Dilation rate for the convolution
        dropout (float): Dropout probability
    """

    
    def __init__(self, channels, dilation_rate, dropout = 0.1):
        super(ResidualDilatedBlock, self).__init__()

        
        self.norm1 = nn.BatchNorm1d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv1d(
            channels, channels, 
            kernel_size=3, 
            dilation=dilation_rate, 
            padding=dilation_rate,
            bias=False
        )

        # Second normalization and activation
        self.norm2 = nn.BatchNorm1d(channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv2=nn.Conv1d(
            channels, channels, 
            kernel_size=3, 
            dilation=dilation_rate, 
            padding=dilation_rate, 
            bias = False
        ) 
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        """Forward pass with residual connection"""
        identity = x
        # Pre-activation + first conv
        out = self.norm1(x) 
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out=self.relu2(out)
        out=self.dropout(out) 
        out=self.conv2(out)
        # Residual connection
        out = out + identity
        return out

class DilatedTemporalConvEncoder(nn.Module):
    """
    Dilated Temporal Convolutional Network (TCN) for 1D Signal Encoding
    
    This encoder implements a sophisticated temporal modeling architecture with:
    - Initial downsampling for computational efficiency
    - Eight sequential residual dilated blocks with exponentially increasing dilation
    - Pre-activation normalization throughout
    - Global temporal pooling for fixed-size representations
    - Final projection to specified feature dimension
    
    Architecture Details:
    - Dilation pattern: [1, 2, 4, 8, 16, 32, 64, 128] (2^l for l=0 to 7)
    - Receptive field: Covers the entire input sequence
    - Skip connections: Enable stable gradient flow
    - Output dimension: 512-dimensional feature vectors
    
    Args:
        input_shape (tuple): Shape of input (sequence_length, channels)
        num_blocks (int): Number of dilated blocks (default: 8)
        base_channels (int): Base number of channels (default: 64)
        dropout (float): Dropout probability (default: 0.1)
    """

    def __init__( self, input_shape=(4000, 1), num_blocks=8, base_channels=64, dropout=0.1):
        super(DilatedTemporalConvEncoder, self).__init__()
        
        self.input_length = input_shape[0]
        self.input_channels = input_shape[1] if len(input_shape) > 1 else 1
        self.num_blocks = num_blocks
        self.base_channels = base_channels

        self.initial_conv = nn.Sequential(
            nn.Conv1d(self.input_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU( inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Calculate the sequence length after initial downsampling
        # After maxpool (stride=2): length / 4
        self.downsampled_length = self.input_length // 4
        
        # Eight residual dilated blocks with exponentially increasing dilation
        self.dilated_blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation_rate = 2 ** i  #[1, 2, 4, 8, 16, 32, 64, 128] 
            block = ResidualDilatedBlock(base_channels, dilation_rate, dropout)
            self.dilated_blocks.append(block)

        # Global adaptive average pooling for fixed-size output
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.projection = nn.Sequential(
            nn.Linear(base_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 512)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance( m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_( m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        """
        Forward pass through the TCN encoder
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length) or 
                            (batch_size, channels, sequence_length)
        
        Returns:
            torch.Tensor: Encoded features of shape (batch_size, 512)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        elif x.dim() == 3 and x.size(1) == self.input_length:
            x = x.transpose(1, 2)  #Swap sequence and channel dimensions if needed

        #Initial downsampling
        x = self.initial_conv(x)
        # Pass through dilated blocks sequentially
        for block in self.dilated_blocks: 
            x = block(x) 
        
        x = self.global_pool(x)  # (batch_size, base_channels, 1)
        x = x.squeeze(-1)  #(batch_size, base_channels)

        features = self.projection(x)
        return features

    def get_receptive_field(self):
        """
        Calculate the theoretical receptive field of the network
        
        Returns:
            int: Receptive field size in samples
        """

        receptive_field = 7
        stride = 2  # Initial conv stride
        
        receptive_field += (3 - 1) * stride 
        stride *= 2  # Maxpool stride

        # Add dilated blocks contribution 
        for i in range(self.num_blocks):
            dilation = 2 ** i
            # Each block has two conv layers with kernel_size=3
            receptive_field += 2 * (3 - 1) * dilation * stride

        
        return receptive_field


class DilatedBlock(nn.Module):
    """Legacy dilated block for backward compatibility"""
    def __init__(self, in_channels, out_channels, dilation_rate, dropout = 0.1):
        super(DilatedBlock, self).__init__()
        self.conv=nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                             dilation=dilation_rate, padding=dilation_rate) 
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.conv(x)
        x=self.norm(x)
        x = self.activation( x)
        x=self.dropout(x)
        return x


class DilatedConvEncoder( nn.Module):
    """Legacy dilated convolution encoder for backward compatibility"""

    def __init__(self, input_shape=(4000, 1)):
        super(DilatedConvEncoder, self).__init__()
        
        self.input_length = input_shape[0]
        self.input_channels = input_shape[1] if len(input_shape) > 1 else 1
        
        self.conv1=nn.Conv1d(self.input_channels,32, kernel_size=7, stride=2, padding=3)
        self.norm1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        # Dilated blocks
        self.dilated1=DilatedBlock(32, 64, dilation_rate=1)
        self.dilated2=DilatedBlock(64, 64, dilation_rate=2)
        self.dilated3 = DilatedBlock(64,128, dilation_rate=4)
        self.dilated4 = DilatedBlock(128, 128, dilation_rate=8)

        self.global_pool=nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128,512)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool1( x)

        x = self.dilated1(x)
        x = self.dilated2(x) 
        x = self.dilated3(x)
        x = self.dilated4(x)

        x = self.global_pool(x).squeeze(-1) 
        x = self.fc(x)
        return x



Encoder = DilatedTemporalConvEncoder 
