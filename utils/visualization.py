"""
Visualization and Utility Functions

This module contains visualization utilities and helper functions for the
heart sound classification framework.
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch 

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


class VisualizationUtils:
    """
    Utility class for creating visualizations and plots
    """
    
    @staticmethod
    def plot_training_history(history, save_path=None):
        """
        Visualize training history
        
        Parameters:
        -----------
        history : dict
            Training history
        save_path : str
            Path to save the plot
        """
        # Create figure with multiple subplots
        plt.figure(figsize = (12,5)) 
        
        # Plot training & validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'])
        if 'val_acc' in history and history['val_acc']:
            plt.plot(history['val_acc'])
            plt.legend(['Train', 'Validation'], loc='lower right')
        else:
            plt.legend(['Train'], loc='lower right')
        plt.title('Model accuracy')
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Epoch')

        # Plot training & validation loss 
        plt.subplot(1,2,2)
        plt.plot(history['train_loss'])
        if 'val_loss' in history and history['val_loss']:
            plt.plot(history['val_loss'])
            plt.legend(['Train', 'Validation'], loc='upper right')
        else:
            plt.legend(['Train'], loc='upper right')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        
        plt.tight_layout()

        
        if save_path: 
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
    
    @staticmethod
    def visualize_features(model, data, labels, method='tsne', title_prefix="", save_prefix=""):
        """
        Visualize learned features using dimensionality reduction
        
        Parameters:
        -----------
        model : DualPathContrastiveLearning
            Trained model
        data : numpy.ndarray
            Input data
        labels : numpy.ndarray
            Labels for coloring
        method : str
            Dimensionality reduction method ('tsne', 'pca', 'umap')
        title_prefix : str
            Prefix for plot titles
        save_prefix : str
            Prefix for saved files
        """
        model.audio_encoder.eval()
        model.spec_encoder.eval() 
        model.fusion_head.eval( )


        with torch.no_grad( ):
            if isinstance(data, np.ndarray):
                data = torch.FloatTensor(data).to(model.device)

            audio_features, spec_features, fused_features = model.get_features(data) 


            # Convert to numpy
            audio_features = audio_features.cpu().numpy()
            spec_features = spec_features.cpu().numpy() 
            fused_features = fused_features.cpu().numpy()

        
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data)//4))
        elif method == 'pca':
            reducer = PCA(n_components = 2, random_state=42)
        elif method == 'umap' and HAS_UMAP:
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            print(f"Method {method} not available, using PCA")
            reducer = PCA(n_components=2,random_state=42)

        features_dict = {
            'Audio (TCN)': audio_features,
            'Spectrogram (ResNet)': spec_features,
            'Fused': fused_features
        }

        
        #Create subplots
        fig, axes = plt.subplots( 1, 3, figsize=(18, 5))

        for idx, (feature_name, features) in enumerate(features_dict.items()):
            features_2d=reducer.fit_transform(features)

            scatter=axes[idx].scatter(
                features_2d[:, 0], features_2d[:, 1], 
                c=labels, cmap='viridis', alpha=0.7, s=50
            )
            axes[idx].set_title(f'{title_prefix}{feature_name} Features ({method.upper()})')
            axes[idx].set_xlabel(f'{method.upper()} Component 1')
            axes[idx].set_ylabel(f'{method.upper()} Component 2')

            plt.colorbar(scatter,ax=axes[idx])
        plt.tight_layout( )

        if save_prefix:
            save_path = f"{save_prefix}_{method}_features.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature visualization saved to: {save_path}")

        plt.show()

        return fig

    @staticmethod
    def plot_audio_samples( audio_data,labels, n_samples=4, sampling_rate=4000, save_path=None):
        """
        Plot sample audio waveforms
        
        Parameters:
        -----------
        audio_data : numpy.ndarray
            Audio data
        labels : numpy.ndarray
            Labels
        n_samples : int
            Number of samples to plot per class
        sampling_rate : int
            Sampling rate
        save_path : str
            Path to save the plot
        """
        fig, axes = plt.subplots(2, n_samples, figsize = (15, 6))
        
        unique_labels = np.unique( labels)


        for class_idx, label in enumerate(unique_labels):
            class_mask = labels == label
            class_data = audio_data[class_mask]
            
            # Select random samples 
            sample_indices = np.random.choice(len(class_data),min(n_samples, len(class_data)), replace = False)
            for sample_idx in range(n_samples):
                if sample_idx < len(sample_indices):
                    sample = class_data[sample_indices[sample_idx]]
                    time = np.arange(len(sample)) / sampling_rate

                    axes[class_idx, sample_idx].plot(time, sample)
                    axes[class_idx, sample_idx].set_title(f'Class {label} - Sample {sample_idx+1}')
                    axes[class_idx, sample_idx].set_xlabel('Time (s)')
                    axes[class_idx, sample_idx].set_ylabel('Amplitude')
                    axes[class_idx, sample_idx].grid(True, alpha=0.3)
                else:
                    axes[class_idx, sample_idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show() 

        return fig
    
    @staticmethod 
    def plot_model_architecture_summary():
        """
        Create a summary plot of the dual-path architecture
        """
        fig, ax=plt.subplots(figsize=(12, 8))

        # Remove axes 
        ax.set_xlim(0, 10)
        ax.set_ylim(0,10)
        ax.axis('off')
        
        ax.text(5, 9.5, 'Dual-Path Contrastive Learning Architecture', 
                fontsize=20, fontweight='bold', ha='center')
        
        # Input
        ax.add_patch(plt.Rectangle((4, 8), 2, 0.5, fill=True, color='lightblue', alpha=0.7))
        ax.text(5, 8.25, 'Raw Audio\n(4000 samples)', ha='center', va='center', fontsize=10)

        
        # Audio path (left)
        ax.add_patch(plt.Rectangle((1, 6.5), 3, 1, fill=True, color='lightgreen', alpha=0.7))
        ax.text(2.5, 7, 'TCN Encoder\n(Dilated Temporal\nConvolutions)', ha='center', va='center', fontsize=10)
        
        # Spectrogram path (right)
        ax.add_patch(plt.Rectangle((6, 6.5), 3, 1, fill=True, color='lightcoral', alpha=0.7))
        ax.text(7.5, 7, 'Mel-Spectrogram\n+\nResNet-50 Encoder', ha='center', va='center', fontsize=10)

        ax.add_patch(plt.Rectangle((1, 5), 2, 0.5, fill=True, color='lightgreen', alpha=0.5))
        ax.text(2, 5.25, 'Audio Features\n(512-dim)', ha='center', va='center', fontsize=9)
        ax.add_patch(plt.Rectangle((7, 5), 2, 0.5, fill=True, color='lightcoral', alpha=0.5))
        ax.text(8, 5.25, 'Spec Features\n(512-dim)', ha='center', va='center', fontsize=9)

        
        ax.add_patch(plt.Rectangle((4, 3.5), 2, 0.5, fill=True, color='gold', alpha=0.7))
        ax.text(5, 3.75, 'Attention Fusion\n(512-dim)', ha='center', va='center', fontsize=10)
        ax.add_patch(plt.Rectangle((1, 2), 2, 0.5, fill=True, color='mediumpurple', alpha=0.7))
        ax.text(2, 2.25, 'Projection Head\n(Contrastive)', ha='center', va='center', fontsize=9)
        
        ax.add_patch(plt.Rectangle((4, 2), 2, 0.5, fill=True, color='orange', alpha=0.7))
        ax.text(5, 2.25, 'Classifier Head\n(Supervised)', ha='center', va='center', fontsize=9)

        ax.add_patch(plt.Rectangle((7, 2), 2, 0.5, fill=True, color='pink', alpha=0.7))
        ax.text(8, 2.25, 'Prototypical Head\n(Few-shot)', ha='center', va='center', fontsize=9)
        
        ax.arrow(4.5, 8, -1.8, -0.4, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax.arrow(5.5, 8, 1.8, -0.4, head_width=0.1, head_length=0.1, fc='black', ec='black')
        # Paths to features
        ax.arrow(2.5, 6.5, 0, -0.9, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax.arrow(7.5, 6.5, 0, -0.9, head_width=0.1, head_length=0.1, fc='black', ec='black')


        ax.arrow(2.5, 5, 2, -1.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax.arrow(7.5, 5, -2, -1.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        ax.arrow(4.5, 3.5, -1.8, -1.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax.arrow(5, 3.5, 0, -0.9, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax.arrow(5.5, 3.5, 1.8, -1.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Legend
        ax.text(0.5, 1, 'Key Features:', fontsize=12, fontweight='bold')
        ax.text(0.5, 0.7, '• Dual-path processing (time + frequency domain)', fontsize=10)
        ax.text(0.5, 0.4, '• Multi-modal contrastive learning', fontsize=10)
        ax.text(0.5, 0.1, '• Advanced TCN for temporal modeling', fontsize=10)

        
        plt.tight_layout()
        plt.show()

        return fig

    def plot_training_curves(self,epochs,train_losses,val_losses = None, save_path=None):
        """
        Plot training and validation loss curves
        
        Args:
            epochs (list): List of epoch numbers
            train_losses (list): Training losses
            val_losses (list, optional): Validation losses
            save_path (str, optional): Path to save the plot
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        if val_losses is not None:
            ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress')
        ax.legend( )
        ax.grid(True,alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            class_names (list, optional): Class names for display
            save_path (str, optional): Path to save the plot
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax = ax)

        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        ax.set(xticks = np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels = class_names,
               yticklabels=class_names,
               title='Confusion Matrix',
               ylabel='True Label',
               xlabel='Predicted Label')

        thresh = cm.max( ) / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_publication_ready_results(models_results, save_prefix="publication"):
    """
    Create publication-ready figures and tables
    
    Parameters:
    -----------
    models_results : dict
        Dictionary of model results
    save_prefix : str
        Prefix for saved files
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    metrics = ['AUROC', 'AUPRC', 'F1-Score', 'Accuracy', 'Brier Score', 'ECE']
    table_data = []

    for model_name, results in models_results.items(): 
        row = [model_name]
        for metric in ['auroc', 'auprc', 'f1_score', 'accuracy', 'brier_score', 'ece']:
            value = results['metrics'][metric]
            ci = results['bootstrap_ci'][metric]
            cell_text = f"{value:.3f}\n[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]"
            row.append(cell_text)
        table_data.append( row)

    # Create table 
    table = ax.table(cellText=table_data, 
                    colLabels=['Model'] + metrics,
                    cellLoc='center',
                    loc='center',
                    colWidths = [0.2] + [0.13]*len(metrics))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)

    for i in range(len(metrics) + 1):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Model Performance Comparison with 95% Confidence Intervals', 
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_performance_table.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Publication-ready results saved with prefix: {save_prefix}")

def visualize_filters(model, layer_idx, save_path=None):
    """
    Visualize filters of a convolutional layer
    
    Parameters:
    -----------
    model : nn.Module
        Model
    layer_idx : int
        Index of the convolutional layer
    save_path : str
        Path to save the plot
    """
    layer = None
    for i,(name,module) in enumerate(model.named_modules()):
        if isinstance(module, torch.nn.Conv1d) and i == layer_idx:
            layer = module
            break

    if layer is None or not isinstance(layer, torch.nn.Conv1d):
        print(f"Could not find Conv1d layer at index {layer_idx}")
        return
    
    # Get filters
    filters = layer.weight.data.cpu().numpy()

    n_filters = min(16, filters.shape[0])
    # Create figure
    fig,axes = plt.subplots(4,4,figsize=(10, 10))

    for i,ax in enumerate(axes.flat): 
        if i < n_filters: 
            ax.plot(filters[i, 0, :])
            ax.set_title(f'Filter {i+1}')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.suptitle(f'Filters from convolutional layer {layer_idx}')
    plt.subplots_adjust(  top=0.92)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


Visualizer = VisualizationUtils
