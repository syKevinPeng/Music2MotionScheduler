import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import lightning as pl

class Baseline(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 128,  # Jukebox embedding dimension
        num_labels: int = 10,  # Number of predefined dance labels
        seq_length: int = 5,   # Number of 5s chunks (adjust based on input)
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Transformer Encoder (captures temporal relationships between chunks)
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # Positional encoding (for sequence order)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_length, d_model))
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_length, input_dim)
        Returns:
            logits: (batch_size, seq_length, num_labels)
        """
        # Project input
        x = self.input_proj(x)  # (B, S, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (B, S, d_model)
        
        # Predict labels for all positions
        logits = self.classifier(x)  # (B, S, num_labels)
        # Apply softmax to get probabilities
        logits = F.softmax(logits, dim=-1)
        return logits
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
    
    def loss_function(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, seq_length, num_labels)
            labels: (batch_size, seq_length, num_labels)
        Returns:
            loss: scalar tensor
        """
        # Reshape logits and labels for loss calculation
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1, logits.size(-1))
        
        # Calculate loss
        loss = F.cross_entropy(logits, labels)
        
        return loss
    def training_step(self, batch, batch_idx):
        audio_feature, label = batch
        logits = self(audio_feature)
        print(f'output shape: {logits.shape}, label shape: {label.shape}')
        loss = self.loss_function(logits, label)
        
        # Log the loss
        self.log('train_loss', loss)
        
        return loss
    
    def prediction_step(self, batch, batch_idx):
        audio_feature, label = batch
        logits = self(audio_feature)
        
        # Get the predicted labels
        _, predicted_labels = torch.max(logits, dim=-1)
        
        return predicted_labels
