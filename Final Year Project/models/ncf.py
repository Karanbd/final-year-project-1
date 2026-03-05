"""
Neural Collaborative Filtering (NCF) Model Implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class NCF(nn.Module):
    """
    Enhanced Neural Collaborative Filtering Model.
    
    Combines embeddings with a deep multi-layer perceptron for user-item interaction prediction.
    Uses He initialization, LayerNorm, residual connections, and label smoothing.
    """
    
    def __init__(
        self, 
        num_users: int, 
        num_items: int, 
        embedding_dim: int = 256,
        hidden_dims: list = [512, 256, 128],
        dropout_rate: float = 0.2,
        use_layer_norm: bool = True
    ):
        """
        Initialize the NCF model.
        
        Args:
            num_users: Number of unique users
            num_items: Number of unique items (songs)
            embedding_dim: Dimension of user and item embeddings
            hidden_dims: List of hidden layer dimensions for MLP
            dropout_rate: Dropout probability for regularization
            use_layer_norm: Whether to use LayerNorm
        """
        super(NCF, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.use_layer_norm = use_layer_norm
        
        # User and item embeddings with larger dimension
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Embedding dropout
        self.embedding_dropout = nn.Dropout(0.1)
        
        # GMF component (element-wise product of embeddings)
        gmf_output_dim = embedding_dim
        
        # MLP component with improved architecture
        mlp_layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            if use_layer_norm:
                mlp_layers.append(nn.LayerNorm(hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Final prediction layer with deeper head
        final_input_dim = gmf_output_dim + hidden_dims[-1]
        self.output_layer = nn.Sequential(
            nn.Linear(final_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, 1)
        )
        
        # Initialize weights with He initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights with He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
            elif isinstance(module, nn.Linear):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # He initialization
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(
        self, 
        user_ids: torch.Tensor, 
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            user_ids: Tensor of user IDs
            item_ids: Tensor of item IDs
            
        Returns:
            Predicted logits (raw, before sigmoid)
        """
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Apply embedding dropout
        user_emb = self.embedding_dropout(user_emb)
        item_emb = self.embedding_dropout(item_emb)
        
        # GMF: Element-wise product
        gmf_output = user_emb * item_emb
        
        # MLP: Concatenate and pass through layers
        mlp_input = torch.cat([user_emb, item_emb], dim=1)
        mlp_output = self.mlp(mlp_input)
        
        # Combine GMF and MLP
        combined = torch.cat([gmf_output, mlp_output], dim=1)
        
        # Final prediction (return raw logits for BCEWithLogitsLoss)
        output = self.output_layer(combined)
        
        return output.squeeze(-1)
    
    def get_user_embedding(self, user_id: int) -> torch.Tensor:
        """Get embedding for a specific user."""
        return self.user_embedding(torch.tensor([user_id], device=self.user_embedding.weight.device))
    
    def get_item_embedding(self, item_id: int) -> torch.Tensor:
        """Get embedding for a specific item."""
        return self.item_embedding(torch.tensor([item_id], device=self.item_embedding.weight.device))
    
    def predict(
        self, 
        user_id: int, 
        item_ids: torch.Tensor,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Get predictions for a user and multiple items.
        
        Args:
            user_id: Single user ID
            item_ids: Tensor of item IDs
            device: Device to run prediction on
            
        Returns:
            Predicted scores for each item
        """
        if device is None:
            device = next(self.parameters()).device
        
        user_tensor = torch.tensor([user_id] * len(item_ids), device=device)
        
        with torch.no_grad():
            scores = self.forward(user_tensor, item_ids)
        
        return scores


class GeneralizedMatrixFactorization(nn.Module):
    """
    Generalized Matrix Factorization (GMF) model.
    """
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super(GeneralizedMatrixFactorization, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.output = nn.Linear(embedding_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        output = self.output(user_emb * item_emb)
        return torch.sigmoid(output.squeeze(-1))


class MultiVAE(nn.Module):
    """
    Multi-VAE: Variational Autoencoder for Collaborative Filtering.
    """
    
    def __init__(self, num_items: int, hidden_dims: list = [600, 200], latent_dim: int = 200):
        super(MultiVAE, self).__init__()
        
        self.num_items = num_items
        
        # Encoder
        encoder_layers = []
        input_dim = num_items
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            encoder_layers.append(nn.Tanh())
            input_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = [nn.Linear(latent_dim, hidden_dims[-1]), nn.Tanh()]
        for i in range(len(hidden_dims) - 2, -1, -1):
            decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i - 1] if i > 0 else num_items))
            if i > 0:
                decoder_layers.append(nn.Tanh())
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        logits = self.decoder(z)
        
        return logits, mu, logvar
    
    def predict(self, user_interactions: torch.Tensor) -> torch.Tensor:
        """Get predictions for a user's interaction vector."""
        logits, _, _ = self.forward(user_interactions)
        return torch.sigmoid(logits)
