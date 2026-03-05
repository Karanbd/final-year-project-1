"""
Hybrid Recommendation Model Implementation.
Combines collaborative filtering with audio content features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class HybridModel(nn.Module):
    """
    Enhanced Hybrid Recommendation Model.
    
    Combines user embeddings with audio content embeddings using attention mechanism.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        audio_embedding_dim: int = 768,
        user_embedding_dim: int = 256,
        hidden_dims: list = [512, 256, 128],
        dropout_rate: float = 0.3,
        use_attention: bool = True,
        num_heads: int = 4
    ):
        """
        Initialize the Hybrid model.
        
        Args:
            num_users: Number of unique users
            num_items: Number of items (songs)
            audio_embedding_dim: Dimension of audio embeddings (from AST)
            user_embedding_dim: Dimension of user embeddings
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability for regularization
            use_attention: Whether to use attention mechanism
            num_heads: Number of attention heads
        """
        super(HybridModel, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.audio_embedding_dim = audio_embedding_dim
        self.user_embedding_dim = user_embedding_dim
        self.use_attention = use_attention
        
        # User embedding layer
        self.user_embedding = nn.Embedding(num_users, user_embedding_dim)
        
        # Item (song) embedding layer for audio
        self.item_embedding = nn.Embedding(num_items, audio_embedding_dim)
        
        # Embedding dropout
        self.embedding_dropout = nn.Dropout(0.1)
        
        # Audio embedding projection with deeper network
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_embedding_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, user_embedding_dim)
        )
        
        # Cross-attention for user-audio interaction
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=user_embedding_dim,
                num_heads=num_heads,
                dropout=dropout_rate,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(user_embedding_dim)
        
        # Combined MLP with deeper architecture
        mlp_layers = []
        input_dim = user_embedding_dim * 2  # user + projected audio
        
        for hidden_dim in hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.LayerNorm(hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Output layer with deeper head
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
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
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the hybrid model.
        
        Args:
            user_ids: Tensor of user IDs
            item_ids: Tensor of item IDs
            
        Returns:
            Predicted probabilities
        """
        # Get user embeddings
        user_emb = self.user_embedding(user_ids)
        
        # Get item (audio) embeddings
        audio_emb_raw = self.item_embedding(item_ids)
        
        # Apply embedding dropout
        user_emb = self.embedding_dropout(user_emb)
        
        # Project audio embeddings to same dimension
        audio_emb = self.audio_projection(audio_emb_raw)
        
        # Apply attention if enabled
        if self.use_attention:
            # Prepare for attention: [batch, 1, dim]
            user_expanded = user_emb.unsqueeze(1)
            audio_expanded = audio_emb.unsqueeze(1)
            
            # Cross-attention
            attended, _ = self.attention(user_expanded, audio_expanded, audio_expanded)
            attended = attended.squeeze(1)
            
            # Residual connection
            user_emb = user_emb + self.attention_norm(attended)
        
        # Concatenate user and audio embeddings
        combined = torch.cat([user_emb, audio_emb], dim=1)
        
        # Pass through MLP
        mlp_output = self.mlp(combined)
        
        # Final prediction
        output = self.output_layer(mlp_output)
        
        return output.squeeze(-1)
    
    def get_user_embedding(self, user_id: int) -> torch.Tensor:
        """Get embedding for a specific user."""
        return self.user_embedding(torch.tensor([user_id], device=self.user_embedding.weight.device))
    
    def predict(
        self,
        user_id: int,
        item_ids: torch.Tensor,
        device: torch.device = None
    ) -> torch.Tensor:
        """Get predictions for a user and multiple items."""
        if device is None:
            device = next(self.parameters()).device
        
        user_tensor = torch.tensor([user_id] * len(item_ids), device=device)
        
        with torch.no_grad():
            scores = self.forward(user_tensor, item_ids)
        
        return scores


class AttentionHybridModel(nn.Module):
    """
    Hybrid Model with Cross-Attention Mechanism.
    
    Uses attention to dynamically weight different aspects of audio features.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        audio_embedding_dim: int = 768,
        user_embedding_dim: int = 64,
        num_heads: int = 4,
        dropout_rate: float = 0.3
    ):
        super(AttentionHybridModel, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        
        # User embedding
        self.user_embedding = nn.Embedding(num_users, user_embedding_dim)
        
        # Item embedding for audio
        self.item_embedding = nn.Embedding(num_items, audio_embedding_dim)
        
        # Audio projection
        self.audio_projection = nn.Linear(audio_embedding_dim, user_embedding_dim)
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=user_embedding_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Feed-forward after attention
        self.feed_forward = nn.Sequential(
            nn.Linear(user_embedding_dim, user_embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(user_embedding_dim * 2, user_embedding_dim),
            nn.LayerNorm(user_embedding_dim)
        )
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(user_embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        # User embedding
        user_emb = self.user_embedding(user_ids)  # [batch, dim]
        
        # Get audio embeddings for items
        audio_emb_raw = self.item_embedding(item_ids)
        
        # Project audio
        audio_emb = self.audio_projection(audio_emb_raw)  # [batch, dim]
        
        # Prepare for attention: [batch, 1, dim]
        user_emb_expanded = user_emb.unsqueeze(1)
        audio_emb_expanded = audio_emb.unsqueeze(1)
        
        # Cross-attention: user attends to audio
        attended, _ = self.cross_attention(
            user_emb_expanded, audio_emb_expanded, audio_emb_expanded
        )
        
        # Residual connection with feed-forward
        combined = attended.squeeze(1) + user_emb
        combined = self.feed_forward(combined)
        
        # Final combination
        final_combined = torch.cat([combined, user_emb], dim=1)
        
        output = self.output_layer(final_combined)
        
        return output.squeeze(-1)


class DeepHybridModel(nn.Module):
    """
    Deep Hybrid Model with separate towers for user and audio.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        audio_embedding_dim: int = 768,
        user_tower_dim: int = 64,
        audio_tower_dim: int = 128,
        output_dim: int = 64,
        dropout_rate: float = 0.3
    ):
        super(DeepHybridModel, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        
        # User tower
        self.user_embedding = nn.Embedding(num_users, user_tower_dim)
        self.user_tower = nn.Sequential(
            nn.Linear(user_tower_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_dim)
        )
        
        # Audio tower
        self.item_embedding = nn.Embedding(num_items, audio_embedding_dim)
        self.audio_tower = nn.Sequential(
            nn.Linear(audio_embedding_dim, audio_tower_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(audio_tower_dim, output_dim)
        )
        
        # Output fusion
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)
        user_out = self.user_tower(user_emb)
        
        audio_emb = self.item_embedding(item_ids)
        audio_out = self.audio_tower(audio_emb)
        
        combined = torch.cat([user_out, audio_out], dim=1)
        output = self.fusion(combined)
        
        return output.squeeze(-1)
