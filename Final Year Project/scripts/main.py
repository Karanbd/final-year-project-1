"""
Main Orchestration Script for Music Recommendation System.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import random
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config
from models.ncf import NCF
from models.hybrid import HybridModel, AttentionHybridModel
from datasets.music_dataset import MusicInteractionDataset, MusicDatasetWithAudio, train_test_split
from utils.audio_processor import AudioProcessor, extract_embeddings_from_directory, normalize_embeddings
from utils.data_preparation import (
    create_random_interactions, encode_ids, create_negative_samples,
    create_content_based_interactions, train_test_split_by_user,
    create_audio_embedding_matrix, get_item_popularity, 
    create_popularity_based_negative_samples
)
from utils.evaluation import evaluate_model, print_evaluation_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the best available device."""
    return config.get_device()


def train_model(
    model: nn.Module,
    train_df: pd.DataFrame,
    num_epochs: int,
    batch_size: int = 256,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    device: Optional[torch.device] = None,
    val_df: Optional[pd.DataFrame] = None,
    patience: int = 3,
    checkpoint_path: Optional[str] = None
) -> nn.Module:
    """
    Train a recommendation model.
    
    Args:
        model: PyTorch model
        train_df: Training DataFrame
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        device: Device to train on
        val_df: Optional validation DataFrame
        patience: Early stopping patience
        checkpoint_path: Path to save best model
        
    Returns:
        Trained model
    """
    if device is None:
        device = get_device()
    
    model = model.to(device)
    
    # Create dataset and dataloader
    dataset = MusicInteractionDataset(train_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        num_batches = 0
        
        for users, items, labels in dataloader:
            users = users.to(device)
            items = items.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            preds = model(users, items)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")
        
        # Validation
        if val_df is not None:
            model.eval()
            val_dataset = MusicInteractionDataset(val_df)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for users, items, labels in val_loader:
                    users = users.to(device)
                    items = items.to(device)
                    labels = labels.to(device).unsqueeze(1)
                    
                    preds = model(users, items)
                    loss = criterion(preds, labels)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}")
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss - config.MIN_DELTA:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                
                if checkpoint_path:
                    torch.save(best_model_state, checkpoint_path)
                    logger.info(f"Saved best model to {checkpoint_path}")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    logger.info("Early stopping triggered!")
                    break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Loaded best model state")
    
    return model


def run_ncf_pipeline(
    embeddings_dict: Dict[str, torch.Tensor],
    device: torch.device
) -> Tuple[NCF, Dict]:
    """
    Run the NCF pipeline.
    
    Args:
        embeddings_dict: Dictionary of song embeddings
        device: Device to use
        
    Returns:
        Tuple of (trained model, evaluation results)
    """
    logger.info("=" * 50)
    logger.info("Running NCF Pipeline")
    logger.info("=" * 50)
    
    # Step 1: Create or load interactions
    if os.path.exists(config.INTERACTIONS_SAVE_PATH):
        logger.info("Loading existing interactions...")
        df = pd.read_csv(config.INTERACTIONS_SAVE_PATH)
    else:
        logger.info("Creating random interactions...")
        df = create_random_interactions(
            embeddings_dict,
            num_users=config.NUM_USERS,
            min_songs_per_user=config.MIN_SONGS_PER_USER,
            max_songs_per_user=config.MAX_SONGS_PER_USER,
            random_seed=config.RANDOM_SEED
        )
        df.to_csv(config.INTERACTIONS_SAVE_PATH, index=False)
    
    # Step 2: Encode IDs
    df, user_encoder, song_encoder = encode_ids(df)
    num_users = df["user_id"].nunique()
    num_songs = df["song_id"].nunique()
    logger.info(f"Users: {num_users}, Songs: {num_songs}")
    
    # Step 3: Train/test split
    train_df, test_df = train_test_split_by_user(df, test_ratio=config.TEST_RATIO)
    
    # Step 4: Create negative samples
    train_df = create_negative_samples(train_df, num_songs, config.NEGATIVE_SAMPLE_RATIO)
    test_df = create_negative_samples(test_df, num_songs, config.NEGATIVE_SAMPLE_RATIO)
    logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Step 5: Train NCF model
    ncf_model = NCF(
        num_users=num_users,
        num_items=num_songs,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dims=config.NCF_HIDDEN_DIMS,
        dropout_rate=0.2
    ).to(device)
    
    checkpoint_path = config.MODEL_CHECKPOINT_PATH.replace(".pt", "_ncf.pt")
    ncf_model = train_model(
        ncf_model,
        train_df,
        num_epochs=config.NCF_EPOCHS,
        batch_size=config.BATCH_SIZE,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        device=device,
        patience=config.PATIENCE,
        checkpoint_path=checkpoint_path
    )
    
    # Step 6: Evaluate
    logger.info("Evaluating NCF model...")
    results = evaluate_model(ncf_model, train_df, test_df, num_songs, k_values=[5, 10, 20], device=device)
    print_evaluation_results(results)
    
    return ncf_model, results


def run_hybrid_pipeline(
    embeddings_dict: Dict[str, torch.Tensor],
    device: torch.device,
    num_users: int = 500
) -> Tuple[HybridModel, Dict]:
    """
    Run the Hybrid pipeline.
    
    Args:
        embeddings_dict: Dictionary of song embeddings
        device: Device to use
        num_users: Number of users
        
    Returns:
        Tuple of (trained model, evaluation results)
    """
    logger.info("=" * 50)
    logger.info("Running Hybrid Pipeline")
    logger.info("=" * 50)
    
    # Load audio embeddings
    ast_embeddings = torch.load(config.EMBEDDINGS_SAVE_PATH)
    embedding_dim = 768
    
    # Get number of songs from embeddings
    num_songs = len(ast_embeddings)
    logger.info(f"Number of songs: {num_songs}")
    
    # Create audio embedding matrix
    audio_matrix = torch.zeros(num_songs, embedding_dim)
    for idx, emb in enumerate(ast_embeddings.values()):
        if idx < num_songs:
            audio_matrix[idx] = emb
    
    audio_matrix = audio_matrix.to(device)
    
    # Normalize embeddings
    normalized_audio = normalize_embeddings(audio_matrix, method="l2")
    
    # Create content-based interactions
    content_df = create_content_based_interactions(
        num_users=num_users,
        num_items=num_songs,
        normalized_audio_embeddings=normalized_audio,
        likes_per_user=30,
        random_seed=config.RANDOM_SEED
    )
    
    # Train/test split
    train_df, test_df = train_test_split_by_user(content_df, test_ratio=config.TEST_RATIO)
    
    # Add negative samples
    train_df = create_negative_samples(train_df, num_songs, config.NEGATIVE_SAMPLE_RATIO)
    test_df = create_negative_samples(test_df, num_songs, config.NEGATIVE_SAMPLE_RATIO)
    
    # Train Hybrid model
    hybrid_model = HybridModel(
        num_users=num_users,
        audio_embedding_dim=embedding_dim,
        user_embedding_dim=config.EMBEDDING_DIM,
        hidden_dims=config.HYBRID_HIDDEN_DIMS,
        dropout_rate=0.3
    ).to(device)
    
    checkpoint_path = config.MODEL_CHECKPOINT_PATH.replace(".pt", "_hybrid.pt")
    hybrid_model = train_model(
        hybrid_model,
        train_df,
        num_epochs=config.HYBRID_EPOCHS,
        batch_size=config.BATCH_SIZE,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        device=device,
        patience=config.PATIENCE,
        checkpoint_path=checkpoint_path
    )
    
    # Evaluate
    logger.info("Evaluating Hybrid model...")
    results = evaluate_model(hybrid_model, train_df, test_df, num_songs, k_values=[5, 10, 20], device=device)
    print_evaluation_results(results)
    
    return hybrid_model, results


def main():
    """Main execution function."""
    # Set seed
    set_seed(config.RANDOM_SEED)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Step 1: Extract audio embeddings
    logger.info("Step 1: Extracting audio embeddings...")
    base_path = config.AUDIO_BASE_PATH
    
    embeddings_dict = extract_embeddings_from_directory(
        base_path=base_path,
        max_songs=200,
        save_path=config.EMBEDDINGS_SAVE_PATH,
        load_existing=True,
        device=device
    )
    
    logger.info(f"Total embeddings: {len(embeddings_dict)}")
    
    # Step 2: Run NCF pipeline
    ncf_model, ncf_results = run_ncf_pipeline(embeddings_dict, device)
    
    # Step 3: Run Hybrid pipeline
    hybrid_model, hybrid_results = run_hybrid_pipeline(embeddings_dict, device)
    
    logger.info("=" * 50)
    logger.info("All pipelines completed!")
    logger.info("=" * 50)
    
    return ncf_model, hybrid_model, ncf_results, hybrid_results


if __name__ == "__main__":
    main()
