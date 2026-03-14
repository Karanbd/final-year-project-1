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
    create_clustered_interactions, encode_ids, create_negative_samples,
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
    batch_size: int = 512,
    lr: float = 0.0005,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    val_df: Optional[pd.DataFrame] = None,
    patience: int = 7,
    checkpoint_path: Optional[str] = None,
    use_class_weights: bool = True,
    positive_weight: float = 3.0,
    label_smoothing: float = 0.1,
    warmup_epochs: int = 3,
    gradient_clip: float = 1.0
) -> nn.Module:
    """
    Train a recommendation model with improved techniques.
    
    Args:
        model: PyTorch model
        train_df: Training DataFrame
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay for regularization
        device: Device to train on
        val_df: Optional validation DataFrame
        patience: Early stopping patience
        checkpoint_path: Path to save best model
        use_class_weights: Whether to use class weights
        positive_weight: Weight for positive samples
        label_smoothing: Label smoothing factor
        warmup_epochs: Number of warmup epochs
        gradient_clip: Gradient clipping threshold
        
    Returns:
        Trained model
    """
    if device is None:
        device = get_device()
    
    model = model.to(device)
    
    # Create dataset and dataloader
    dataset = MusicInteractionDataset(train_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Loss function with class weights and label smoothing
    if use_class_weights:
        # Calculate class weights
        num_positive = (train_df['interaction'] == 1).sum()
        num_negative = (train_df['interaction'] == 0).sum()
        pos_weight = torch.tensor([num_negative / num_positive * positive_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCELoss()
    
    # Optimizer with AdamW
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler with warmup
    if warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )
    else:
        warmup_scheduler = None
        main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    global_step = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        num_batches = 0
        
        for users, items, labels in dataloader:
            users = users.to(device)
            items = items.to(device)
            labels = labels.to(device).float()
            
            optimizer.zero_grad()
            preds = model(users, items)
            
            # Apply label smoothing to labels
            if label_smoothing > 0:
                smoothed_labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing
                loss = criterion(preds, smoothed_labels)
            else:
                loss = criterion(preds, labels)
            loss.backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            global_step += 1
        
        avg_loss = total_loss / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        # Learning rate warmup
        if warmup_scheduler is not None and epoch < warmup_epochs:
            warmup_scheduler.step()
        
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
                    labels = labels.to(device).float()
                    
                    preds = model(users, items)
                    loss = criterion(preds, labels)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}")
            
            # Learning rate scheduling after warmup
            if warmup_scheduler is None or epoch >= warmup_epochs:
                main_scheduler.step(avg_val_loss)
            
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
    Run the NCF pipeline with improved training.
    
    Args:
        embeddings_dict: Dictionary of song embeddings
        device: Device to use
        
    Returns:
        Tuple of (trained model, evaluation results)
    """
    logger.info("=" * 50)
    logger.info("Running NCF Pipeline - IMPROVED")
    logger.info("=" * 50)
    
    # Step 1: Create or load interactions
    if os.path.exists(config.INTERACTIONS_SAVE_PATH):
        logger.info("Loading existing interactions...")
        df = pd.read_csv(config.INTERACTIONS_SAVE_PATH)
    else:
        logger.info("Creating CLUSTERED interactions with realistic user tastes...")
        df = create_clustered_interactions(
            embeddings_dict,
            num_users=config.NUM_USERS,
            songs_per_user=(config.MIN_SONGS_PER_USER + config.MAX_SONGS_PER_USER) // 2,
            num_clusters=20,
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
    
    # Step 4: Create negative samples (with popularity-based if enabled)
    if config.USE_POPULARITY_NEGATIVE:
        logger.info("Using popularity-based negative sampling...")
        item_popularity = get_item_popularity(train_df)
        train_df = create_popularity_based_negative_samples(
            train_df, num_songs, item_popularity, config.NEGATIVE_SAMPLE_RATIO
        )
        test_df = create_popularity_based_negative_samples(
            test_df, num_songs, item_popularity, config.NEGATIVE_SAMPLE_RATIO
        )
    else:
        train_df = create_negative_samples(train_df, num_songs, config.NEGATIVE_SAMPLE_RATIO)
        test_df = create_negative_samples(test_df, num_songs, config.NEGATIVE_SAMPLE_RATIO)
    
    logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Step 5: Train NCF model with improved settings
    ncf_model = NCF(
        num_users=num_users,
        num_items=num_songs,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dims=config.NCF_HIDDEN_DIMS,
        dropout_rate=config.DROPOUT,
        use_layer_norm=True
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
        checkpoint_path=checkpoint_path,
        use_class_weights=config.USE_CLASS_WEIGHTS,
        positive_weight=config.POSITIVE_WEIGHT,
        label_smoothing=config.LABEL_SMOOTHING,
        warmup_epochs=config.WARMUP_EPOCHS,
        gradient_clip=config.GRADIENT_CLIP
    )
    
    # Step 6: Evaluate (excluding training items from recommendations)
    logger.info("Evaluating NCF model...")
    results = evaluate_model(
        ncf_model, train_df, test_df, num_songs, 
        k_values=[5, 10, 20], 
        device=device,
        exclude_train_items=True  # Important fix!
    )
    print_evaluation_results(results)
    
    # Save the trained NCF model
    ncf_save_path = config.NCF_MODEL_PATH
    torch.save(ncf_model.state_dict(), ncf_save_path)
    logger.info(f"Saved NCF model to {ncf_save_path}")
    
    return ncf_model, results


def run_hybrid_pipeline(
    embeddings_dict: Dict[str, torch.Tensor],
    device: torch.device,
    num_users: int = 500,
    num_songs: int = None,
    ncf_train_df: pd.DataFrame = None
) -> Tuple[HybridModel, Dict]:
    """
    Run the Hybrid pipeline with improved training.
    
    Args:
        embeddings_dict: Dictionary of song embeddings
        device: Device to use
        num_users: Number of users
        num_songs: Number of songs (should match NCF)
        ncf_train_df: The NCF training DataFrame with encoded IDs (to use same songs)
        
    Returns:
        Tuple of (trained model, evaluation results)
    """
    logger.info("=" * 50)
    logger.info("Running Hybrid Pipeline - IMPROVED")
    logger.info("=" * 50)
    
    # Load audio embeddings
    ast_embeddings = torch.load(config.EMBEDDINGS_SAVE_PATH)
    embedding_dim = 768
    
    # CRITICAL: Use num_songs from NCF if provided, otherwise fall back to embeddings
    # This ensures both models use the same song IDs!
    if num_songs is None:
        num_songs = len(ast_embeddings)
    logger.info(f"Number of songs: {num_songs}")
    
    # If we have NCF training data, use the same songs!
    if ncf_train_df is not None:
        # Get unique song IDs from NCF training data
        ncf_song_ids = set(ncf_train_df['song_id'].unique())
        num_songs = len(ncf_song_ids)
        logger.info(f"Using {num_songs} songs from NCF training data (matching NCF model)")
    
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
        likes_per_user=config.MIN_SONGS_PER_USER,  # More songs per user
        random_seed=config.RANDOM_SEED
    )
    
    # Train/test split
    train_df, test_df = train_test_split_by_user(content_df, test_ratio=config.TEST_RATIO)
    
    # Add negative samples (with popularity-based if enabled)
    if config.USE_POPULARITY_NEGATIVE:
        logger.info("Using popularity-based negative sampling...")
        item_popularity = get_item_popularity(train_df)
        train_df = create_popularity_based_negative_samples(
            train_df, num_songs, item_popularity, config.NEGATIVE_SAMPLE_RATIO
        )
        test_df = create_popularity_based_negative_samples(
            test_df, num_songs, item_popularity, config.NEGATIVE_SAMPLE_RATIO
        )
    else:
        train_df = create_negative_samples(train_df, num_songs, config.NEGATIVE_SAMPLE_RATIO)
        test_df = create_negative_samples(test_df, num_songs, config.NEGATIVE_SAMPLE_RATIO)
    
    # Train Hybrid model with attention mechanism
    hybrid_model = HybridModel(
        num_users=num_users,
        num_items=num_songs,
        audio_embedding_dim=embedding_dim,
        user_embedding_dim=config.EMBEDDING_DIM,
        hidden_dims=config.HYBRID_HIDDEN_DIMS,
        dropout_rate=config.DROPOUT,
        use_attention=True,
        num_heads=config.ATTENTION_HEADS
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
        checkpoint_path=checkpoint_path,
        use_class_weights=config.USE_CLASS_WEIGHTS,
        positive_weight=config.POSITIVE_WEIGHT,
        label_smoothing=config.LABEL_SMOOTHING,
        warmup_epochs=config.WARMUP_EPOCHS,
        gradient_clip=config.GRADIENT_CLIP
    )
    
    # Evaluate (excluding training items)
    logger.info("Evaluating Hybrid model...")
    results = evaluate_model(
        hybrid_model, train_df, test_df, num_songs, 
        k_values=[5, 10, 20], 
        device=device, 
        audio_embeddings=normalized_audio,
        exclude_train_items=True  # Important fix!
    )
    print_evaluation_results(results)
    
    # Save the trained Hybrid model
    hybrid_save_path = config.HYBRID_MODEL_PATH
    torch.save(hybrid_model.state_dict(), hybrid_save_path)
    logger.info(f"Saved Hybrid model to {hybrid_save_path}")
    
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
    
    # Get num_songs from NCF to use in Hybrid
    df = pd.read_csv(config.INTERACTIONS_SAVE_PATH)
    df, _, _ = encode_ids(df)
    num_songs = df["song_id"].nunique()
    logger.info(f"Using {num_songs} songs for Hybrid model (matching NCF)")
    
    # Step 3: Run Hybrid pipeline
    hybrid_model, hybrid_results = run_hybrid_pipeline(embeddings_dict, device, num_songs=num_songs)
    
    logger.info("=" * 50)
    logger.info("All pipelines completed!")
    logger.info("=" * 50)
    
    return ncf_model, hybrid_model, ncf_results, hybrid_results


if __name__ == "__main__":
    main()
