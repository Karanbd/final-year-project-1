"""
Data Preparation Utilities.
Functions for preparing and processing recommendation data.
"""

import torch
import pandas as pd
import numpy as np
import random
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_random_interactions(
    embeddings: Dict[str, torch.Tensor],
    num_users: int = 500,
    min_songs_per_user: int = 20,
    max_songs_per_user: int = 50,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Create random user-song interactions.
    
    Args:
        embeddings: Dictionary of song_id -> embedding
        num_users: Number of users
        min_songs_per_user: Minimum songs per user
        max_songs_per_user: Maximum songs per user
        random_seed: Random seed
        
    Returns:
        DataFrame with interactions
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    song_ids = list(embeddings.keys())
    
    interactions = []
    for user_id in range(num_users):
        num_listened = random.randint(min_songs_per_user, max_songs_per_user)
        listened_songs = random.sample(song_ids, min(num_listened, len(song_ids)))
        
        for song_id in listened_songs:
            interactions.append([user_id, song_id, 1])
    
    df = pd.DataFrame(interactions, columns=["user_id", "song_id", "interaction"])
    logger.info(f"Created {len(df)} interactions for {num_users} users")
    return df


def encode_ids(
    df: pd.DataFrame,
    user_col: str = "user_id",
    song_col: str = "song_id"
) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Encode user and song IDs.
    
    Args:
        df: DataFrame with user_id and song_id columns
        user_col: User column name
        song_col: Song column name
        
    Returns:
        Tuple of (encoded_df, user_encoder, song_encoder)
    """
    user_encoder = LabelEncoder()
    song_encoder = LabelEncoder()
    
    df = df.copy()
    df[user_col] = user_encoder.fit_transform(df[user_col])
    df[song_col] = song_encoder.fit_transform(df[song_col].astype(str))
    
    return df, user_encoder, song_encoder


def create_negative_samples(
    df: pd.DataFrame,
    num_items: int,
    negative_ratio: int = 3,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Create negative samples (items user hasn't interacted with).
    
    Args:
        df: DataFrame with positive interactions
        num_items: Total number of items
        negative_ratio: Ratio of negative to positive samples
        random_seed: Random seed
        
    Returns:
        DataFrame with positive and negative samples
    """
    np.random.seed(random_seed)
    
    all_items = set(range(num_items))
    new_data = []
    
    for user in df["user_id"].unique():
        user_items = set(df[df["user_id"] == user]["song_id"])
        non_interacted = list(all_items - user_items)
        
        if len(non_interacted) == 0:
            continue
        
        # Positive samples
        for item in user_items:
            new_data.append([user, item, 1])
        
        # Negative samples
        num_negatives = min(len(user_items) * negative_ratio, len(non_interacted))
        negative_items = np.random.choice(non_interacted, size=num_negatives, replace=False)
        
        for item in negative_items:
            new_data.append([user, item, 0])
    
    result_df = pd.DataFrame(new_data, columns=["user_id", "song_id", "interaction"])
    logger.info(f"Created dataset with {len(result_df)} samples (pos: {len(df)}, neg: {len(result_df) - len(df)})")
    return result_df


def create_popularity_based_negative_samples(
    df: pd.DataFrame,
    num_items: int,
    item_popularity: Optional[Dict[int, int]] = None,
    negative_ratio: int = 3,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Create negative samples with popularity-biased sampling.
    
    Items that are less popular have higher probability of being selected as negatives.
    This helps reduce popularity bias in the model.
    
    Args:
        df: DataFrame with positive interactions
        num_items: Total number of items
        item_popularity: Dictionary of item_id -> interaction count
        negative_ratio: Ratio of negative to positive samples
        random_seed: Random seed
        
    Returns:
        DataFrame with positive and negative samples
    """
    np.random.seed(random_seed)
    
    if item_popularity is None:
        # Calculate popularity from dataframe
        item_popularity = df["song_id"].value_counts().to_dict()
    
    # Calculate inverse popularity weights (less popular = higher weight)
    all_items = set(range(num_items))
    item_weights = {}
    
    for item in all_items:
        pop = item_popularity.get(item, 0)
        # Inverse popularity: add 1 to avoid division by zero
        item_weights[item] = 1.0 / (pop + 1)
    
    # Normalize weights
    total_weight = sum(item_weights.values())
    item_probs = {k: v / total_weight for k, v in item_weights.items()}
    
    # Create samples
    new_data = []
    
    for user in df["user_id"].unique():
        user_items = set(df[df["user_id"] == user]["song_id"])
        non_interacted = list(all_items - user_items)
        
        if len(non_interacted) == 0:
            continue
        
        # Get probabilities for non-interacted items
        neg_probs = np.array([item_probs.get(i, 0) for i in non_interacted])
        neg_probs = neg_probs / neg_probs.sum()
        
        # Positive samples
        for item in user_items:
            new_data.append([user, item, 1])
        
        # Negative samples (popularity-biased)
        num_negatives = min(len(user_items) * negative_ratio, len(non_interacted))
        negative_items = np.random.choice(
            non_interacted, 
            size=num_negatives, 
            replace=False,
            p=neg_probs
        )
        
        for item in negative_items:
            new_data.append([user, item, 0])
    
    result_df = pd.DataFrame(new_data, columns=["user_id", "song_id", "interaction"])
    return result_df


def create_content_based_interactions(
    num_users: int,
    num_items: int,
    normalized_audio_embeddings: torch.Tensor,
    likes_per_user: int = 30,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Create content-based interactions using audio similarity.
    
    For each user, selects a seed song and recommends similar songs.
    
    Args:
        num_users: Number of users
        num_items: Number of items
        normalized_audio_embeddings: Normalized audio embedding matrix
        likes_per_user: Number of songs to like per user
        random_seed: Random seed
        
    Returns:
        DataFrame with content-based interactions
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    content_data = []
    
    for user_id in range(num_users):
        # Random seed song
        seed_song = random.randint(0, num_items - 1)
        
        # Get similarity to all songs
        seed_vector = normalized_audio_embeddings[seed_song]
        similarities = torch.matmul(normalized_audio_embeddings, seed_vector)
        
        # Get top similar songs
        top_similar = torch.topk(similarities, k=likes_per_user + 1).indices.tolist()
        top_similar = [s for s in top_similar if s != seed_song][:likes_per_user]
        
        for song_id in top_similar:
            content_data.append([user_id, song_id, 1])
    
    df = pd.DataFrame(content_data, columns=["user_id", "song_id", "interaction"])
    logger.info(f"Created {len(df)} content-based interactions")
    return df


def train_test_split_by_user(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    user_col: str = "user_id",
    item_col: str = "song_id",
    label_col: str = "interaction",
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test sets per user.
    
    Args:
        df: DataFrame with interactions
        test_ratio: Ratio of items per user to use for testing
        user_col: User column name
        item_col: Item column name
        label_col: Label column name
        random_seed: Random seed
        
    Returns:
        Tuple of (train_df, test_df)
    """
    np.random.seed(random_seed)
    
    train_rows = []
    test_rows = []
    
    for user in df[user_col].unique():
        user_data = df[df[user_col] == user]
        items = user_data[item_col].tolist()
        labels = user_data[label_col].tolist()
        
        if len(items) < 2:
            # If user has only 1 item, put it in training
            for item, label in zip(items, labels):
                train_rows.append([user, item, label])
            continue
        
        # Shuffle indices
        indices = np.arange(len(items))
        np.random.shuffle(indices)
        
        split_idx = max(1, int(len(items) * test_ratio))
        
        for i in indices[split_idx:]:
            train_rows.append([user, items[i], labels[i]])
        
        for i in indices[:split_idx]:
            test_rows.append([user, items[i], labels[i]])
    
    train_df = pd.DataFrame(train_rows, columns=[user_col, item_col, label_col])
    test_df = pd.DataFrame(test_rows, columns=[user_col, item_col, label_col])
    
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df


def create_audio_embedding_matrix(
    num_items: int,
    embeddings_dict: Dict[str, torch.Tensor],
    song_encoder: LabelEncoder,
    embedding_dim: int = 768,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Create audio embedding matrix for items.
    
    Args:
        num_items: Number of items
        embeddings_dict: Dictionary of song_id -> embedding
        song_encoder: Label encoder for songs
        embedding_dim: Dimension of embeddings
        device: Device to create tensor on
        
    Returns:
        Audio embedding matrix [num_items, embedding_dim]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    audio_matrix = torch.zeros(num_items, embedding_dim)
    
    for idx in range(num_items):
        # Get original song ID from encoder
        original_song_id = str(song_encoder.classes_[idx]).zfill(6)
        
        if original_song_id in embeddings_dict:
            audio_matrix[idx] = embeddings_dict[original_song_id]
        else:
            # Try without zero-padding
            original_song_id_no_pad = song_encoder.classes_[idx]
            if original_song_id_no_pad in embeddings_dict:
                audio_matrix[idx] = embeddings_dict[original_song_id_no_pad]
    
    return audio_matrix.to(device)


def balance_dataset(
    df: pd.DataFrame,
    label_col: str = "interaction",
    method: str = "undersample",
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Balance the dataset by undersampling or oversampling.
    
    Args:
        df: DataFrame with labels
        label_col: Label column name
        method: 'undersample' or 'oversample'
        random_seed: Random seed
        
    Returns:
        Balanced DataFrame
    """
    np.random.seed(random_seed)
    
    pos_df = df[df[label_col] == 1]
    neg_df = df[df[label_col] == 0]
    
    if method == "undersample":
        # Undersample majority class
        if len(neg_df) > len(pos_df):
            neg_df = neg_df.sample(n=len(pos_df), random_state=random_seed)
        else:
            pos_df = pos_df.sample(n=len(neg_df), random_state=random_seed)
    elif method == "oversample":
        # Oversample minority class
        if len(pos_df) < len(neg_df):
            pos_df = pos_df.sample(n=len(neg_df), replace=True, random_state=random_seed)
        else:
            neg_df = neg_df.sample(n=len(pos_df), replace=True, random_state=random_seed)
    
    return pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1, random_state=random_seed)


def get_item_popularity(
    df: pd.DataFrame,
    item_col: str = "song_id"
) -> Dict[int, int]:
    """
    Get popularity counts for each item.
    
    Args:
        df: DataFrame with interactions
        item_col: Item column name
        
    Returns:
        Dictionary of item_id -> count
    """
    return df[item_col].value_counts().to_dict()
