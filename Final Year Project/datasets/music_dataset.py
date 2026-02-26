"""
Music Dataset Implementation.
PyTorch Dataset classes for music recommendation.
"""

import torch
from torch.utils.data import Dataset, Subset
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict


class MusicInteractionDataset(Dataset):
    """
    Dataset for user-item interactions in music recommendation.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "song_id",
        label_col: str = "interaction"
    ):
        """
        Initialize the Music Interaction Dataset.
        
        Args:
            df: DataFrame with user-item interactions
            user_col: Name of the user column
            item_col: Name of the item (song) column
            label_col: Name of the interaction label column
        """
        self.users = torch.tensor(df[user_col].values, dtype=torch.long)
        self.items = torch.tensor(df[item_col].values, dtype=torch.long)
        self.labels = torch.tensor(df[label_col].values, dtype=torch.float)
        
        self.num_users = self.users.max().item() + 1
        self.num_items = self.items.max().item() + 1
    
    def __len__(self) -> int:
        return len(self.users)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.items[idx], self.labels[idx]
    
    def get_num_users(self) -> int:
        return self.num_users
    
    def get_num_items(self) -> int:
        return self.num_items


class MusicDatasetWithAudio(Dataset):
    """
    Dataset that includes audio embeddings for items.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        audio_embeddings: torch.Tensor,
        user_col: str = "user_id",
        item_col: str = "song_id",
        label_col: str = "interaction"
    ):
        """
        Initialize the Music Dataset with Audio.
        
        Args:
            df: DataFrame with user-item interactions
            audio_embeddings: Tensor of audio embeddings [num_items, embedding_dim]
            user_col: Name of the user column
            item_col: Name of the item column
            label_col: Name of the interaction label column
        """
        self.users = torch.tensor(df[user_col].values, dtype=torch.long)
        self.items = torch.tensor(df[item_col].values, dtype=torch.long)
        self.labels = torch.tensor(df[label_col].values, dtype=torch.float)
        self.audio_embeddings = audio_embeddings
        
        self.num_users = self.users.max().item() + 1
        self.num_items = self.items.max().item() + 1
    
    def __len__(self) -> int:
        return len(self.users)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        user = self.users[idx]
        item = self.items[idx]
        label = self.labels[idx]
        audio_emb = self.audio_embeddings[item]
        
        return user, item, audio_emb, label
    
    def get_num_users(self) -> int:
        return self.num_users
    
    def get_num_items(self) -> int:
        return self.num_items


class MusicSessionDataset(Dataset):
    """
    Dataset for session-based music recommendation.
    Handles sequential listening history.
    """
    
    def __init__(
        self,
        sessions: List[List[int]],
        sequence_length: int = 10
    ):
        """
        Initialize Session-based Dataset.
        
        Args:
            sessions: List of user sessions, each session is a list of item IDs
            sequence_length: Length of input sequence
        """
        self.sequence_length = sequence_length
        self.sequences = []
        self.next_items = []
        
        for session in sessions:
            if len(session) > sequence_length:
                for i in range(len(session) - sequence_length):
                    self.sequences.append(session[i:i + sequence_length])
                    self.next_items.append(session[i + sequence_length])
        
        self.sequences = torch.tensor(self.sequences, dtype=torch.long)
        self.next_items = torch.tensor(self.next_items, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.next_items[idx]


def train_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    user_col: str = "user_id",
    item_col: str = "song_id",
    label_col: str = "interaction",
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split interactions into train and test sets per user.
    
    Args:
        df: DataFrame with interactions
        test_ratio: Ratio of items per user to use for testing
        user_col: User column name
        item_col: Item column name
        label_col: Label column name
        random_seed: Random seed for reproducibility
        
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
        
        # Shuffle indices
        indices = np.arange(len(items))
        np.random.shuffle(indices)
        
        split_idx = int(len(items) * test_ratio)
        
        for i in indices[split_idx:]:
            train_rows.append([user, items[i], labels[i]])
        
        for i in indices[:split_idx]:
            test_rows.append([user, items[i], labels[i]])
    
    train_df = pd.DataFrame(train_rows, columns=[user_col, item_col, label_col])
    test_df = pd.DataFrame(test_rows, columns=[user_col, item_col, label_col])
    
    return train_df, test_df


def create_user_item_matrix(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "song_id",
    rating_col: str = "interaction"
) -> torch.Tensor:
    """
    Create user-item interaction matrix.
    
    Args:
        df: DataFrame with interactions
        user_col: User column name
        item_col: Item column name
        rating_col: Rating/interaction column name
        
    Returns:
        User-item matrix as tensor
    """
    num_users = df[user_col].max() + 1
    num_items = df[item_col].max() + 1
    
    matrix = torch.zeros(num_users, num_items)
    
    for _, row in df.iterrows():
        matrix[row[user_col], row[item_col]] = row[rating_col]
    
    return matrix
