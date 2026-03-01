"""
Quick evaluation script for NCF model.
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config
from models.ncf import NCF
from utils.data_preparation import encode_ids, create_negative_samples, train_test_split_by_user
from utils.evaluation import evaluate_model

# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load interactions
print("Loading interactions...")
df = pd.read_csv(config.INTERACTIONS_SAVE_PATH)

# Encode IDs
df, user_encoder, song_encoder = encode_ids(df)
num_users = df["user_id"].nunique()
num_songs = df["song_id"].nunique()
print(f"Users: {num_users}, Songs: {num_songs}")

# Train/test split
train_df, test_df = train_test_split_by_user(df, test_ratio=config.TEST_RATIO)

# Add negative samples
train_df = create_negative_samples(train_df, num_songs, config.NEGATIVE_SAMPLE_RATIO)
test_df = create_negative_samples(test_df, num_songs, config.NEGATIVE_SAMPLE_RATIO)
print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

# Create NCF model
print("Loading NCF model...")
ncf_model = NCF(
    num_users=num_users,
    num_items=num_songs,
    embedding_dim=config.EMBEDDING_DIM,
    hidden_dims=config.NCF_HIDDEN_DIMS,
    dropout_rate=0.2
).to(device)

# Load saved weights
ncf_path = "C:/Users/admin/Desktop/Music/final-year-project/Final Year Project/ncf_model.pt"
if os.path.exists(ncf_path):
    ncf_model.load_state_dict(torch.load(ncf_path, map_location=device))
    print("Loaded trained NCF model")
else:
    print("No trained NCF model found!")
    exit(1)

# Evaluate
print("Evaluating NCF model...")
results = evaluate_model(
    ncf_model, 
    train_df, 
    test_df, 
    num_songs, 
    k_values=[5, 10, 20], 
    device=device
)

# Print results
print("\n" + "="*50)
print("NCF MODEL EVALUATION RESULTS")
print("="*50)
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
