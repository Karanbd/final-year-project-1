"""
Quick evaluation script to get hybrid model precision.
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config
from models.hybrid import HybridModel
from utils.data_preparation import encode_ids, create_negative_samples, train_test_split_by_user
from utils.evaluation import evaluate_model
from utils.audio_processor import normalize_embeddings

# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load audio embeddings FIRST to get the correct number of songs
print("Loading audio embeddings...")
ast_embeddings = torch.load(config.EMBEDDINGS_SAVE_PATH)
embedding_dim = 768
actual_num_songs = len(ast_embeddings)
print(f"Songs in embeddings: {actual_num_songs}")

# Create audio embedding matrix
audio_matrix = torch.zeros(actual_num_songs, embedding_dim)

for idx in range(actual_num_songs):
    song_id = str(idx).zfill(6)
    if song_id in ast_embeddings:
        audio_matrix[idx] = ast_embeddings[song_id]

audio_matrix = audio_matrix.to(device)
normalized_audio = normalize_embeddings(audio_matrix, method="l2")

# Load interactions
print("Loading interactions...")
df = pd.read_csv(config.INTERACTIONS_SAVE_PATH)

# Encode IDs
df, user_encoder, song_encoder = encode_ids(df)
num_users = df["user_id"].nunique()
# Use actual number of songs from embeddings
num_songs = actual_num_songs
print(f"Users: {num_users}, Songs: {num_songs}")

# Train/test split
train_df, test_df = train_test_split_by_user(df, test_ratio=config.TEST_RATIO)

# Add negative samples
train_df = create_negative_samples(train_df, num_songs, config.NEGATIVE_SAMPLE_RATIO)
test_df = create_negative_samples(test_df, num_songs, config.NEGATIVE_SAMPLE_RATIO)
print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

# Create Hybrid model
print("Loading Hybrid model...")
hybrid_model = HybridModel(
    num_users=num_users,
    num_items=num_songs,
    audio_embedding_dim=embedding_dim,
    user_embedding_dim=config.EMBEDDING_DIM,
    hidden_dims=config.HYBRID_HIDDEN_DIMS,
    dropout_rate=0.3
).to(device)

# Load saved weights
if os.path.exists(config.HYBRID_MODEL_PATH):
    hybrid_model.load_state_dict(torch.load(config.HYBRID_MODEL_PATH, map_location=device))
    print("Loaded trained Hybrid model")
else:
    print("No trained model found!")
    exit(1)

# Evaluate
print("Evaluating Hybrid model...")
results = evaluate_model(
    hybrid_model, 
    train_df, 
    test_df, 
    num_songs, 
    k_values=[5, 10, 20], 
    device=device,
    audio_embeddings=normalized_audio
)

# Print results
print("\n" + "="*50)
print("HYBRID MODEL EVALUATION RESULTS")
print("="*50)
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
