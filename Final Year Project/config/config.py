"""
Configuration settings for the Music Recommendation System.
"""

import os
from pathlib import Path

# ============================================
# Paths Configuration
# ============================================
# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "checkpoints"
OUTPUT_DIR = BASE_DIR / "outputs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Local paths (Windows)
LOCAL_DATA_PATH = "C:/Users/admin/Desktop"

# Audio data paths
AUDIO_BASE_PATH = os.path.join(LOCAL_DATA_PATH, "fma_small", "fma_small")
AUDIO_SAMPLE_PATH = os.path.join(LOCAL_DATA_PATH, "fma_small", "fma_small", "000", "000002.mp3")

# Model save paths
EMBEDDINGS_SAVE_PATH = os.path.join(LOCAL_DATA_PATH, "ast_embeddings (2).pt")
INTERACTIONS_SAVE_PATH = os.path.join(LOCAL_DATA_PATH, "fake_interactions.csv")
MODEL_CHECKPOINT_PATH = os.path.join(LOCAL_DATA_PATH, "model_checkpoint.pt")

# ============================================
# Model Configuration
# ============================================
# Audio Spectrogram Transformer
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
AUDIO_EMBEDDING_DIM = 768  # AST output dimension

# NCF Model
EMBEDDING_DIM = 128  # Increased from 64
NCF_HIDDEN_DIMS = [256, 128, 64]  # Increased dimensions

# Hybrid Model
HYBRID_HIDDEN_DIMS = [512, 256, 128]  # Increased dimensions

# ============================================
# Training Configuration
# ============================================
# Dataset
NUM_USERS = 500
MIN_SONGS_PER_USER = 20
MAX_SONGS_PER_USER = 50

# Hyperparameters - IMPROVED
BATCH_SIZE = 128  # Reduced for better gradient updates
NCF_EPOCHS = 30  # Increased from 10
HYBRID_EPOCHS = 15  # Increased from 5
LEARNING_RATE = 0.0005  # Reduced for more stable training
WEIGHT_DECAY = 1e-4  # Increased for better regularization
DROPOUT = 0.2  # Added dropout

# Early stopping - More patient
PATIENCE = 5  # Increased from 3
MIN_DELTA = 0.0005  # Smaller delta for more patience

# Evaluation
K = 10  # For precision@k, recall@k, etc.
TEST_RATIO = 0.2

# Negative sampling - More negatives for better learning
NEGATIVE_SAMPLE_RATIO = 5  # Increased from 3

# Learning rate scheduling
USE_SCHEDULER = True
SCHEDULER_PATIENCE = 2
SCHEDULER_FACTOR = 0.5

# ============================================
# Device Configuration
# ============================================
# Set automatically based on availability
def get_device():
    """Get the best available device."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# Random Seed
# ============================================
RANDOM_SEED = 42

# ============================================
# Logging Configuration
# ============================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
