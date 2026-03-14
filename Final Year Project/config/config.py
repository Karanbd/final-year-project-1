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
EMBEDDINGS_SAVE_PATH = os.path.join(LOCAL_DATA_PATH, "ast_embeddings (3).pt")
INTERACTIONS_SAVE_PATH = os.path.join(LOCAL_DATA_PATH, "fake_interactions.csv")
MODEL_CHECKPOINT_PATH = os.path.join(BASE_DIR, "model_checkpoint.pt")
NCF_MODEL_PATH = os.path.join(BASE_DIR, "ncf_model.pt")
HYBRID_MODEL_PATH = os.path.join(BASE_DIR, "hybrid_model.pt")

# ============================================
# Model Configuration
# ============================================
# Audio Spectrogram Transformer
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
AUDIO_EMBEDDING_DIM = 768  # AST output dimension

# NCF Model - IMPROVED for better precision
EMBEDDING_DIM = 512  # Increased for richer representations
NCF_HIDDEN_DIMS = [512, 256, 128]  # Deeper network

# Hybrid Model - IMPROVED
HYBRID_HIDDEN_DIMS = [512, 256, 128]  # Deeper network
ATTENTION_HEADS = 4  # Number of attention heads

# ============================================
# Training Configuration - IMPROVED
# ============================================
# Dataset - More interactions for better learning
NUM_USERS = 1000
MIN_SONGS_PER_USER = 40  # Balanced for clustering
MAX_SONGS_PER_USER = 100  # Increased from 50

# Hyperparameters - OPTIMIZED for precision
BATCH_SIZE = 512  # Larger batch for stable gradients
NCF_EPOCHS = 50  # More epochs for precision boost
HYBRID_EPOCHS = 30  # More epochs for precision boost
LEARNING_RATE = 0.0002  # Lower LR for precision
WEIGHT_DECAY = 1e-4  # Regularization
DROPOUT = 0.2  # Dropout for regularization

# Class balancing
USE_CLASS_WEIGHTS = True
POSITIVE_WEIGHT = 3.0  # Weight positive samples more

# Label smoothing - Helps prevent overconfidence
LABEL_SMOOTHING = 0.1

# Early stopping - More patient for deep learning
PATIENCE = 10  # More patience for convergence
MIN_DELTA = 0.001  # Minimum improvement threshold

# Evaluation
K = 10  # For precision@k, recall@k, etc.
TEST_RATIO = 0.2

# Negative sampling - More negatives for better learning
NEGATIVE_SAMPLE_RATIO = 8  # Balanced for training speed

# Learning rate scheduling
USE_SCHEDULER = True
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5
WARMUP_EPOCHS = 3  # Learning rate warmup

# Gradient clipping
GRADIENT_CLIP = 1.0

# Use popularity-based negative sampling
USE_POPULARITY_NEGATIVE = True

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
