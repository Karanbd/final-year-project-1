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

# Drive path for Google Drive (for Colab)
DRIVE_PATH = os.environ.get("DRIVE_PATH", "/content/drive/MyDrive")

# Audio data paths
AUDIO_BASE_PATH = os.path.join(DRIVE_PATH, "fma_small", "fma_small")
AUDIO_SAMPLE_PATH = os.path.join(DRIVE_PATH, "fma_small", "fma_small", "000", "000002.mp3")

# Model save paths
EMBEDDINGS_SAVE_PATH = os.path.join(DRIVE_PATH, "2nd_Project_Music", "ast_embeddings.pt")
INTERACTIONS_SAVE_PATH = os.path.join(DRIVE_PATH, "2nd_Project_Music", "fake_interactions.csv")
MODEL_CHECKPOINT_PATH = os.path.join(DRIVE_PATH, "2nd_Project_Music", "model_checkpoint.pt")

# ============================================
# Model Configuration
# ============================================
# Audio Spectrogram Transformer
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
AUDIO_EMBEDDING_DIM = 768  # AST output dimension

# NCF Model
EMBEDDING_DIM = 64
NCF_HIDDEN_DIMS = [128, 64, 32]

# Hybrid Model
HYBRID_HIDDEN_DIMS = [256, 128, 64]

# ============================================
# Training Configuration
# ============================================
# Dataset
NUM_USERS = 500
MIN_SONGS_PER_USER = 20
MAX_SONGS_PER_USER = 50

# Hyperparameters
BATCH_SIZE = 256
NCF_EPOCHS = 10
HYBRID_EPOCHS = 5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

# Early stopping
PATIENCE = 3
MIN_DELTA = 0.001

# Evaluation
K = 10  # For precision@k, recall@k, etc.
TEST_RATIO = 0.2

# Negative sampling
NEGATIVE_SAMPLE_RATIO = 3  # 3x negative samples per positive

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
