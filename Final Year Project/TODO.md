# TODO: Music Recommendation System Improvements

## Phase 1: Refactor Code into Proper Modules ✅

- [x] 1.1 Update config/config.py with all settings from k.py
- [x] 1.2 Implement models/ncf.py with proper NCF architecture
- [x] 1.3 Implement models/hybrid.py with improved hybrid architecture
- [x] 1.4 Update datasets/music_dataset.py with proper PyTorch Dataset
- [x] 1.5 Update utils/audio_processor.py with AST embedding extraction
- [x] 1.6 Update utils/data_preparation.py with data handling functions
- [x] 1.7 Update utils/evaluation.py with PyTorch-compatible metrics + new metrics
- [x] 1.8 Update scripts/main.py as the main orchestration script

## Phase 2: Fix Bugs (Incorporated in Phase 1)

- [x] 2.1 Add validation set split during training - Added in train_model function
- [x] 2.2 Add early stopping - Added in train_model function
- [x] 2.3 Add model checkpointing - Added in train_model function
- [x] 2.4 Fix Precision@K edge cases - Added check for empty sets
- [x] 2.5 Add proper error handling - Added try-except blocks throughout
- [x] 2.6 Add data normalization - Added normalize_embeddings function

## Phase 3: Improve Models (Incorporated in Phase 1)

- [x] 3.1 Add dropout and batch normalization - Added in NCF and Hybrid models
- [x] 3.2 Add learning rate scheduler - Added ReduceLROnPlateau in train_model
- [x] 3.3 Implement better negative sampling - Added popularity-based sampling
- [x] 3.4 Add Recall@K, NDCG@K, MAP@K metrics - Added in evaluation.py
- [x] 3.5 Implement model ensemble - Added in evaluation (ready for use)
- [x] 3.6 Add cross-attention mechanism - Added AttentionHybridModel

## Completed Features:

### Models:

- NCF with GMF + MLP architecture
- HybridModel combining user + audio embeddings
- AttentionHybridModel with cross-attention
- DeepHybridModel with separate towers
- MultiVAE for collaborative filtering

### Data Processing:

- AudioSpectrogram Transformer embedding extraction
- Batch processing for audio files
- Multiple negative sampling strategies
- Train/test split per user

### Evaluation:

- Precision@K, Recall@K, NDCG@K
- MAP@K, MRR@K, Hit Rate@K
- Comprehensive evaluation framework
- Early stopping and model checkpointing

### Training:

- Learning rate scheduling
- Weight decay regularization
- Batch normalization
- Dropout for regularization
- Validation-based early stopping
