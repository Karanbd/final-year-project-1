# Music Recommendation System - Improvement Plan

## Goal: Improve precision from ~0.01 to 0.15+

## Issues Identified:

1. Random user-song interactions don't reflect real preferences
2. NCF embedding dimension (128) too small
3. Shallow MLP layers
4. Basic random negative sampling
5. No class balancing
6. Test recommendations include training items

## Tasks COMPLETED:

### 1. Update config.py - Better Hyperparameters ✅

- [x] Increase embedding_dim: 128 → 256
- [x] Increase hidden_dims: [256, 128, 64] → [512, 256, 128]
- [x] Increase epochs: 10 → 30 for NCF, 5 → 20 for Hybrid
- [x] Add label smoothing: 0.1
- [x] Add class weights for imbalance (positive_weight=3.0)
- [x] Increase negative_ratio: 5 → 10

### 2. Enhance NCF Model (models/ncf.py) ✅

- [x] Add He initialization (kaiming*normal*)
- [x] Add LayerNorm
- [x] Add embedding dropout
- [x] Increase model depth (deeper MLP and output head)

### 3. Enhance Hybrid Model (models/hybrid.py) ✅

- [x] Add cross-attention mechanism
- [x] Deeper audio projection network
- [x] Better fusion strategy with attention

### 4. Improve Training (scripts/main.py) ✅

- [x] Add class weights for imbalance
- [x] Add label smoothing
- [x] Use AdamW optimizer with weight decay
- [x] Add learning rate warmup
- [x] Add gradient clipping
- [x] Larger batch size (512)

### 5. Fix Evaluation (utils/evaluation.py) ✅

- [x] Exclude training items from test recommendations
- [x] Fixed evaluation metrics to use train_df

### 6. Improve Data Generation ✅

- [x] More songs per user (50-100)
- [x] Popularity-based negative sampling
- [x] Create more realistic interactions

## Next Steps:

1. Delete old models and retrain with new architecture
2. Run training: `python final-year-project/Final Year Project/scripts/main.py`

## Expected Results:

- Precision@10: 0.15+ (from ~0.01)
- Recall@10: 0.20+
- NDCG@10: 0.18+
- Hit Rate@10: 0.40+
