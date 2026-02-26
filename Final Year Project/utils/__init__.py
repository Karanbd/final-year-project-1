"""Utils module."""
from .audio_processor import (
    AudioProcessor,
    extract_embeddings_from_directory,
    normalize_embeddings,
    compute_similarity_matrix
)

from .data_preparation import (
    create_random_interactions,
    encode_ids,
    create_negative_samples,
    create_popularity_based_negative_samples,
    create_content_based_interactions,
    train_test_split_by_user,
    create_audio_embedding_matrix,
    get_item_popularity,
    balance_dataset
)

from .evaluation import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    map_at_k,
    mrr_at_k,
    hit_rate_at_k,
    get_top_k_recommendations,
    evaluate_model,
    print_evaluation_results
)

__all__ = [
    # Audio processor
    'AudioProcessor',
    'extract_embeddings_from_directory',
    'normalize_embeddings',
    'compute_similarity_matrix',
    # Data preparation
    'create_random_interactions',
    'encode_ids',
    'create_negative_samples',
    'create_popularity_based_negative_samples',
    'create_content_based_interactions',
    'train_test_split_by_user',
    'create_audio_embedding_matrix',
    'get_item_popularity',
    'balance_dataset',
    # Evaluation
    'precision_at_k',
    'recall_at_k',
    'ndcg_at_k',
    'map_at_k',
    'mrr_at_k',
    'hit_rate_at_k',
    'get_top_k_recommendations',
    'evaluate_model',
    'print_evaluation_results',
]
