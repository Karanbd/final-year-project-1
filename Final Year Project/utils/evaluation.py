"""
Evaluation Metrics for Recommendation Systems.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_top_k_recommendations(
    model,
    user_id: int,
    num_items: int,
    k: int = 10,
    device: Optional[torch.device] = None,
    exclude_items: Optional[List[int]] = None,
    audio_embeddings: Optional[torch.Tensor] = None
) -> List[int]:
    """
    Get top-k song recommendations for a user.
    
    Args:
        model: Trained model (NCF or Hybrid)
        user_id: User ID
        num_items: Total number of items
        k: Number of recommendations
        device: Device to run on
        exclude_items: Items to exclude from recommendations
        audio_embeddings: Audio embedding matrix (for older Hybrid models)
        
    Returns:
        List of top-k item IDs
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    with torch.no_grad():
        # Create user and item tensors
        user_tensor = torch.tensor([user_id] * num_items, device=device)
        item_tensor = torch.arange(num_items, device=device)
        
        # Get predictions using item IDs (works for both NCF and Hybrid)
        scores = model(user_tensor, item_tensor)
        scores = scores.squeeze().cpu()
        
        # Exclude items if specified
        if exclude_items is not None and len(exclude_items) > 0:
            mask = torch.ones(num_items, dtype=torch.bool)
            mask[exclude_items] = False
            scores = scores * mask.float()
        
        # Get top-k
        top_k = torch.topk(scores, k=min(k, num_items)).indices.tolist()
    
    return top_k


def precision_at_k(
    model,
    test_df,
    num_items: int,
    k: int = 10,
    device: Optional[torch.device] = None,
    audio_embeddings: Optional[torch.Tensor] = None
) -> float:
    """
    Calculate Precision@K.
    
    Args:
        model: Trained recommendation model
        test_df: Test DataFrame with user-item interactions
        num_items: Total number of items
        k: Number of top items to consider
        device: Device to run on
        audio_embeddings: Audio embedding matrix (for Hybrid models)
        
    Returns:
        Precision@K score
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    precision_scores = []
    
    with torch.no_grad():
        for user in test_df["user_id"].unique():
            # Get top-k recommendations
            top_k = get_top_k_recommendations(
                model, user, num_items, k, device, 
                exclude_items=None, audio_embeddings=audio_embeddings
            )
            
            # Get ground truth
            true_items = set(test_df[test_df["user_id"] == user]["song_id"])
            
            if len(true_items) == 0:
                continue
            
            # Calculate hits
            hits = len(set(top_k) & true_items)
            precision_scores.append(hits / k)
    
    return np.mean(precision_scores) if precision_scores else 0.0


def recall_at_k(
    model,
    test_df,
    num_items: int,
    k: int = 10,
    device: Optional[torch.device] = None,
    audio_embeddings: Optional[torch.Tensor] = None
) -> float:
    """
    Calculate Recall@K.
    
    Args:
        model: Trained recommendation model
        test_df: Test DataFrame with user-item interactions
        num_items: Total number of items
        k: Number of top items to consider
        device: Device to run on
        audio_embeddings: Audio embedding matrix (for Hybrid models)
        
    Returns:
        Recall@K score
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    recall_scores = []
    
    with torch.no_grad():
        for user in test_df["user_id"].unique():
            # Get top-k recommendations
            top_k = get_top_k_recommendations(
                model, user, num_items, k, device,
                audio_embeddings=audio_embeddings
            )
            
            # Get ground truth
            true_items = set(test_df[test_df["user_id"] == user]["song_id"])
            
            if len(true_items) == 0:
                continue
            
            # Calculate hits
            hits = len(set(top_k) & true_items)
            recall_scores.append(hits / len(true_items))
    
    return np.mean(recall_scores) if recall_scores else 0.0


def ndcg_at_k(
    model,
    test_df,
    num_items: int,
    k: int = 10,
    device: Optional[torch.device] = None,
    audio_embeddings: Optional[torch.Tensor] = None
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG)@K.
    
    Args:
        model: Trained recommendation model
        test_df: Test DataFrame with user-item interactions
        num_items: Total number of items
        k: Number of top items to consider
        device: Device to run on
        audio_embeddings: Audio embedding matrix (for Hybrid models)
        
    Returns:
        NDCG@K score
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    ndcg_scores = []
    
    with torch.no_grad():
        for user in test_df["user_id"].unique():
            # Get top-k recommendations
            top_k = get_top_k_recommendations(
                model, user, num_items, k, device,
                audio_embeddings=audio_embeddings
            )
            
            # Get ground truth
            true_items = set(test_df[test_df["user_id"] == user]["song_id"])
            
            if len(true_items) == 0:
                continue
            
            # Calculate DCG
            dcg = 0.0
            for i, item in enumerate(top_k):
                if item in true_items:
                    dcg += 1.0 / np.log2(i + 2)  # i+2 because i is 0-indexed
            
            # Calculate IDCG
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), k)))
            
            # Calculate NDCG
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def map_at_k(
    model,
    test_df,
    num_items: int,
    k: int = 10,
    device: Optional[torch.device] = None,
    audio_embeddings: Optional[torch.Tensor] = None
) -> float:
    """
    Calculate Mean Average Precision (MAP)@K.
    
    Args:
        model: Trained recommendation model
        test_df: Test DataFrame with user-item interactions
        num_items: Total number of items
        k: Number of top items to consider
        device: Device to run on
        audio_embeddings: Audio embedding matrix (for Hybrid models)
        
    Returns:
        MAP@K score
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    ap_scores = []
    
    with torch.no_grad():
        for user in test_df["user_id"].unique():
            # Get top-k recommendations
            top_k = get_top_k_recommendations(
                model, user, num_items, k, device,
                audio_embeddings=audio_embeddings
            )
            
            # Get ground truth
            true_items = set(test_df[test_df["user_id"] == user]["song_id"])
            
            if len(true_items) == 0:
                continue
            
            # Calculate Average Precision
            hits = 0
            sum_precision = 0.0
            
            for i, item in enumerate(top_k):
                if item in true_items:
                    hits += 1
                    precision_at_i = hits / (i + 1)
                    sum_precision += precision_at_i
            
            # AP = sum of precisions at hits / min(k, |relevant items|)
            ap = sum_precision / min(len(true_items), k)
            ap_scores.append(ap)
    
    return np.mean(ap_scores) if ap_scores else 0.0


def mrr_at_k(
    model,
    test_df,
    num_items: int,
    k: int = 10,
    device: Optional[torch.device] = None,
    audio_embeddings: Optional[torch.Tensor] = None
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR)@K.
    
    Args:
        model: Trained recommendation model
        test_df: Test DataFrame with user-item interactions
        num_items: Total number of items
        k: Number of top items to consider
        device: Device to run on
        audio_embeddings: Audio embedding matrix (for Hybrid models)
        
    Returns:
        MRR@K score
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    rr_scores = []
    
    with torch.no_grad():
        for user in test_df["user_id"].unique():
            # Get top-k recommendations
            top_k = get_top_k_recommendations(
                model, user, num_items, k, device,
                audio_embeddings=audio_embeddings
            )
            
            # Get ground truth
            true_items = set(test_df[test_df["user_id"] == user]["song_id"])
            
            if len(true_items) == 0:
                continue
            
            # Find rank of first relevant item
            for i, item in enumerate(top_k):
                if item in true_items:
                    rr_scores.append(1.0 / (i + 1))
                    break
            else:
                rr_scores.append(0.0)
    
    return np.mean(rr_scores) if rr_scores else 0.0


def hit_rate_at_k(
    model,
    test_df,
    num_items: int,
    k: int = 10,
    device: Optional[torch.device] = None,
    audio_embeddings: Optional[torch.Tensor] = None
) -> float:
    """
    Calculate Hit Rate@K (proportion of users with at least one hit).
    
    Args:
        model: Trained recommendation model
        test_df: Test DataFrame with user-item interactions
        num_items: Total number of items
        k: Number of top items to consider
        device: Device to run on
        audio_embeddings: Audio embedding matrix (for Hybrid models)
        
    Returns:
        Hit Rate@K score
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    hits = 0
    total_users = 0
    
    with torch.no_grad():
        for user in test_df["user_id"].unique():
            # Get top-k recommendations
            top_k = get_top_k_recommendations(
                model, user, num_items, k, device,
                audio_embeddings=audio_embeddings
            )
            
            # Get ground truth
            true_items = set(test_df[test_df["user_id"] == user]["song_id"])
            
            if len(true_items) == 0:
                continue
            
            total_users += 1
            
            if len(set(top_k) & true_items) > 0:
                hits += 1
    
    return hits / total_users if total_users > 0 else 0.0


def evaluate_model(
    model,
    train_df,
    test_df,
    num_items: int,
    k_values: List[int] = [5, 10, 20],
    device: Optional[torch.device] = None,
    audio_embeddings: Optional[torch.Tensor] = None
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate model with multiple metrics at multiple k values.
    
    Args:
        model: Trained model
        train_df: Training DataFrame
        test_df: Test DataFrame
        num_items: Total number of items
        k_values: List of k values to evaluate
        device: Device to run on
        audio_embeddings: Audio embedding matrix (for Hybrid models)
        
    Returns:
        Dictionary of metric_name -> {k: score}
    """
    if device is None:
        device = next(model.parameters()).device
    
    results = {
        "precision": {},
        "recall": {},
        "ndcg": {},
        "map": {},
        "mrr": {},
        "hit_rate": {}
    }
    
    for k in k_values:
        logger.info(f"Evaluating at k={k}")
        
        results["precision"][k] = precision_at_k(
            model, test_df, num_items, k, device, audio_embeddings
        )
        results["recall"][k] = recall_at_k(
            model, test_df, num_items, k, device, audio_embeddings
        )
        results["ndcg"][k] = ndcg_at_k(
            model, test_df, num_items, k, device, audio_embeddings
        )
        results["map"][k] = map_at_k(
            model, test_df, num_items, k, device, audio_embeddings
        )
        results["mrr"][k] = mrr_at_k(
            model, test_df, num_items, k, device, audio_embeddings
        )
        results["hit_rate"][k] = hit_rate_at_k(
            model, test_df, num_items, k, device, audio_embeddings
        )
    
    return results


def print_evaluation_results(results: Dict[str, Dict[int, float]]):
    """
    Pretty print evaluation results.
    
    Args:
        results: Results dictionary from evaluate_model
    """
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    
    k_values = list(next(iter(results.values())).keys())
    
    # Header
    print(f"{'Metric':<15}", end="")
    for k in k_values:
        print(f"@{k:>8}", end="")
    print()
    print("-" * 70)
    
    # Rows
    for metric, scores in results.items():
        print(f"{metric:<15}", end="")
        for k in k_values:
            print(f"{scores[k]:>8.4f}", end="")
        print()
    
    print("=" * 70)


# Legacy functions for compatibility
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy."""
    return np.mean(y_true == y_pred)


def precision_legacy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate precision (legacy)."""
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    if true_positive + false_positive == 0:
        return 0.0
    return true_positive / (true_positive + false_positive)


def recall_legacy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate recall (legacy)."""
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    if true_positive + false_negative == 0:
        return 0.0
    return true_positive / (true_positive + false_negative)


def f1_score_legacy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate F1 score (legacy)."""
    prec = precision_legacy(y_true, y_pred)
    rec = recall_legacy(y_true, y_pred)
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)
