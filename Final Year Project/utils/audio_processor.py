"""
Audio Processing Utilities.
Handles audio embedding extraction using Audio Spectrogram Transformer (AST).
"""

import os
import torch
import librosa
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Audio processor for extracting embeddings from music files.
    """
    
    def __init__(
        self,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        device: Optional[torch.device] = None,
        sample_rate: int = 16000,
        max_duration: float = 10.0
    ):
        """
        Initialize the Audio Processor.
        
        Args:
            model_name: Name of the pretrained AST model
            device: Device to run the model on
            sample_rate: Audio sample rate
            max_duration: Maximum duration of audio to process (seconds)
        """
        self.model_name = model_name
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        
        self.processor = None
        self.model = None
        self.is_loaded = False
    
    def load_model(self):
        """Load the AST model and processor."""
        if self.is_loaded:
            return
        
        try:
            from transformers import AutoProcessor, ASTModel
            
            logger.info(f"Loading AST model: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = ASTModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading AST model: {e}")
            raise
    
    def load_audio(
        self,
        audio_path: str,
        offset: Optional[float] = None,
        duration: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file.
        
        Args:
            audio_path: Path to audio file
            offset: Start reading at this time (seconds)
            duration: Only load this much audio (seconds)
            
        Returns:
            Tuple of (audio waveform, sample rate)
        """
        if duration is None:
            duration = self.max_duration
        
        try:
            y, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                offset=offset,
                duration=duration,
                mono=True
            )
            return y, sr
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            raise
    
    def extract_embedding(
        self,
        audio_path: str,
        aggregate: str = "mean"
    ) -> torch.Tensor:
        """
        Extract audio embedding for a single file.
        
        Args:
            audio_path: Path to audio file
            aggregate: Aggregation method for sequence tokens ('mean', 'max', 'cls')
            
        Returns:
            Audio embedding tensor
        """
        if not self.is_loaded:
            self.load_model()
        
        # Load audio
        y, sr = self.load_audio(audio_path)
        
        # Process audio
        inputs = self.processor(y, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Aggregate tokens
        if aggregate == "mean":
            embedding = outputs.last_hidden_state.mean(dim=1)
        elif aggregate == "max":
            embedding = outputs.last_hidden_state.max(dim=1)[0]
        elif aggregate == "cls":
            embedding = outputs.last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregate}")
        
        return embedding.squeeze(0).cpu()
    
    def extract_embeddings_batch(
        self,
        audio_paths: List[str],
        batch_size: int = 16,
        aggregate: str = "mean"
    ) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings for multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            batch_size: Batch size for processing
            aggregate: Aggregation method
            
        Returns:
            Dictionary mapping file names to embeddings
        """
        if not self.is_loaded:
            self.load_model()
        
        embeddings = {}
        
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]
            batch_audios = []
            
            for path in batch_paths:
                try:
                    y, sr = self.load_audio(path)
                    batch_audios.append((y, sr))
                except Exception as e:
                    logger.warning(f"Skipping {path}: {e}")
                    continue
            
            if not batch_audios:
                continue
            
            # Process batch
            audios = [a[0] for a in batch_audios]
            sample_rates = [a[1] for a in batch_audios]
            
            inputs = self.processor(
                audios,
                sampling_rate=sample_rates[0],
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract embeddings
            if aggregate == "mean":
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            elif aggregate == "max":
                batch_embeddings = outputs.last_hidden_state.max(dim=1)[0]
            elif aggregate == "cls":
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Store embeddings
            for j, path in enumerate(batch_paths):
                song_id = os.path.splitext(os.path.basename(path))[0]
                embeddings[song_id] = batch_embeddings[j].cpu()
            
            logger.info(f"Processed {min(i + batch_size, len(audio_paths))}/{len(audio_paths)} files")
        
        return embeddings


def extract_embeddings_from_directory(
    base_path: str,
    max_songs: int = 200,
    extensions: List[str] = [".mp3", ".wav", ".flac", ".ogg"],
    save_path: Optional[str] = None,
    load_existing: bool = True,
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    Extract embeddings for all audio files in a directory.
    
    Args:
        base_path: Base directory containing audio files
        max_songs: Maximum number of songs to process
        extensions: List of audio file extensions to include
        save_path: Path to save/load embeddings
        load_existing: Whether to load existing embeddings if available
        device: Device to use
        
    Returns:
        Dictionary of song_id -> embedding
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check for existing embeddings
    if load_existing and save_path and os.path.exists(save_path):
        logger.info(f"Loading existing embeddings from {save_path}")
        embeddings = torch.load(save_path)
        logger.info(f"Loaded {len(embeddings)} embeddings")
        return embeddings
    
    # Initialize processor
    processor = AudioProcessor(device=device)
    processor.load_model()
    
    # Find audio files
    audio_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, file))
    
    audio_files = audio_files[:max_songs]
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Extract embeddings
    embeddings = processor.extract_embeddings_batch(audio_files)
    
    # Save embeddings
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(embeddings, save_path)
        logger.info(f"Saved embeddings to {save_path}")
    
    return embeddings


def normalize_embeddings(
    embeddings: torch.Tensor,
    method: str = "l2"
) -> torch.Tensor:
    """
    Normalize embeddings.
    
    Args:
        embeddings: Embedding tensor
        method: Normalization method ('l2', 'minmax', 'standard')
        
    Returns:
        Normalized embeddings
    """
    if method == "l2":
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)
    elif method == "minmax":
        min_val = embeddings.min(dim=1, keepdim=True)[0]
        max_val = embeddings.max(dim=1, keepdim=True)[0]
        return (embeddings - min_val) / (max_val - min_val + 1e-8)
    elif method == "standard":
        mean = embeddings.mean(dim=1, keepdim=True)
        std = embeddings.std(dim=1, keepdim=True)
        return (embeddings - mean) / (std + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_similarity_matrix(
    embeddings: torch.Tensor,
    method: str = "cosine"
) -> torch.Tensor:
    """
    Compute pairwise similarity matrix for embeddings.
    
    Args:
        embeddings: Embedding tensor [num_items, embedding_dim]
        method: Similarity method ('cosine', 'dot')
        
    Returns:
        Similarity matrix [num_items, num_items]
    """
    if method == "cosine":
        # Normalize embeddings
        normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return torch.matmul(normalized, normalized.T)
    elif method == "dot":
        return torch.matmul(embeddings, embeddings.T)
    else:
        raise ValueError(f"Unknown similarity method: {method}")
