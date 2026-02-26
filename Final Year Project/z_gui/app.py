"""
Music Recommendation GUI
Displays recommended songs in a playlist format
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config
from models.ncf import NCF
from models.hybrid import HybridModel
from utils.audio_processor import AudioProcessor, normalize_embeddings
from utils.data_preparation import create_audio_embedding_matrix
from utils.evaluation import get_top_k_recommendations


# Page configuration
st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎵",
    layout="wide"
)


@st.cache_data
def load_embeddings():
    """Load pre-computed audio embeddings."""
    if os.path.exists(config.EMBEDDINGS_SAVE_PATH):
        return torch.load(config.EMBEDDINGS_SAVE_PATH)
    return None


@st.cache_data
def load_interactions():
    """Load user-song interactions."""
    if os.path.exists(config.INTERACTIONS_SAVE_PATH):
        return pd.read_csv(config.INTERACTIONS_SAVE_PATH)
    return None


def load_trained_models(num_users, num_songs, device):
    """Load trained NCF and Hybrid models."""
    models = {}
    
    ncf_path = config.MODEL_CHECKPOINT_PATH.replace(".pt", "_ncf.pt")
    if os.path.exists(ncf_path):
        ncf_model = NCF(
            num_users=num_users,
            num_items=num_songs,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dims=config.NCF_HIDDEN_DIMS,
            dropout_rate=config.DROPOUT
        ).to(device)
        ncf_model.load_state_dict(torch.load(ncf_path, map_location=device))
        ncf_model.eval()
        models['ncf'] = ncf_model
    
    hybrid_path = config.MODEL_CHECKPOINT_PATH.replace(".pt", "_hybrid.pt")
    if os.path.exists(hybrid_path):
        # Load audio embeddings for hybrid model
        embeddings_dict = load_embeddings()
        if embeddings_dict:
            audio_matrix = create_audio_embedding_matrix(embeddings_dict, num_songs)
            audio_matrix = normalize_embeddings(audio_matrix, method="l2").to(device)
            
            hybrid_model = HybridModel(
                num_users=num_users,
                num_items=num_songs,
                audio_embedding_dim=config.AUDIO_EMBEDDING_DIM,
                user_embedding_dim=config.EMBEDDING_DIM,
                hidden_dims=config.HYBRID_HIDDEN_DIMS,
                dropout_rate=config.DROPOUT
            ).to(device)
            hybrid_model.load_state_dict(torch.load(hybrid_path, map_location=device))
            hybrid_model.eval()
            models['hybrid'] = hybrid_model
    
    return models


def get_recommendations(model, user_id, num_songs, audio_embeddings=None, k=20):
    """Get top-k song recommendations for a user."""
    device = next(model.parameters()).device
    
    # Get all song IDs
    all_song_ids = torch.arange(num_songs).to(device)
    user_tensor = torch.tensor([user_id] * num_songs).to(device)
    
    with torch.no_grad():
        scores = model(user_tensor, all_song_ids)
    
    # Get top-k indices
    top_k_indices = torch.topk(scores, k).indices.cpu().numpy()
    top_k_scores = torch.topk(scores, k).values.cpu().numpy()
    
    return top_k_indices, top_k_scores


def create_playlist_card(song_id, score, rank, audio_path=None):
    """Create a styled card for a song in the playlist."""
    
    # Try to extract track info from song ID
    track_number = str(song_id).zfill(6)
    
    col1, col2, col3 = st.columns([1, 8, 2])
    
    with col1:
        st.markdown(f"**#{rank}**")
    
    with col2:
        st.markdown(f"**Track ID: {track_number}**")
        if audio_path and os.path.exists(audio_path):
            st.audio(audio_path, format='audio/mp3')
    
    with col3:
        st.progress(float(score))
        st.caption(f"{score:.2%}")
    
    st.divider()


def main():
    """Main GUI function."""
    
    # Header
    st.title("🎵 Music Recommendation System")
    st.markdown("### Personalized Playlist Generator")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Load data
    device = config.get_device()
    st.sidebar.info(f"Device: {device}")
    
    # Load embeddings and interactions
    embeddings_dict = load_embeddings()
    interactions_df = load_interactions()
    
    if embeddings_dict is None:
        st.warning("⚠️ No audio embeddings found. Please run the training first.")
        st.code("python scripts/main.py")
        return
    
    if interactions_df is None:
        st.warning("⚠️ No interaction data found. Please run the training first.")
        return
    
    # Get unique users and songs
    num_users = interactions_df['user_id'].nunique()
    num_songs = interactions_df['song_id'].nunique()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Dataset Stats:**")
    st.sidebar.markdown(f"- Users: {num_users}")
    st.sidebar.markdown(f"- Songs: {num_songs}")
    st.sidebar.markdown(f"- Interactions: {len(interactions_df)}")
    
    # Load models
    models = load_trained_models(num_users, num_songs, device)
    
    if not models:
        st.warning("⚠️ No trained models found. Please run the training first.")
        st.code("python scripts/main.py")
        return
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Available Models:**")
    for model_name in models.keys():
        st.sidebar.success(f"✓ {model_name.upper()}")
    
    # Main content
    st.markdown("## 👤 User Selection")
    
    # User selection
    user_id = st.selectbox(
        "Select a user ID:",
        range(num_users),
        format_func=lambda x: f"User {x}"
    )
    
    # Model selection
    st.markdown("## 🔧 Model Selection")
    model_type = st.radio(
        "Select recommendation model:",
        options=list(models.keys()),
        format_func=lambda x: x.upper(),
        horizontal=True
    )
    
    # Number of recommendations
    num_recommendations = st.slider(
        "Number of songs to recommend:",
        min_value=5,
        max_value=50,
        value=20
    )
    
    # Get recommendations button
    if st.button("🎯 Get Recommendations", type="primary"):
        with st.spinner("Generating personalized playlist..."):
            # Get recommendations
            model = models[model_type]
            top_indices, top_scores = get_recommendations(
                model, 
                user_id, 
                num_songs,
                k=num_recommendations
            )
            
            # Display results
            st.markdown("## 📋 Your Personalized Playlist")
            st.markdown(f"**Recommended for User {user_id} using {model_type.upper()}**")
            
            # Create playlist container
            playlist_container = st.container()
            
            with playlist_container:
                for rank, (song_id, score) in enumerate(zip(top_indices, top_scores), 1):
                    # Try to find audio file
                    audio_path = None
                    track_folder = str(song_id).zfill(3)[:3]
                    potential_path = os.path.join(config.AUDIO_BASE_PATH, track_folder, f"{str(song_id).zfill(6)}.mp3")
                    
                    if os.path.exists(potential_path):
                        audio_path = potential_path
                    
                    create_playlist_card(song_id, score, rank, audio_path)
            
            # Show user history
            st.markdown("## 📜 User Listening History")
            user_history = interactions_df[interactions_df['user_id'] == user_id]
            
            if len(user_history) > 0:
                st.dataframe(
                    user_history[['song_id', 'rating']].head(20),
                    use_container_width=True
                )
            
            # Download playlist
            playlist_df = pd.DataFrame({
                'Rank': range(1, num_recommendations + 1),
                'Song ID': top_indices,
                'Score': top_scores
            })
            
            csv = playlist_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Playlist (CSV)",
                data=csv,
                file_name=f"playlist_user_{user_id}.csv",
                mime="text/csv"
            )
    
    # About section
    st.markdown("---")
    st.markdown("## ℹ️ About")
    st.markdown("""
    This music recommendation system uses:
    - **NCF (Neural Collaborative Filtering)**: Deep learning-based collaborative filtering
    - **Hybrid Model**: Combines collaborative filtering with audio content features
    
    The model is trained on user-song interactions and audio embeddings extracted using 
    Audio Spectrogram Transformer (AST).
    """)


if __name__ == "__main__":
    main()
