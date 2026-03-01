"""
Flask API for Music Recommendation System
"""

from flask import Flask, request, jsonify
import pandas as pd
import torch
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config
from models.ncf import NCF

app = Flask(__name__)

# Global variables
models = {}
embeddings_dict = None
interactions_df = None
device = None


def load_data():
    global embeddings_dict, interactions_df, device
    device = config.get_device()
    
    if os.path.exists(config.EMBEDDINGS_SAVE_PATH):
        embeddings_dict = torch.load(config.EMBEDDINGS_SAVE_PATH, map_location=device)
    
    if os.path.exists(config.INTERACTIONS_SAVE_PATH):
        interactions_df = pd.read_csv(config.INTERACTIONS_SAVE_PATH)
    
    return device


def load_models(num_users, num_songs):
    global models, device
    if device is None:
        device = load_data()
    
    models = {}
    
    # Load NCF model
    ncf_path = config.NCF_MODEL_PATH
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
        print("NCF model loaded!")
    
    return models


def get_recommendations(model, user_id, num_songs, k=20):
    device = next(model.parameters()).device
    all_song_ids = torch.arange(num_songs).to(device)
    user_tensor = torch.tensor([user_id] * num_songs).to(device)
    
    with torch.no_grad():
        scores = model(user_tensor, all_song_ids)
    
    top_k_indices = torch.topk(scores, k).indices.cpu().numpy()
    top_k_scores = torch.topk(scores, k).values.cpu().numpy()
    
    return top_k_indices.tolist(), top_k_scores.tolist()


@app.route('/')
def index():
    from flask import send_from_directory
    return send_from_directory('.', 'index.html')


@app.route('/api/stats', methods=['GET'])
def get_stats():
    global interactions_df
    if interactions_df is None:
        load_data()
    
    if interactions_df is None:
        return jsonify({'error': 'No data found. Run training first.'}), 400
    
    num_users = interactions_df['user_id'].nunique()
    num_songs = interactions_df['song_id'].nunique()
    num_interactions = len(interactions_df)
    
    return jsonify({
        'num_users': num_users,
        'num_songs': num_songs,
        'num_interactions': num_interactions,
        'available_models': list(models.keys()) if models else ['ncf']
    })


@app.route('/api/recommend', methods=['POST'])
def recommend():
    global models, interactions_df, device
    
    data = request.json
    user_id = int(data.get('user_id', 0))
    model_type = data.get('model_type', 'ncf')
    k = int(data.get('k', 20))
    
    if interactions_df is None:
        load_data()
    
    if interactions_df is None:
        return jsonify({'error': 'No data found'}), 400
    
    num_users = interactions_df['user_id'].nunique()
    num_songs = interactions_df['song_id'].nunique()
    
    if not models:
        load_models(num_users, num_songs)
    
    if model_type not in models:
        model_type = 'ncf' if 'ncf' in models else list(models.keys())[0]
    
    model = models[model_type]
    top_indices, top_scores = get_recommendations(model, user_id, num_songs, k)
    
    recommendations = []
    for song_id, score in zip(top_indices, top_scores):
        recommendations.append({'song_id': int(song_id), 'score': float(score)})
    
    return jsonify({
        'user_id': user_id,
        'model_type': model_type,
        'recommendations': recommendations
    })


@app.route('/api/user-history', methods=['GET'])
def user_history():
    global interactions_df
    if interactions_df is None:
        load_data()
    
    user_id = int(request.args.get('user_id', 0))
    user_data = interactions_df[interactions_df['user_id'] == user_id]
    
    history = []
    for _, row in user_data.head(50).iterrows():
        history.append({'song_id': int(row['song_id']), 'rating': float(row.get('rating', 1.0))})
    
    return jsonify({'history': history})


if __name__ == '__main__':
    print("Loading data...")
    load_data()
    print("Ready!")
    app.run(debug=True, host='0.0.0.0', port=5000)
