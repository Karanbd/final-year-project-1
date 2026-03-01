"""
Flask API for Music Recommendation System
Serves predictions from NCF model with user authentication
"""

from flask import Flask, request, jsonify, session
from flask_cors import CORS
import pandas as pd
import torch
import os
import sys
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config
from models.ncf import NCF

app = Flask(__name__)
app.secret_key = 'music-recommender-secret-key-2024'

# Enable CORS
CORS(app)

# Global variables for models and data
models = {}
embeddings_dict = None
interactions_df = None
device = None
user_encoder = None
song_encoder = None

# User database (in-memory for demo - use a real database in production)
users_db = {}

# Load user database if exists
USERS_FILE = os.path.join(os.path.dirname(__file__), 'users.json')
if os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'r') as f:
        users_db = json.load(f)


def save_users():
    with open(USERS_FILE, 'w') as f:
        json.dump(users_db, f)


def load_data():
    global embeddings_dict, interactions_df, device, user_encoder, song_encoder

    device = config.get_device()

    # Load embeddings
    if os.path.exists(config.EMBEDDINGS_SAVE_PATH):
        embeddings_dict = torch.load(config.EMBEDDINGS_SAVE_PATH, map_location=device)

    # Load interactions
    if os.path.exists(config.INTERACTIONS_SAVE_PATH):
        interactions_df = pd.read_csv(config.INTERACTIONS_SAVE_PATH)

        # Encode IDs
        user_encoder = LabelEncoder()
        song_encoder = LabelEncoder()

        interactions_df['user_id'] = user_encoder.fit_transform(interactions_df['user_id'])
        interactions_df['song_id'] = song_encoder.fit_transform(interactions_df['song_id'].astype(str))

    return device


def load_models(num_users, num_songs):
    global models, device, embeddings_dict, song_encoder

    if device is None:
        device = load_data()

    models = {}

    # Try to load Hybrid model first
    hybrid_path = config.HYBRID_MODEL_PATH
    if os.path.exists(hybrid_path) and embeddings_dict is not None and song_encoder is not None:
        try:
            from models.hybrid import HybridModel
            from utils.data_preparation import create_audio_embedding_matrix
            from utils.audio_processor import normalize_embeddings
            
            audio_matrix = create_audio_embedding_matrix(
                num_items=num_songs,
                embeddings_dict=embeddings_dict,
                song_encoder=song_encoder,
                embedding_dim=config.AUDIO_EMBEDDING_DIM,
                device=device
            )
            audio_matrix = normalize_embeddings(audio_matrix, method="l2")
            
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
            print("Hybrid model loaded successfully!")
        except Exception as e:
            print(f"Could not load Hybrid model: {e}")

    # Load NCF model as fallback
    ncf_path = config.NCF_MODEL_PATH
    if os.path.exists(ncf_path):
        try:
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
            print("NCF model loaded successfully!")
        except Exception as e:
            print(f"Could not load NCF model: {e}")

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


# Authentication Routes

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')
    email = data.get('email', '').strip()

    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400

    if username in users_db:
        return jsonify({'error': 'Username already exists'}), 400

    users_db[username] = {
        'password': password,
        'email': email,
        'created_at': str(pd.Timestamp.now())
    }
    save_users()

    return jsonify({'success': True, 'message': 'User registered successfully'})


@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if username not in users_db:
        return jsonify({'error': 'Invalid username or password'}), 401

    if users_db[username]['password'] != password:
        return jsonify({'error': 'Invalid username or password'}), 401

    session['user'] = username

    return jsonify({
        'success': True,
        'username': username,
        'message': 'Login successful'
    })


@app.route('/api/logout', methods=['POST'])
def logout():
    session.pop('user', None)
    return jsonify({'success': True, 'message': 'Logged out successfully'})


@app.route('/api/check-session', methods=['GET'])
def check_session():
    username = session.get('user')
    if username:
        return jsonify({'logged_in': True, 'username': username})
    return jsonify({'logged_in': False})


# Recommendation Routes

@app.route('/api/stats', methods=['GET'])
def get_stats():
    global interactions_df

    if interactions_df is None:
        load_data()

    if interactions_df is None:
        return jsonify({'error': 'No interaction data found. Please run training first.'}), 400

    num_users = interactions_df['user_id'].nunique()
    num_songs = interactions_df['song_id'].nunique()
    num_interactions = len(interactions_df)

    return jsonify({
        'num_users': num_users,
        'num_songs': num_songs,
        'num_interactions': num_interactions,
        'available_models': ['ncf']
    })


@app.route('/api/recommend', methods=['POST'])
def recommend():
    global models, interactions_df, device

    # Check authentication
    username = session.get('user')
    if not username:
        return jsonify({'error': 'Please login first'}), 401

    data = request.json
    user_id = int(data.get('user_id', 0))
    num_recommendations = int(data.get('k', 20))

    # Load data if needed
    if interactions_df is None:
        load_data()

    if interactions_df is None:
        return jsonify({'error': 'No interaction data found. Please run training first.'}), 400

    num_users = interactions_df['user_id'].nunique()
    num_songs = interactions_df['song_id'].nunique()

    # Load models if needed
    if not models:
        load_models(num_users, num_songs)

    # Use hybrid if available, otherwise fall back to ncf
    if 'hybrid' in models:
        model_type = 'hybrid'
        model = models['hybrid']
    elif 'ncf' in models:
        model_type = 'ncf'
        model = models['ncf']
    else:
        return jsonify({'error': 'No trained models found. Please train the model first.'}), 400

    # Get recommendations
    top_indices, top_scores = get_recommendations(model, user_id, num_songs, num_recommendations)

    recommendations = []
    for song_id, score in zip(top_indices, top_scores):
        recommendations.append({
            'song_id': int(song_id),
            'score': float(score)
        })

    return jsonify({
        'user_id': user_id,
        'model_type': model_type,
        'recommendations': recommendations,
        'username': username
    })


@app.route('/api/user-history', methods=['GET'])
def user_history():
    global interactions_df

    username = session.get('user')
    if not username:
        return jsonify({'error': 'Please login first'}), 401

    if interactions_df is None:
        load_data()

    if interactions_df is None:
        return jsonify({'error': 'No interaction data found.'}), 400

    user_id = int(request.args.get('user_id', 0))
    user_history = interactions_df[interactions_df['user_id'] == user_id]

    history = []
    for _, row in user_history.head(50).iterrows():
        history.append({
            'song_id': int(row['song_id']),
            'rating': float(row.get('rating', 1.0))
        })

    return jsonify({'history': history})


if __name__ == '__main__':
    print("Loading data...")
    load_data()
    print("Data loaded successfully!")
    print(f"Device: {device}")
    print(f"Users registered: {len(users_db)}")
    app.run(debug=True, host='0.0.0.0', port=5000)
