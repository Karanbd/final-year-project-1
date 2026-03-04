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

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# YouTube API
from googleapiclient.discovery import build

# YouTube API Configuration
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY', '')

def get_youtube_client():
    """Initialize and return YouTube API client."""
    if not YOUTUBE_API_KEY:
        print("Warning: YOUTUBE_API_KEY not found in environment variables")
        return None
    try:
        return build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    except Exception as e:
        print(f"Error building YouTube client: {e}")
        return None


def search_youtube_video(song_name, artist=None, max_results=5):
    """
    Search for music videos on YouTube.
    Returns list of video details.
    """
    youtube = get_youtube_client()
    if not youtube:
        return None

    # Build search query
    if artist:
        search_query = f"{song_name} {artist} official audio"
    else:
        search_query = f"{song_name} official audio"

    try:
        request = youtube.search().list(
            q=search_query,
            part="snippet",
            type="video",
            videoCategoryId="10",  # Category 10 is Music
            maxResults=max_results
        )
        response = request.execute()

        results = []
        if response.get('items'):
            for item in response['items']:
                results.append({
                    'video_id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'thumbnail': item['snippet']['thumbnails']['high']['url'],
                    'channel': item['snippet']['channelTitle']
                })
        return results if results else None
    except Exception as e:
        print(f"YouTube search error: {e}")
        return None


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
    global interactions_df, models

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
        'available_models': list(models.keys()) if models else ['ncf']
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


# YouTube Search Route
@app.route('/api/search-youtube', methods=['POST'])
def search_youtube():
    """Search for songs on YouTube and return video details."""
    data = request.json
    song_name = data.get('song_name', '')
    artist = data.get('artist', None)
    
    if not song_name:
        return jsonify({'error': 'Song name is required'}), 400
    
    results = search_youtube_video(song_name, artist, max_results=10)
    
    if results:
        return jsonify({'results': results})
    else:
        return jsonify({'error': 'No videos found or YouTube API not configured'}), 404


@app.route('/api/content-based-recommend', methods=['POST'])
def content_based_recommend():
    """
    Content-based recommendation: Search for a song and get similar songs
    using the hybrid model's audio embeddings. Also fetches YouTube videos.
    """
    global models, embeddings_dict, interactions_df, device, song_encoder
    
    data = request.json
    song_name = data.get('song_name', '')
    artist = data.get('artist', None)
    k = data.get('k', 20)  # Number of recommendations
    
    if not song_name:
        return jsonify({'error': 'Song name is required'}), 400
    
    # Load data if needed
    if interactions_df is None:
        load_data()
    
    if embeddings_dict is None:
        return jsonify({'error': 'No audio embeddings found. Please run training first.'}), 400
    
    num_songs = interactions_df['song_id'].nunique()
    
    # Load hybrid model if needed
    if 'hybrid' not in models:
        load_models(interactions_df['user_id'].nunique(), num_songs)
    
    if 'hybrid' not in models:
        return jsonify({'error': 'Hybrid model not loaded. Please train the model first.'}), 400
    
    try:
        from utils.data_preparation import create_audio_embedding_matrix
        from utils.audio_processor import normalize_embeddings
        import torch.nn.functional as F
        
        # Create audio embedding matrix for all songs
        audio_matrix = create_audio_embedding_matrix(
            num_items=num_songs,
            embeddings_dict=embeddings_dict,
            song_encoder=song_encoder,
            embedding_dim=config.AUDIO_EMBEDDING_DIM,
            device=device
        )
        
        # Normalize embeddings for cosine similarity
        audio_matrix_normalized = normalize_embeddings(audio_matrix, method="l2")
        
        # Get the hybrid model's audio embedding layer
        hybrid_model = models['hybrid']
        audio_embeddings = hybrid_model.item_embedding.weight.data
        
        # Project audio embeddings using the model's projection layer
        with torch.no_grad():
            projected_embeddings = hybrid_model.audio_projection(audio_embeddings)
            projected_embeddings = F.normalize(projected_embeddings, p=2, dim=1)
        
        # Try to find the queried song in the dataset
        query_song_id = None
        song_name_lower = song_name.lower()
        
        # Search through encoder classes to find a match
        for idx, song_id in enumerate(song_encoder.classes_):
            song_id_str = str(song_id).lower()
            if song_name_lower in song_id_str or song_id_str in song_name_lower:
                query_song_id = idx
                break
        
        # Use the found song's embedding as query, or fallback to first song
        if query_song_id is not None:
            query_embedding = projected_embeddings[query_song_id:query_song_id+1]
        else:
            query_embedding = projected_embeddings[0:1]
        
        # Compute cosine similarity between query and all songs
        similarities = torch.matmul(query_embedding, projected_embeddings.T)
        
        # Get top-k similar songs
        exclude_idx = query_song_id if query_song_id is not None else 0
        top_k_indices = torch.topk(similarities.squeeze(), k + 1).indices.cpu().numpy()
        
        # Filter out the query song and get top k
        recommendations = []
        for idx in top_k_indices:
            if idx != exclude_idx:
                score = similarities[0, idx].item()
                recommendations.append({
                    'song_id': int(idx),
                    'score': float(score),
                    'youtube_video': None  # Will be filled below
                })
                if len(recommendations) >= k:
                    break
        
        # Search YouTube for the query song
        query_youtube_results = search_youtube_video(song_name, artist, max_results=1)
        query_youtube = query_youtube_results[0] if query_youtube_results else None
        
        # For recommended songs, search YouTube for similar songs
        # We'll use the query song name to get related videos
        youtube_results = search_youtube_video(song_name, artist, max_results=k+5)
        
        # Assign YouTube videos to recommendations
        if youtube_results:
            for i, rec in enumerate(recommendations):
                if i < len(youtube_results):
                    rec['youtube_video'] = youtube_results[i]
                else:
                    # If we don't have enough YouTube results, try different search terms
                    alt_query = f"music similar to {song_name}"
                    alt_results = search_youtube_video(alt_query, None, max_results=1)
                    if alt_results:
                        rec['youtube_video'] = alt_results[0]
        
        return jsonify({
            'query_song': song_name,
            'query_song_id': int(exclude_idx) if query_song_id is not None else None,
            'youtube_video': query_youtube,
            'recommendations': recommendations[:k]
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to get recommendations: {str(e)}'}), 500


@app.route('/api/get-youtube-video', methods=['GET'])
def get_youtube_video():
    """Get YouTube video details for a list of songs."""
    song_names = request.args.getlist('songs')
    
    if not song_names:
        return jsonify({'error': 'No songs provided'}), 400
    
    results = []
    for song_name in song_names[:10]:
        video_info = search_youtube_video(song_name, None, max_results=1)
        if video_info:
            results.append({
                'song_name': song_name,
                **video_info[0]
            })
    
    return jsonify({'results': results})


if __name__ == '__main__':
    print("Loading data...")
    load_data()
    print("Data loaded successfully!")
    print(f"Device: {device}")
    print(f"Users registered: {len(users_db)}")
    print(f"YouTube API configured: {'Yes' if YOUTUBE_API_KEY else 'No'}")
    app.run(debug=True, host='0.0.0.0', port=5000)
