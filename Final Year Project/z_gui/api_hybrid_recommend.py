
from flask import Flask, request, jsonify
import torch
import pandas as pd
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config
from models.hybrid import HybridModel
from models.ncf import NCF
app = Flask(__name__)

device = config.get_device()
models = {}
interactions_df = None

def load_data():
    global interactions_df
    import os
    if os.path.exists(config.INTERACTIONS_SAVE_PATH):
        interactions_df = pd.read_csv(config.INTERACTIONS_SAVE_PATH)

def load_hybrid_model():
    global models
    if interactions_df is None:
        load_data()
    num_users = interactions_df['user_id'].nunique()
    num_songs = interactions_df['song_id'].nunique()
    
    # Try hybrid first
    try:
        hybrid_model = HybridModel(
            num_users=num_users,
            num_items=num_songs,
            audio_embedding_dim=config.AUDIO_EMBEDDING_DIM,
            user_embedding_dim=config.EMBEDDING_DIM,
            hidden_dims=config.HYBRID_HIDDEN_DIMS,
            dropout_rate=config.DROPOUT,
            use_attention=True
        ).to(device)
        hybrid_model.load_state_dict(torch.load(config.HYBRID_MODEL_PATH, map_location=device))
        hybrid_model.eval()
        models['hybrid'] = hybrid_model
        print("✅ Hybrid model loaded for user recs!")
        return True
    except Exception as e:
        print(f"❌ Hybrid load failed: {e}")
        # Fallback to NCF
        try:
            ncf_model = NCF(
                num_users=num_users,
                num_items=num_songs,
                embedding_dim=config.EMBEDDING_DIM,
                hidden_dims=config.NCF_HIDDEN_DIMS,
                dropout_rate=config.DROPOUT
            ).to(device)
            ncf_model.load_state_dict(torch.load(config.NCF_MODEL_PATH, map_location=device))
            ncf_model.eval()
            models['ncf'] = ncf_model
            print("🔄 Using NCF fallback")
            return False
        except:
            return False

@app.route('/api/hybrid-recommend', methods=['POST'])
def hybrid_recommend():
    data = request.json
    user_id = int(data.get('user_id', 0))
    k = int(data.get('k', 20))
    
    if interactions_df is None:
        load_data()
    
    if not models:
        load_hybrid_model()
    
    model = models.get('hybrid')
    if not model:
        return jsonify({'error': 'Hybrid model not ready. Wait training complete.'}), 400
    
    num_songs = interactions_df['song_id'].nunique()
    
    all_song_ids = torch.arange(num_songs, device=device)
    user_tensor = torch.tensor([user_id] * num_songs, device=device)
    
    with torch.no_grad():
        scores = model(user_tensor, all_song_ids)
    
    top_k_indices = torch.topk(scores, k).indices.cpu().tolist()
    top_k_scores = torch.topk(scores, k).values.cpu().tolist()
    
    recs = [{'song_id': int(sid), 'score': float(score)} for sid, score in zip(top_k_indices, top_k_scores)]
    
    return jsonify({
        'user_id': user_id,
        'model_type': 'hybrid',
        'recommendations': recs
    })

if __name__ == '__main__':
    load_data()
    print("Hybrid Recs API ready on port 5001")
    app.run(debug=True, port=5001)

