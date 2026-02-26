import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, ASTModel
import librosa
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder


# ============================================
# CONFIGURATION - Change these to your paths
# ============================================
drive_path = "/content/drive/MyDrive"  # Or use local path like "./data"
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
EMBEDDINGS_SAVE_PATH = "/content/drive/MyDrive/2nd_Project_Music/ast_embeddings.pt"
INTERACTIONS_SAVE_PATH = "/content/drive/MyDrive/2nd_Project_Music/fake_interactions.csv"

# Hyperparameters
NUM_USERS = 500
MIN_SONGS_PER_USER = 20
MAX_SONGS_PER_USER = 50
EMBEDDING_DIM = 64
BATCH_SIZE = 256
HYBRID_EPOCHS = 5
NCF_EPOCHS = 10
K = 10  # For precision@k


# ============================================
# Device setup
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================
# Audio embedding extraction
# ============================================
def load_ast_model():
    """Load the Audio Spectrogram Transformer model"""
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = ASTModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return processor, model


def get_audio_embedding(audio_path, processor, model):
    """Extract audio embedding for a single file"""
    y, sr = librosa.load(audio_path, sr=16000, duration=10.0)
    inputs = processor(y, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding


def extract_embeddings(base_path, max_songs=200):
    """Extract embeddings for all mp3 files in directory"""
    processor, model = load_ast_model()
    
    # Load existing embeddings if available
    if os.path.exists(EMBEDDINGS_SAVE_PATH):
        embeddings = torch.load(EMBEDDINGS_SAVE_PATH)
        print(f"Loaded existing embeddings: {len(embeddings)}")
    else:
        embeddings = {}
    
    count = 0
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".mp3"):
                song_id = file.replace(".mp3", "")
                
                if song_id in embeddings:
                    continue
                
                path = os.path.join(root, file)
                try:
                    emb = get_audio_embedding(path, processor, model)
                    embeddings[song_id] = emb.squeeze(0).cpu()
                    count += 1
                    print(f"Saved: {song_id}")
                except Exception as e:
                    print(f"Error with {song_id}: {e}")
                
                if count >= max_songs:
                    break
        if count >= max_songs:
            break
    
    torch.save(embeddings, EMBEDDINGS_SAVE_PATH)
    print(f"Saved progress. Total embeddings: {len(embeddings)}")
    return embeddings


# ============================================
# Dataset class
# ============================================
class MusicDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.songs = torch.tensor(df["song_id"].values, dtype=torch.long)
        self.labels = torch.tensor(df["interaction"].values, dtype=torch.float)
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.songs[idx], self.labels[idx]


# ============================================
# NCF Model
# ============================================
class NCF(nn.Module):
    def __init__(self, num_users, num_songs, embedding_dim=64):
        super(NCF, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.song_embedding = nn.Embedding(num_songs, embedding_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, user, song):
        user_emb = self.user_embedding(user)
        song_emb = self.song_embedding(song)
        
        x = torch.cat([user_emb, song_emb], dim=1)
        output = self.mlp(x)
        
        return torch.sigmoid(output)


# ============================================
# Hybrid Model (NCF + Audio Content)
# ============================================
class HybridModel(nn.Module):
    def __init__(self, num_users, audio_embedding_matrix, user_emb_dim=64):
        super(HybridModel, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, user_emb_dim)
        
        # Freeze audio embeddings
        self.audio_embedding = nn.Embedding.from_pretrained(
            audio_embedding_matrix,
            freeze=True
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(user_emb_dim + 768, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, user, song):
        user_emb = self.user_embedding(user)
        audio_emb = self.audio_embedding(song)
        
        x = torch.cat([user_emb, audio_emb], dim=1)
        out = self.mlp(x)
        
        return torch.sigmoid(out)


# ============================================
# Data preparation functions
# ============================================
def create_interactions_df(embeddings):
    """Create random user-song interactions"""
    song_ids = list(embeddings.keys())
    
    interactions = []
    for user_id in range(NUM_USERS):
        num_listened = random.randint(MIN_SONGS_PER_USER, MAX_SONGS_PER_USER)
        listened_songs = random.sample(song_ids, num_listened)
        
        for song_id in listened_songs:
            interactions.append([user_id, song_id, 1])
    
    df = pd.DataFrame(interactions, columns=["user_id", "song_id", "interaction"])
    df.to_csv(INTERACTIONS_SAVE_PATH, index=False)
    print(f"Created interactions: {len(df)}")
    return df


def encode_ids(df):
    """Encode user and song IDs"""
    user_encoder = LabelEncoder()
    song_encoder = LabelEncoder()
    
    df["user_id"] = user_encoder.fit_transform(df["user_id"])
    df["song_id"] = song_encoder.fit_transform(df["song_id"])
    
    return df, user_encoder, song_encoder


def create_negative_samples(df, num_songs):
    """Add negative samples (songs user hasn't listened to)"""
    all_songs = set(range(num_songs))
    new_data = []
    
    for user in df["user_id"].unique():
        user_songs = set(df[df["user_id"] == user]["song_id"])
        non_listened = list(all_songs - user_songs)
        
        # Positive samples
        for song in user_songs:
            new_data.append([user, song, 1])
        
        # Negative samples (3x)
        for _ in range(len(user_songs) * 3):
            neg_song = random.choice(non_listened)
            new_data.append([user, neg_song, 0])
    
    return pd.DataFrame(new_data, columns=["user_id", "song_id", "interaction"])


def create_content_based_data(num_users, num_songs, normalized_audio, likes_per_user=30):
    """Create content-based interactions using audio similarity"""
    content_data = []
    
    for user_id in range(num_users):
        seed_song = random.randint(0, num_songs - 1)
        
        seed_vector = normalized_audio[seed_song]
        similarities = torch.matmul(normalized_audio, seed_vector)
        
        top_similar = torch.topk(similarities, k=likes_per_user + 1).indices.tolist()
        top_similar = [s for s in top_similar if s != seed_song][:likes_per_user]
        
        for song_id in top_similar:
            content_data.append([user_id, song_id, 1])
    
    return pd.DataFrame(content_data, columns=["user_id", "song_id", "interaction"])


def train_test_split(content_df, test_ratio=0.2):
    """Split data into train/test sets"""
    train_rows = []
    test_rows = []
    
    for user in content_df["user_id"].unique():
        user_data = content_df[content_df["user_id"] == user]
        songs = user_data["song_id"].tolist()
        
        random.shuffle(songs)
        split_idx = int(len(songs) * test_ratio)
        
        for s in songs[split_idx:]:
            train_rows.append([user, s, 1])
        
        for s in songs[:split_idx]:
            test_rows.append([user, s, 1])
    
    return (
        pd.DataFrame(train_rows, columns=["user_id", "song_id", "interaction"]),
        pd.DataFrame(test_rows, columns=["user_id", "song_id", "interaction"])
    )


def get_top_k_recommendations(model, user_id, num_songs, k=10):
    """Get top-k song recommendations for a user"""
    model.eval()
    
    with torch.no_grad():
        user_tensor = torch.tensor([user_id] * num_songs).to(device)
        song_tensor = torch.arange(num_songs).to(device)
        
        scores = model(user_tensor, song_tensor)
        scores = scores.squeeze().cpu()
        
        top_k = torch.topk(scores, k=k).indices.tolist()
    
    return top_k


def precision_at_k(model, test_df, num_songs, k=10):
    """Calculate Precision@K"""
    model.eval()
    precision_scores = []
    
    for user in test_df["user_id"].unique():
        top_k = get_top_k_recommendations(model, user, num_songs, k)
        
        true_songs = set(test_df[test_df["user_id"] == user]["song_id"])
        
        hits = len(set(top_k) & true_songs)
        precision_scores.append(hits / k)
    
    return sum(precision_scores) / len(precision_scores) if precision_scores else 0


def train_model(model, train_df, num_epochs, batch_size=256, lr=0.001):
    """Train the model"""
    dataset = MusicDataset(train_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for users, songs, labels in dataloader:
            users = users.to(device)
            songs = songs.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            preds = model(users, songs)
            loss = criterion(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model


# ============================================
# Main execution
# ============================================
if __name__ == "__main__":
    # Step 1: Extract audio embeddings
    audio_path = drive_path + "/fma_small/fma_small/000/000002.mp3"
    base_path = drive_path + "/fma_small/fma_small"
    
    embeddings = extract_embeddings(base_path, max_songs=200)
    
    # Step 2: Create or load interactions
    if os.path.exists(INTERACTIONS_SAVE_PATH):
        df = pd.read_csv(INTERACTIONS_SAVE_PATH)
    else:
        df = create_interactions_df(embeddings)
    
    # Step 3: Encode IDs
    df, user_encoder, song_encoder = encode_ids(df)
    num_users = df["user_id"].nunique()
    num_songs = df["song_id"].nunique()
    
    print(f"Users: {num_users}, Songs: {num_songs}")
    
    # Step 4: Create negative samples
    new_df = create_negative_samples(df, num_songs)
    print(f"Dataset with negatives: {len(new_df)}")
    
    # Step 5: Train NCF model
    ncf_model = NCF(num_users, num_songs, EMBEDDING_DIM).to(device)
    ncf_model = train_model(ncf_model, new_df, NCF_EPOCHS, BATCH_SIZE)
    
    # Step 6: Load AST embeddings and create audio matrix
    ast_embeddings = torch.load(EMBEDDINGS_SAVE_PATH)
    embedding_dim = 768
    audio_embedding_matrix = torch.zeros(num_songs, embedding_dim)
    
    for idx in range(num_songs):
        original_song_id = str(song_encoder.classes_[idx]).zfill(6)
        if original_song_id in ast_embeddings:
            audio_embedding_matrix[idx] = ast_embeddings[original_song_id]
    
    audio_embedding_matrix = audio_embedding_matrix.to(device)
    print(f"Audio embedding matrix: {audio_embedding_matrix.shape}")
    
    # Step 7: Train Hybrid model
    hybrid_model = HybridModel(num_users, audio_embedding_matrix, EMBEDDING_DIM).to(device)
    hybrid_model = train_model(hybrid_model, new_df, HYBRID_EPOCHS, BATCH_SIZE)
    
    # Step 8: Content-based recommendations
    normalized_audio = F.normalize(audio_embedding_matrix, p=2, dim=1)
    content_df = create_content_based_data(num_users, num_songs, normalized_audio)
    print(f"Content-based data: {len(content_df)}")
    
    # Step 9: Add negative samples for content data
    final_df = create_negative_samples(content_df, num_songs)
    print(f"Final dataset: {len(final_df)}")
    
    # Step 10: Train NCF on content-based data
    ncf_model2 = NCF(num_users, num_songs, EMBEDDING_DIM).to(device)
    ncf_model2 = train_model(ncf_model2, final_df, NCF_EPOCHS, BATCH_SIZE)
    
    # Step 11: Evaluate
    prec = precision_at_k(ncf_model2, final_df, num_songs, K)
    print(f"Precision@{K}: {prec:.4f}")
    
    # Step 12: Train/test split evaluation
    train_pos_df, test_pos_df = train_test_split(content_df)
    train_df_split = create_negative_samples(train_pos_df, num_songs)
    
    ncf_model3 = NCF(num_users, num_songs, EMBEDDING_DIM).to(device)
    ncf_model3 = train_model(ncf_model3, train_df_split, NCF_EPOCHS, BATCH_SIZE)
    
    prec_test = precision_at_k(ncf_model3, test_pos_df, num_songs, K)
    print(f"Test Precision@{K}: {prec_test:.4f}")
