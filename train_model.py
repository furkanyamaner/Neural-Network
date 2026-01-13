from model_architecture import GameRecommenderNet
import preprocesses
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 1. PyTorch Dataset Implementation
# Wraps our user-item interactions into batches for efficient neural network training
class SteamDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

# --- NEGATIVE SAMPLING STRATEGY ---
def create_negative_samples(df, n_negatives=4):
    """
    To prevent the model from only learning what users like, we generate 'negative' samples.
    By randomly selecting games a user has never played and assigning a 0.0 rating,
    we force the model to learn personal boundaries and reduce popularity bias.
    """
    neg_samples = []
    all_game_indices = df['game_idx'].unique()
    unique_users = df['user_idx'].unique()
    
    print("Generating negative samples to balance the dataset...")
    
    for user in unique_users:
        # Identify games already played by the user
        played_games = set(df[df['user_idx'] == user]['game_idx'].values)
        # Identify the pool of unplayed games
        unplayed_games = list(set(all_game_indices) - played_games)
        
        # Select n_negatives random games from the unplayed pool
        if len(unplayed_games) > n_negatives:
            neg_selection = np.random.choice(unplayed_games, n_negatives, replace=False)
            for game in neg_selection:
                neg_samples.append([user, game, 0.0]) # Assign a zero-rating signal
                
    return pd.DataFrame(neg_samples, columns=['user_idx', 'game_idx', 'rating'])

# --- DATA PREPARATION PIPELINE ---

# Step 1: Generate negative feedback signals
negative_feedback_df = create_negative_samples(preprocesses.play_df)

# Step 2: Combine actual play data (positive) with generated negative samples
positive_feedback_df = preprocesses.play_df[['user_idx', 'game_idx', 'rating']]
full_dataset_df = pd.concat([positive_feedback_df, negative_feedback_df], ignore_index=True)

# Step 3: Train/Test Split (80% Training, 20% Evaluation)
train_df, test_df = train_test_split(full_dataset_df, test_size=0.2, random_state=42)

# Step 4: Initialize DataLoaders
train_loader = DataLoader(
    SteamDataset(train_df['user_idx'].values, train_df['game_idx'].values, train_df['rating'].values), 
    batch_size=64, shuffle=True
)
test_loader = DataLoader(
    SteamDataset(test_df['user_idx'].values, test_df['game_idx'].values, test_df['rating'].values), 
    batch_size=64, shuffle=False
)

# --- MODEL INITIALIZATION & HYPERPARAMETERS ---

# Initialize the Neural Collaborative Filtering model
model = GameRecommenderNet(
    num_users=len(preprocesses.play_df['user_idx'].unique()), 
    num_items=len(preprocesses.play_df['game_idx'].unique())
)

# Loss Function: Mean Squared Error is ideal for regression-based rating prediction
criterion = nn.MSELoss()

# Optimizer: Adam with Weight Decay to introduce L2 regularization and prevent overfitting
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

# --- TRAINING PHASE ---

history = []
epochs = 20
print(f"Starting training for {epochs} epochs...")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for user, item, rating in train_loader:
        optimizer.zero_grad() # Reset gradients
        
        # Forward Pass
        predictions = model(user, item).squeeze()
        
        # Calculate Loss
        loss = criterion(predictions, rating)
        
        # Backward Pass & Optimization
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    history.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs} | Average MSE Loss: {avg_loss:.4f}")

# --- VISUALIZATION ---

plt.figure(figsize=(10, 5))
plt.plot(history, label='Training Loss', color='#6c63ff', linewidth=2)
plt.title('Neural Network Training Progress')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('training_loss_plot.png')
print("Training visualization saved as 'training_loss_plot.png'.")

# --- MODEL PERSISTENCE ---

torch.save(model.state_dict(), 'steam_model.pth')
print("Model parameters successfully saved to 'steam_model.pth'.")

# --- EVALUATION METRICS ---

def evaluate_model(trained_model, data_loader):
    """
    Computes RMSE and MAE on the unseen test set to validate performance.
    These metrics are essential for the final project report.
    """
    trained_model.eval()
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for user, item, rating in data_loader:
            outputs = trained_model(user, item).squeeze()
            y_pred.extend(outputs.tolist())
            y_true.extend(rating.tolist())
            
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"\n--- Model Evaluation Results ---")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    return rmse, mae

# Execute final evaluation
evaluate_model(model, test_loader)