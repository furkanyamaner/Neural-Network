import torch
import pandas as pd
import preprocesses 
from model_architecture import GameRecommenderNet

# --- MODEL INITIALIZATION ---

# Define the input dimensions based on the preprocessed dataset
num_users = 11350
num_items = 3600

# Initialize the architecture and load the pre-trained weights
model = GameRecommenderNet(num_users, num_items)
model.load_state_dict(torch.load('steam_model.pth'))

# Set the model to evaluation (inference) mode
# This disables layers like Dropout to ensure consistent predictions
model.eval() 

def get_recommendations(user_id, top_n=5):
    """
    Generates personalized game recommendations for a given user by 
    predicting ratings for games they haven't interacted with yet.
    """
    # Step 1: Map the raw User ID to the internal categorical index
    try:
        user_idx = preprocesses.play_df[preprocesses.play_df['user_id'] == user_id]['user_idx'].iloc[0]
    except IndexError:
        return "Error: User ID not found in the dataset."

    # Step 2: Identify games the user has already played to avoid redundant suggestions
    played_games = preprocesses.play_df[preprocesses.play_df['user_id'] == user_id]['game_idx'].unique()
    
    # Step 3: Filter the global game pool to find candidates for recommendation
    all_game_indices = preprocesses.play_df['game_idx'].unique()
    unplayed_games = [g for g in all_game_indices if g not in played_games]
    
    # Step 4: Prepare input tensors for the Neural Network
    # We create a paired list of the target user index and all candidate game indices
    user_tensor = torch.tensor([user_idx] * len(unplayed_games), dtype=torch.long)
    game_tensor = torch.tensor(unplayed_games, dtype=torch.long)
    
    # Step 5: Perform inference to predict the engagement 'rating' for each game
    with torch.no_grad():
        # The model outputs a continuous value representing the predicted interest level
        predictions = model(user_tensor, game_tensor).squeeze()
    
    # Step 6: Rank the candidate games based on the highest predicted scores
    top_indices = predictions.argsort(descending=True)[:top_n]
    recommended_indices = [unplayed_games[i] for i in top_indices]
    
    # Step 7: Map the numerical indices back to their original Steam titles
    game_mapping_df = preprocesses.play_df[['game_idx', 'game_title']].drop_duplicates().set_index('game_idx')
    recommended_titles = game_mapping_df.loc[recommended_indices]['game_title'].values
    
    return recommended_titles

# --- TEST EXECUTION ---

# Testing the recommendation engine with a sample User ID from the dataset
sample_user_id = 59945701
print(f"Generating top recommendations for User {sample_user_id}:")
print(get_recommendations(sample_user_id))