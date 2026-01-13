import pandas as pd
import numpy as np


cols = ['user_id', 'game_title', 'behavior', 'value', 'zero']
# Note: Ensure the dataset file is in the same directory or provide the absolute path
df = pd.read_csv('steam-200k.csv', names=cols, usecols=range(5))

# Drop the unnecessary 'zero' column to clean the dataframe
df = df.drop(columns=['zero'])


# According to the dataset stats, 'play' accounts for 35% of the data
play_df = df[df['behavior'] == 'play'].copy()
purchase_df = df[df['behavior'] == 'purchase'].copy()

# Feature Engineering: Personalizing playtime ratings
# We calculate the maximum playtime per user to scale individual engagement
user_max_play = play_df.groupby('user_id')['value'].transform('max')

# Normalization: Scale ratings based on the user's own play sessions
# This mitigates the 'Popularity Bias' where extreme outliers affect the model
play_df['rating'] = play_df['value'] / user_max_play

# Logarithmic Transformation: Smoothen the distribution of ratings
# We normalize the result between 0 and 1 for better neural network convergence
play_df['rating'] = np.log1p(play_df['rating'] * 10) / np.log1p(10)

# Label Encoding: Convert User IDs and Game Titles into categorical indices
# These indices will be fed into the Embedding layers of our Neural Network
play_df['user_idx'] = play_df['user_id'].astype('category').cat.codes
play_df['game_idx'] = play_df['game_title'].astype('category').cat.codes

# Log basic dataset statistics for verification
print(f"Total Unique Users: {play_df['user_idx'].nunique()}")
print(f"Total Unique Games: {play_df['game_idx'].nunique()}")
print(play_df.head())

# Content Metadata: Mapping titles to genres for enhanced UI interpretation
# [cite_start]This adds a hybrid component to our collaborative filtering model [cite: 9]
genre_map = {
    'The Elder Scrolls V Skyrim': 'RPG / Open World',
    'Fallout 4': 'RPG / Post-Apocalyptic',
    'Left 4 Dead 2': 'FPS / Zombies',
    'Dota 2': 'MOBA / Strategy',
    'Team Fortress 2': 'FPS / Multiplayer',
    'Hearts of Iron IV': 'Grand Strategy / War',
    'Football Manager 2015': 'Simulation / Sports',
    'Counter-Strike Global Offensive': 'FPS / Competitive',
    'Sid Meier\'s Civilization V': 'Strategy / Turn-Based',
    'Portal 2': 'Puzzle / Adventure',
    'Garry\'s Mod': 'Sandbox',
    'Warframe': 'Action / RPG',
    'Unturned': 'Survival',
    'Terraria': 'Sandbox / Adventure'
}

def get_genre(game_title):
    """Returns the genre for a specific game title, defaults to 'Game / General'."""
    return genre_map.get(game_title, "Game / General")