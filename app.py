import streamlit as st
import torch
import pandas as pd
import preprocesses
from model_architecture import GameRecommenderNet

# --- Starting the program without running the training process (already completed) ---
# --- We can start with 'streamlit run app.py' command in the terminal ---
# --- The training process is already completed and the model weights are saved to the file 'steam_model.pth' in the current directory ---
# -------------------------------------------------------------------------------------------------------------------------------------------------

# --- PAGE CONFIGURATION ---
# Setting the wide layout to provide a professional dashboard experience
st.set_page_config(page_title="Neural Game Recommender", page_icon="🎮", layout="wide")

# --- CUSTOM INTERFACE STYLING ---
# Implementing CSS for a modern dark-themed aesthetic and interactive game cards
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #6c63ff; color: white; border: none; transition: 0.3s; }
    .stButton>button:hover { background-color: #554fcc; }
    .game-card { padding: 15px; border-radius: 12px; background-color: #1e1e26; margin-bottom: 12px; border-left: 6px solid #6c63ff; box-shadow: 2px 2px 10px rgba(0,0,0,0.2); }
    .genre-tag { background-color: #3d3d4d; color: #b3b3ff; padding: 2px 8px; border-radius: 8px; font-size: 0.8rem; display: inline-block; margin-top: 5px; }
    .user-sidebar { height: 400px; overflow-y: auto; background-color: #1e1e26; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """
    Initializes the GameRecommenderNet and loads pre-trained weights.
    Cached to ensure high performance and prevent redundant loading.
    """
    num_users = 11350
    num_items = 3600
    model = GameRecommenderNet(num_users, num_items)
    model.load_state_dict(torch.load('steam_model.pth'))
    model.eval() # Set model to inference mode
    return model

# Load the neural network architecture
model = load_model()

# --- DASHBOARD LAYOUT ---
# 70% Left Column for Main Operations, 30% Right Column for User Quick-Selection
col1, col2 = st.columns([0.7, 0.3])

with col2:
    st.subheader("👥 Sample Users")
    st.write("Select a User ID from the dataset to test the engine:")
    
    # Retrieve top 100 most active users to ensure meaningful recommendations
    top_users = preprocesses.play_df['user_id'].value_counts().head(100).index.tolist()
    
    # Interactive selection list
    selected_from_list = st.selectbox("Quick Selection List", top_users, index=None, placeholder="Choose a User ID...")
    
    if selected_from_list:
        st.info(f"Active User ID: {selected_from_list}")

with col1:
    st.title("🎮 Neural Game Recommender")
    st.write("Welcome to the Neural Network-powered Steam recommendation engine!")
    st.divider()

    # User Input Field (Synchronized with the sample list)
    default_id = str(selected_from_list) if selected_from_list else ""
    user_input = st.text_input("Enter Steam User ID:", value=default_id)

    if st.button("Generate Recommendations"):
        if user_input:
            try:
                user_id = int(user_input)
                with st.spinner('Neural Network is analyzing user preferences...'):
                    # Data lookup for the selected user
                    user_idx_row = preprocesses.play_df[preprocesses.play_df['user_id'] == user_id]
                    
                    if user_idx_row.empty:
                        st.error("User ID not found! Please ensure the ID exists in the dataset.")
                    else:
                        # Map raw ID to categorical index
                        user_idx = user_idx_row['user_idx'].iloc[0]
                        
                        # Filter out games already owned by the user
                        played_games = preprocesses.play_df[preprocesses.play_df['user_id'] == user_id]['game_idx'].unique()
                        all_game_indices = preprocesses.play_df['game_idx'].unique()
                        candidate_games = [g for g in all_game_indices if g not in played_games]
                        
                        # Prepare input tensors for inference
                        user_tensor = torch.tensor([user_idx] * len(candidate_games), dtype=torch.long)
                        game_tensor = torch.tensor(candidate_games, dtype=torch.long)
                        
                        # Execute forward pass through the model
                        with torch.no_grad():
                            predictions = model(user_tensor, game_tensor).squeeze()
                        
                        # Rank top 5 titles based on predicted engagement score
                        top_indices = predictions.argsort(descending=True)[:5]
                        recommended_game_indices = [candidate_games[i] for i in top_indices]
                        
                        # Map indices back to Game Titles
                        game_map = preprocesses.play_df[['game_idx', 'game_title']].drop_duplicates().set_index('game_idx')
                        recommendations = game_map.loc[recommended_game_indices]['game_title'].values
                        
                        st.subheader(f"Top 5 Personalized Recommendations:")
                        for i, game in enumerate(recommendations, 1):
                            # Fetch metadata for enhanced result interpretation
                            genre = preprocesses.get_genre(game)
                            st.markdown(f"""
                                <div class='game-card'>
                                    <div style='font-size: 1.1rem;'><b>{i}. {game}</b></div>
                                    <div class='genre-tag'>🏷️ {genre}</div>
                                </div>
                                """, unsafe_allow_html=True)
            except ValueError:
                st.warning("Invalid input. Please enter a numerical User ID.")

# Sidebar Credit and Methodology Info
st.sidebar.markdown(f"**Developer:**\nFurkan YAMANER")
st.sidebar.info("Methodology: Neural Collaborative Filtering (NCF) with Embedding Layers.")