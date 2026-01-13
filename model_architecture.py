import preprocesses
import torch
import torch.nn as nn

class GameRecommenderNet(nn.Module):
    """
    Neural Collaborative Filtering (NCF) architecture.
    This model utilizes Embedding layers to represent users and items in a 
    latent space, followed by an MLP to learn complex interaction patterns.
    """
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(GameRecommenderNet, self).__init__()

        # Latent space representation for both Users and Items
        # These embeddings capture the hidden characteristics of gaming preferences
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)

        # Multi-Layer Perceptron (MLP) section
        # We concatenate embeddings and pass them through dense layers to predict ratings
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # Dropout layer to prevent overfitting and improve generalization
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Output layer: Returns the predicted engagement rating
        )
    
    def forward(self, user_indices, item_indices):
        """
        Forward pass for the recommendation engine.
        Converts indices to vectors and concatenates them for MLP processing.
        """
        user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding(item_indices)

        # Concatenate User and Item vectors into a single feature vector
        x = torch.cat([user_vec, item_vec], dim=-1)

        # Pass the concatenated vector through the neural network layers
        prediction = self.fc_layers(x)
        return prediction

# Initialize model parameters based on the preprocessed dataset statistics
# These values correspond to the unique user and game counts found in steam-200k.csv
num_users = 11350
num_items = 3600
model = GameRecommenderNet(num_users, num_items)

# Print the model summary to verify layer configurations and parameters
print(model)