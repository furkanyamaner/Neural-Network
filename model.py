import preprocesses
import torch
import torch.nn as nn


class GameRecommenderNet(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(GameRecommenderNet, self).__init__()

        #Kullanıcı ve Oyun için Embedding Katmanları
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)

        # Çok Katmanlı Algılayıcı (MLP) Bölümü
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # Ezberlemeyi (overfitting) önlemek için
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Çıktı: Tahmin edilen Rating
        )
    
    def forward(self, user_indices, item_indices):
        user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding(item_indices)

        # Kullanıcı ve Oyun Vektörlerini Birleştirme
        x = torch.cat([user_vec, item_vec], dim=-1)

        # Sinir ağından geçiriyoruz

        prediction = self.fc_layers(x)
        return prediction
    
    # Modelimizi oluşturalım (Senin sayılarını kullanarak)
num_users = 11350
num_items = 3600
model = GameRecommenderNet(num_users, num_items)

print(model)