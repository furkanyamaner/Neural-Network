import pandas as pd
import numpy as np

cols = ['user_id', 'game_title', 'behavior', 'value', 'zero']
df = pd.read_csv('steam-200k.csv', names=cols, usecols=range(5))

df = df.drop(columns=['zero'])

# 2. Pivot Mantığıyla Veriyi Birleştirme
# Kullanıcının bir oyunu hem alıp hem oynamasını tek satıra indirelim
# Purchase (Satın alma) için 1, Play (Oynama süresi) için gerçek süreyi alalım
# Not: Analiz için 'play' verisine odaklanmak daha güçlü sonuç verir.

play_df = df[df['behavior'] == 'play'].copy()
purchase_df = df[df['behavior']=='purchase'].copy()

# Sadece oynanan oyunlar üzerinden bir "Rating" (Puan) oluşturalım
# Basit bir yöntem: Oynama süresinin logaritmasını alarak uç değerleri törpülemek

play_df['rating'] = np.log1p(play_df['value'])

# 3. ID Atama (Label Encoding)
# Modelin anlayacağı 0, 1, 2... şeklindeki indekslere dönüştürelim

play_df['user_idx'] = play_df['user_id'].astype('category').cat.codes
play_df['game_idx'] = play_df['game_title'].astype('category').cat.codes

print(f"Toplam Kullanıcı Sayısı: {play_df['user_idx'].nunique()}")
print(f"Toplam Oyun Sayısı: {play_df['game_idx'].nunique()}")
print(play_df.head())