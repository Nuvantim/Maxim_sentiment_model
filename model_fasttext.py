import pandas as pd
import numpy as np
import emoji
import os
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import FastText
import fasttext
from tqdm import tqdm
import logging
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# -----------------------------
# 1. Load dataset
# -----------------------------
data = pd.read_csv('sentiment_maxim_gplay.csv')

# -----------------------------
# 2. Clean text (remove emojis, non-alphanumeric, and empty rows)
# -----------------------------
data = pd.read_csv('sentiment_maxim_gplay.csv')
data['review'] = (
    data['review']
    .dropna()
    .apply(lambda s: emoji.replace_emoji(str(s), ''))
    .str.replace('[^a-zA-Z0-9]', ' ', regex=True)
    .replace('', np.nan)
)
data = data.dropna(subset=['review'])
data['review'] = data['review'].str.lower()
data = data[data['label'] != 'NETRAL']

# -----------------------------
# 3. Train FastText Model
# -----------------------------
nltk.download('punkt_tab')
data['token'] = data['review'].apply(word_tokenize)
data['token'] = data['token'].apply(lambda tokens: [t.lower() for t in tokens])

# export clean data
data[['review', 'token', 'label']].to_csv("clean_data.csv", index=False)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = FastText(
    sentences=data['token'],
    vector_size=500,
    window=5,
    min_count=5,
    sg=1,
    epochs=12
)
model.save("models/maxim_fasttext.model")
print("✅ fasttext .model created")

# -----------------------------
# 4. Balancing Data
# -----------------------------
print("=== DISTRIBUSI SEBELUM BALANCE ===")
print(data['label'].value_counts())
print(f"Total baris asli: {len(data)}")

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(data[['review']], data['label'])

data_balanced = pd.DataFrame({
    'review': X_resampled['review'],
    'label': y_resampled
})

# Export to .txt for FastText Supervised
data_balanced['ft_input'] = '__label__' + data_balanced['label'].astype(str) + ' ' + data_balanced['review']
data_balanced['ft_input'].to_csv('temp_train.txt', index=False, header=False, quoting=3, escapechar=' ')

print("\n=== DISTRIBUSI SESUDAH BALANCE ===")
print(data_balanced['label'].value_counts())
print(f"Total baris setelah balancing: {len(data_balanced)}")
# -----------------------------
# 5. Build FTZ Format
# -----------------------------
ftz = fasttext.train_supervised(
    input='temp_train.txt', 
    dim=500,                
    epoch=30,               
    minCount=2,             
    lr=0.2,                 
    wordNgrams=2,
    bucket=1000000
)
ftz.quantize(input='temp_train.txt', retrain=True)
ftz.save_model("models/maxim_fasttext.ftz")
os.remove('temp_train.txt')

print("✅ .ftz model created")
