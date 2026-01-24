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
model.wv.save_word2vec_format("models/maxim_fasttext.vec", binary=False)
model.wv.save_word2vec_format("models/maxim_fasttext.bin", binary=True)
print("✅ fasttext .model created")
