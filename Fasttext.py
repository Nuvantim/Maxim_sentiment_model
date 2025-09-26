import pandas as pd
import numpy as np
import emoji
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import FastText
from tqdm import tqdm
import logging
import json
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 1. Load dataset
# -----------------------------
data = pd.read_csv('sentiment_maxim_gplay.csv')

# -----------------------------
# 2. Clean text (remove emojis, non-alphanumeric, and empty rows)
# -----------------------------
data['review'] = (
    data['review']
    .dropna()
    .apply(lambda s: emoji.replace_emoji(str(s), ''))       # hapus emoji
    .str.replace('[^a-zA-Z0-9]', ' ', regex=True)           # hanya huruf/angka
    .replace('', np.nan)
)
data = data.dropna(subset=['review'])

# -----------------------------
# 3. Remove neutral labels
# -----------------------------
data = data[data['label'] != 'NETRAL']

# -----------------------------
# 4. Tokenize words
# -----------------------------
nltk.download('punkt_tab')   # kalau error bisa pakai "punkt"
data['token'] = data['review'].apply(word_tokenize)

# -----------------------------
# 5. Keep only relevant columns and lowercase tokens
# -----------------------------
data = data[['review', 'token', 'label']]
data['token'] = data['token'].apply(lambda tokens: [t.lower() for t in tokens])

# -----------------------------
# 6. Export clean data
# -----------------------------
data.to_csv("clean_data.csv", index=False)
print("✅ clean_data.csv berhasil dibuat")

# -----------------------------
# 7. Train FastText model on tokenized data
# -----------------------------
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = FastText(
    sentences=data['token'],
    vector_size=500,
    window=5,
    min_count=5,
    sg=1,
    epochs=30
)

# -----------------------------
# 8. Save trained FastText model
# -----------------------------
model.save("models/maxim_fasttext.model")
print("✅ FastText model berhasil disimpan ke models/maxim_fasttext.model")
model.wv.save_word2vec_format("models/maxim_fasttext.vec", binary=False)
print("✅ FastText vectors berhasil diexport ke models/maxim_fasttext.vec")
