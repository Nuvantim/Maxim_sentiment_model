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
print(f"Distribusi awal: {Counter(data['label'])}")
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(data[['review']], data['label'])

data_balanced = pd.DataFrame({
    'review': X_resampled['review'],
    'label': y_resampled
})
print(f"Distribusi setelah Balancing: {Counter(data_balanced['label'])}")

# -----------------------------
# Export FastText formatting
# -----------------------------
data_balanced['ft_input'] = '__label__' + data_balanced['label'].astype(str) + ' ' + data_balanced['review']
data_balanced['ft_input'].to_csv('temp_train.txt', index=False, header=False, quoting=3, escapechar=' ')

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
    epochs=12
)

# -----------------------------
# 8. Save trained FastText model
# -----------------------------
model.save("models/maxim_fasttext.model")
print("✅ FastText model berhasil disimpan ke models/maxim_fasttext.model")
model.wv.save_word2vec_format("models/maxim_fasttext.vec", binary=False)
print("✅ FastText vectors berhasil diexport ke models/maxim_fasttext.vec")

# -----------------------------
# 9. FTZ Format
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

print("✅ FTZ File Created")
