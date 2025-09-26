import pandas as pd
import numpy as np
import emoji
import torch
import joblib
import os
import torch.nn as nn
import json
from gensim.models import FastText
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from transformers import PretrainedConfig
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# 1. Load and preprocess text
# -----------------------------
data = pd.read_csv('clean_data.csv')
data['review'] = data['review'].str.lower()

# -----------------------------
# 2. Split features and labels
# -----------------------------
X = data['review'].astype(str)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# 3. Load pre-trained FastText model
# -----------------------------
ft_model = FastText.load("models/maxim_fasttext.model")

# -----------------------------
# 4. Convert reviews to embeddings
# -----------------------------
def review_to_vec(tokens, model):
    vecs = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vecs, axis=0) if len(vecs) > 0 else np.zeros(model.vector_size)

X_train_vec = np.array([review_to_vec(text.lower().split(), ft_model) for text in X_train])
X_test_vec  = np.array([review_to_vec(text.lower().split(), ft_model) for text in X_test])

# -----------------------------
# 5. Balance training data with SMOTE + undersampling
# -----------------------------
# Undersample mayoritas jadi 70% dari jumlah awal
X_under, y_under = RandomUnderSampler(
    sampling_strategy=0.7, random_state=42
).fit_resample(X_train_vec, y_train)

# SMOTE supaya minoritas = mayoritas hasil under
target_n = Counter(y_under).most_common(1)[0][1]
X_res, y_res = SMOTE(
    sampling_strategy={c: target_n for c in Counter(y_under)}, random_state=42
).fit_resample(X_under, y_under)

# -----------------------------
# 6. Encode labels to integers
# -----------------------------
le = LabelEncoder()
y_res_enc = le.fit_transform(y_res)
y_test_enc = le.transform(y_test)

# -----------------------------
# 7. Convert data to PyTorch tensors
# -----------------------------
X_train_tensor = torch.tensor(X_res, dtype=torch.float32)
y_train_tensor = torch.tensor(y_res_enc, dtype=torch.long)
X_test_tensor  = torch.tensor(X_test_vec, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test_enc, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# -----------------------------
# 8. Define GRU classifier
# -----------------------------
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        if x.dim() == 2:  
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out



input_size = X_train_vec.shape[1]
hidden_size = 64
num_layers = 3
num_classes = len(le.classes_)

model = GRUClassifier(input_size, hidden_size, num_layers, num_classes, dropout=0.2)
device = torch.device('cpu')
model = model.to(device)

# -----------------------------
# 9. Define loss function and optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

# -----------------------------
# 10. Train GRU model
# -----------------------------
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
    scheduler.step(avg_loss)

# -----------------------------
# 11. Evaluate model
# -----------------------------
model.eval()
with torch.no_grad():
    X_test_tensor = X_test_tensor.to(device)
    outputs = model(X_test_tensor)
    y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

print(confusion_matrix(y_test_enc, y_pred))
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))
# -----------------------------
# 12. Save model + config + label encoder
# -----------------------------
save_dir = "./models"
os.makedirs(save_dir, exist_ok=True)

torch.save(model.state_dict(), f"{save_dir}/pytorch_model.bin")

config_dict = {
    "model_type":"maxim-sentiment-models",
    "input_size": input_size,
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "num_classes": num_classes
}

with open(f"{save_dir}/config.json", "w") as f:
    json.dump(config_dict, f, indent=2)

joblib.dump(le, f"{save_dir}/label_encoder.pkl")

print(f"Model saved in {save_dir}")

