import numpy as np
import emoji
import onnxruntime as ort
from gensim.models import KeyedVectors
import joblib
import re
import gradio as gr

# Load model dan komponen

onnx_model_path = "models/maxim-sentiment-models.onnx"
vec_model_path  = "models/maxim_fasttext.vec"
label_encoder_path = "models/label_encoder.pkl"

print("🔹 Loading FastText vectors...")
ft_model = KeyedVectors.load_word2vec_format(vec_model_path)

print("🔹 Loading label encoder...")
le = joblib.load(label_encoder_path)

print("🔹 Loading ONNX model...")
session = ort.InferenceSession(onnx_model_path)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Fungsi konversi teks ke vektor

def review_to_vec(tokens, model):
    vecs = [model[word] for word in tokens if word in model]
    return np.mean(vecs, axis=0) if len(vecs) > 0 else np.zeros(model.vector_size)

# Fungsi prediksi sentimen

def predict_sentiment(text):
    # Bersihkan teks
    tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', emoji.replace_emoji(text.lower(), '')).split()
    
    # Konversi ke vektor
    vec = review_to_vec(tokens, ft_model)
    input_tensor = vec.reshape(1, 1, -1).astype(np.float32)
    
    # Jalankan model ONNX
    outputs = session.run([output_name], {input_name: input_tensor})
    pred_class = np.argmax(outputs[0], axis=1)[0]
    
    # Decode label hasil prediksi
    pred_label = le.inverse_transform([pred_class])[0]
    return f"🟢 Sentimen: {pred_label}"


# Gradio Interface

interface = gr.Interface(
    fn=predict_sentiment,
    flagging_mode="never",
    inputs=gr.Textbox(label="Masukkan ulasan"),
    outputs=gr.Textbox(label="Hasil Prediksi"),
    title="Maxim Analisis Sentimen",
    description="Yuk, berikan ulasanmu tentang Maxim 🤗"
)

if __name__ == "__main__":
    interface.launch(server_name="127.0.0.1", server_port=7860)

