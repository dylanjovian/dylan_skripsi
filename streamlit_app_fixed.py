import os
import re
import torch
import torch.nn as nn
import streamlit as st
from transformers import AutoTokenizer, AutoModel

# =====================================================
# KONFIGURASI
# =====================================================

MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LEN = 128

LABELS = [
    "APLIKASI_POSITIF",
    "APLIKASI_NEGATIF",
    "INTERFACE_POSITIF",
    "INTERFACE_NEGATIF",
    "LAYANAN_POSITIF",
    "LAYANAN_NEGATIF",
    "KEAMANAN_POSITIF",
    "KEAMANAN_NEGATIF"
]

# =====================================================
# PATH MODEL
# =====================================================

POSSIBLE_MODEL_PATHS = [
    "best_model_indobert.pt",
    "./best_model_indobert.pt",
    "/content/drive/MyDrive/Dataset/Skripsi/skenario_7/best_model_indobert.pt"
]

MODEL_PATH = None
for path in POSSIBLE_MODEL_PATHS:
    if os.path.exists(path):
        MODEL_PATH = path
        break

# =====================================================
# PREPROCESSING
# =====================================================

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =====================================================
# MODEL INDOBERT
# =====================================================

class IndoBERTClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    if MODEL_PATH is None:
        raise FileNotFoundError(
            """
            File model tidak ditemukan.

            Pastikan file:
            best_model_indobert.pt

            berada dalam folder project
            atau Google Drive yang sesuai.
            """
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = IndoBERTClassifier(len(LABELS))
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return tokenizer, model

# =====================================================
# PREDIKSI
# =====================================================

def predict(text, tokenizer, model):
    text = preprocess(text)
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(
            encoding["input_ids"],
            encoding["attention_mask"]
        )
        probabilities = torch.sigmoid(logits).squeeze()

    return probabilities.cpu().numpy()

# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(
    page_title="Analisis Sentimen Multi Label",
    page_icon=":bar_chart:",
    layout="wide"
)

st.title("Analisis Sentimen Multi Label IndoBERT")

st.markdown(
    """
    Aplikasi ini menggunakan model IndoBERT
    untuk mendeteksi sentimen berdasarkan aspek:

    - Aplikasi
    - Interface
    - Layanan
    - Keamanan
    """
)

try:
    tokenizer, model = load_model()
    st.success(f"Model berhasil dimuat dari:\n{MODEL_PATH}")
except Exception as e:
    st.error(f"Gagal memuat model:\n{e}")
    st.stop()

review = st.text_area(
    "Masukkan ulasan pengguna",
    height=200,
    placeholder="Contoh: Aplikasi sangat membantu tetapi tampilannya masih kurang menarik."
)

threshold = st.slider(
    "Threshold Prediksi",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

if st.button("Prediksi"):
    if review.strip() == "":
        st.warning("Masukkan ulasan terlebih dahulu.")
        st.stop()

    probs = predict(review, tokenizer, model)
    st.subheader("Hasil Prediksi")

    for label, prob in zip(LABELS, probs):
        prediction = "✅ Positif/Terdeteksi" if prob > threshold else "❌ Tidak Terdeteksi"
        st.write(f"**{label}** : {prediction}")
        st.progress(float(prob))
        st.caption(f"Probabilitas: {prob:.4f}")
