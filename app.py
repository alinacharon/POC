import requests
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
import torch
from transformers import AutoTokenizer
import joblib
from model import MultiLabelDeberta
from huggingface_hub import hf_hub_download

# ========== Loading model and data ==========
st.set_page_config(page_title="Tag Predictor", layout="wide")

REPO_ID = "Framby/deberta_multilabel"
deberta_path = hf_hub_download(
    repo_id=REPO_ID, filename="deberta_multilabel.pt")
mlb_path = hf_hub_download(repo_id=REPO_ID, filename="mlb.pkl")


@st.cache_resource
def load_model_and_tokenizer():
    mlb = joblib.load(mlb_path)
    model = MultiLabelDeberta(num_labels=len(mlb.classes_))
    model.load_state_dict(torch.load(
        deberta_path, map_location="cpu", weights_only=False))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-v3-base", use_fast=False)
    return model, tokenizer, mlb


model, tokenizer, mlb = load_model_and_tokenizer()

# ========== data loading ==========


@st.cache_data
def load_data():
    X = pd.read_csv('X_text.csv')['text_clean'].astype(str)
    Y = pd.read_csv('Y_tags.csv', converters={'Tags': eval})['Tags']
    return X, Y


X, Y = load_data()

# ========== prediction function ==========


def predict_tags(text, threshold=0.5):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding='max_length'
    )
    inputs.pop('token_type_ids', None)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
    binary_preds = (probs >= threshold).astype(int)
    predicted_tags = mlb.inverse_transform(
        np.expand_dims(binary_preds, axis=0))
    return predicted_tags[0]


# ========== interface ==========
st.title("Prédicteur de Tags StackOverflow")

st.markdown("## 1. Analyse des données textuelles")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Distribution de la longueur des questions")
    text_lengths = X.apply(lambda x: len(x.split()))
    fig = px.histogram(text_lengths, nbins=30,
                       title="Distribution de la longueur des questions")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Mots les plus fréquents")
    all_words = " ".join(X).split()
    word_freq = Counter(all_words)
    most_common_words = pd.DataFrame(
        word_freq.most_common(20), columns=['Mot', 'Nombre'])
    fig2 = px.bar(most_common_words, x='Mot', y='Nombre',
                  title="20 mots les plus fréquents")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("### Nuage de mots")
wc = WordCloud(width=800, height=300,
               background_color='white').generate(" ".join(X))
fig_wc, ax = plt.subplots(figsize=(10, 4))
ax.imshow(wc, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig_wc)

st.markdown("---")
st.markdown("## 2. Prédiction des tags")

input_text = st.text_area("Entrez une question StackOverflow", height=150)
threshold = st.slider("Seuil de probabilité", 0.1, 0.9, 0.5, 0.05)

if st.button("Prédire les tags"):
    if input_text.strip():
        tags = predict_tags(input_text, threshold)
        if tags:
            st.success("Tags prédits :")
            st.write(", ".join(tags))
        else:
            st.warning("Aucun tag trouvé pour le seuil sélectionné.")
    else:
        st.warning("Veuillez entrer une question.")
