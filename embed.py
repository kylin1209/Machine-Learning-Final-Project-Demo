import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import os

@st.cache_resource
def load_embedding_model(model_name='all-MiniLM-L6-v2'):
    """
    Loads and caches the SentenceTransformer model.
    'all-MiniLM-L6-v2' is chosen for a fast, high-quality MVP.
    """
    return SentenceTransformer(model_name)

@st.cache_resource
def load_cross_encoder_model(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
    """
    Loads and caches the CrossEncoder model for precision NLP reranking.
    """
    return CrossEncoder(model_name)

@st.cache_resource
def load_tfidf_model_and_matrix(text_list):
    """
    Initializes a Token-Frequency/Inverse-Document-Frequency Vectorizer 
    and Sparse Matrix using the entire global text descriptions to learn 
    baseline statistical keyword importance for Hybrid exact-keyword Retrieval.
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_features=15000, lowercase=True)
    matrix = vectorizer.fit_transform(text_list)
    return vectorizer, matrix


def combine_text_fields(df, field_weights=None):
    """
    Combines text fields with customizable weights/priorities.
    """
    if field_weights is None:
        field_weights = {
            'name': 2.0,              # Title is most important
            'short_description': 2.0,
            'genres': 1.5,
            'tags': 1.0,
            'categories': 0.5
        }
    
    def ensure_str(val):
        return str(val) if val else ""
    
    combined = ""
    for field, weight in field_weights.items():
        if field in df.columns:
            text = df[field].apply(ensure_str)
            # Repeat text by weight (e.g., weight=2.0 repeats field twice for emphasis)
            repetitions = int(weight)
            combined += (f"{field.replace('_', ' ').title()}: " + text + ". ") * repetitions
    
    return combined

@st.cache_data(show_spinner=False)
def generate_embeddings(_model, text_list, show_progress_bar=True):
    """
    Generates embeddings with validation (handles NaN/inf).
    """
    if show_progress_bar:
        batch_size = 32
        embeddings_list = []
        progress_bar = st.progress(0, text="Generating Embeddings...")
        total_batches = (len(text_list) + batch_size - 1) // batch_size
        
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i+batch_size]
            # Encode batch without terminal progress bar
            batch_embeddings = _model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
            embeddings_list.append(batch_embeddings)
            
            # Update progress
            progress = min((i + batch_size) / len(text_list), 1.0)
            batch_num = (i // batch_size) + 1
            progress_bar.progress(progress, text=f"Generating Embeddings... (Batch {batch_num}/{total_batches})")
            
        embeddings = np.vstack(embeddings_list)
        progress_bar.empty()
    else:
        embeddings = _model.encode(text_list, show_progress_bar=False, convert_to_numpy=True)
    
    # Validate embeddings
    if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
        st.warning("⚠️  NaN/Inf values detected in embeddings. Normalizing...")
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    
    return embeddings
