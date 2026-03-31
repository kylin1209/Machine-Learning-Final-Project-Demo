"""
Shared helper functions and decorators.
"""

import os
import random

import numpy as np

def generate_data_audit_report(df, embeddings):
    """
    Generates a comprehensive audit report for Phase 1 data.
    """
    report = {
        'total_games': len(df),
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_by_column': df.isnull().sum().to_dict(),
        'duplicates': len(df) - len(df.drop_duplicates()),
        'price_stats': {
            'min': df['price'].min(),
            'max': df['price'].max(),
            'mean': df['price'].mean(),
            'free_games': (df['price'] == 0).sum()
        },
        'embedding_stats': {
            'shape': embeddings.shape,
            'mean_norm': np.linalg.norm(embeddings, axis=1).mean(),
            'has_nan': np.any(np.isnan(embeddings))
        },
        'year_range': f"{df['release_year'].min():.0f} - {df['release_year'].max():.0f}" 
                      if 'release_year' in df.columns else "N/A"
    }
    return report

def display_audit_in_app():
    """
    Streamlit helper to display audit report in UI (call in app.py).
    """
    import streamlit as st
    if st.checkbox("Show Data Audit Report"):
        audit = generate_data_audit_report(df, dataset_vectors)
        st.json(audit)
def set_reproducibility(seed=42):
	"""
	Configure deterministic seeds for common RNG sources used by the project.
	"""
	os.environ.setdefault("PYTHONHASHSEED", str(seed))
	random.seed(seed)
	np.random.seed(seed)

	try:
		import torch

		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)
	except Exception:
		pass
