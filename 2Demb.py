"""
Streamlit 2D Embedding Visualizer (PCA / t-SNE) + Clustering

How to run:
    1) Install dependencies:
        pip install streamlit gensim scikit-learn plotly numpy pandas matplotlib
    2) Run app:
        streamlit run streamlit_embedding_app.py

Features:
    - Load pre-trained GloVe embeddings (gensim downloader) OR upload your own word->vector .txt/.npy (optional)
    - Pick seed words or provide a list
    - Expand neighborhood by grabbing most-similar words
    - Reduce to 2D using PCA or t-SNE (PCA->t-SNE pipeline recommended)
    - KMeans clustering (auto k heuristic or custom k)
    - Interactive Plotly scatter with cluster coloring and hover labels
    - Download CSV of 2D coordinates + cluster labels

Notes:
    - t-SNE can be slow on many points; keep `Max words` reasonably small (200-800 depending on your machine).

"""

import streamlit as st
import numpy as np
import pandas as pd
import math
from collections import OrderedDict

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import plotly.express as px

# Lazy import gensim only when needed
try:
    import gensim.downloader as api
    HAS_GENSIM = True
except Exception:
    HAS_GENSIM = False

st.set_page_config(page_title='2D Embedding Visualizer', layout='wide')
st.title('2D Embedding Visualizer — PCA / t-SNE + Clustering')

# --- Sidebar controls ---
with st.sidebar:
    st.header('Controls')

    source = st.selectbox('Embeddings source', ['glove-wiki-gigaword-100 (gensim)', 'Upload own (text .txt or .npy)'])

    if source.startswith('glove') and not HAS_GENSIM:
        st.warning('gensim not available in this environment. Upload your embeddings or install gensim.')

    seeds_input = st.text_area('Seed words (comma separated)', 'king, queen, man, woman, cat, dog, python, java, music, guitar')
    num_neighbors = st.number_input('Num similar neighbors per seed', min_value=0, max_value=50, value=8, step=1)
    max_words = st.number_input('Max words (cap total points)', min_value=10, max_value=2000, value=300, step=10)

    method = st.radio('Projection method', ['PCA only', 't-SNE (PCA -> t-SNE)'])
    tsne_perp = st.number_input('t-SNE perplexity', min_value=5, max_value=200, value=30, step=1)

    auto_k = st.checkbox('Auto-choose K (heuristic)', value=True)
    custom_k = None
    if not auto_k:
        custom_k = st.number_input('K for KMeans', min_value=2, max_value=50, value=8, step=1)

    random_state = st.number_input('Random seed', min_value=0, value=42, step=1)

    upload_file = None
    if source.startswith('Upload'):
        upload_file = st.file_uploader('Upload embeddings (word vectors text or .npy)', type=['txt', 'npy', 'npz', 'vec'])

    run_button = st.button('Build & Visualize')

# --- Helper functions ---

@st.cache_data(show_spinner=False)
def load_gensim_model(name='glove-wiki-gigaword-100'):
    return api.load(name)


def load_embeddings_from_text(f):
    # assume text lines: word v1 v2 ...
    d = {}
    for raw in f.read().decode('utf-8').splitlines():
        if not raw.strip():
            continue
        parts = raw.strip().split()
        if len(parts) < 2:
            continue
        w = parts[0]
        vec = np.array([float(x) for x in parts[1:]], dtype=float)
        d[w] = vec
    return d


def load_embeddings_from_npy(f):
    # expect a .npz or .npy containing an array and optionally a vocab
    # User-provided format can vary — try to be flexible
    try:
        arr = np.load(f)
        # if npz with arrays
        if isinstance(arr, np.lib.npyio.NpzFile):
            # look for 'vectors' and 'vocab' or fallback to first two keys
            keys = list(arr.keys())
            if 'vectors' in keys and 'vocab' in keys:
                vectors = arr['vectors']
                vocab = list(arr['vocab'])
                return {w: vectors[i] for i, w in enumerate(vocab)}
            else:
                st.error('Uploaded .npz has unexpected keys: ' + ','.join(keys))
                return {}
        else:
            st.error('Uploaded .npy is not a mapping; please provide text format or .npz with vocab + vectors.')
            return {}
    except Exception as e:
        st.error(f'Failed to load .npy/.npz: {e}')
        return {}


def build_vocab_and_vectors(model, seed_words, num_neighbors, max_words):
    words_ordered = OrderedDict()
    # seeds
    for w in seed_words:
        if hasattr(model, '__contains__') and w in model:
            words_ordered[w] = model[w]
        elif isinstance(model, dict) and w in model:
            words_ordered[w] = model[w]
        else:
            # skip silently
            pass

    # neighbors
    for w in list(words_ordered.keys()):
        try:
            if HAS_GENSIM and hasattr(model, 'most_similar'):
                sims = model.most_similar(w, topn=num_neighbors)
                for neigh, score in sims:
                    if neigh not in words_ordered:
                        words_ordered[neigh] = model[neigh]
            elif isinstance(model, dict):
                # brute force: compute cosine similarity to all words in dict
                # (only do if dict isn't huge)
                all_keys = list(model.keys())
                vec = model[w]
                # compute dot prod similarity quickly
                mat = np.array([model[k] for k in all_keys])
                norms = np.linalg.norm(mat, axis=1) * np.linalg.norm(vec)
                sims = (mat @ vec) / (norms + 1e-12)
                idx = np.argsort(-sims)[:num_neighbors]
                for i in idx:
                    k = all_keys[i]
                    if k not in words_ordered:
                        words_ordered[k] = model[k]
        except Exception:
            continue

    # trim
    all_words = list(words_ordered.keys())[:max_words]
    vectors = np.array([words_ordered[w] for w in all_words])
    return all_words, vectors


def project_vectors(vectors, method='PCA only', tsne_perp=30, random_state=42):
    scaler = StandardScaler()
    vectors_scaled = scaler.fit_transform(vectors)

    if method == 'PCA only':
        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(vectors_scaled)
        return coords
    else:
        # PCA -> t-SNE pipeline
        reduced_dim = min(50, vectors_scaled.shape[1])
        pca50 = PCA(n_components=reduced_dim, random_state=random_state)
        v50 = pca50.fit_transform(vectors_scaled)
        perp = min(max(5, int(tsne_perp)), max(5, len(v50)//3))
        tsne = TSNE(n_components=2, perplexity=perp, learning_rate='auto', init='pca', random_state=random_state)
        coords = tsne.fit_transform(v50)
        return coords


def choose_k(n, auto=True, custom_k=None):
    if auto:
        return max(2, int(math.sqrt(n / 2)))
    else:
        return max(2, int(custom_k))

# --- Main execution when user clicks ---
if run_button:
    seed_words = [s.strip() for s in seeds_input.split(',') if s.strip()]
    if len(seed_words) == 0:
        st.error('Please provide at least one seed word.')
    else:
        with st.spinner('Loading embeddings...'):
            model = None
            if source.startswith('glove'):
                if not HAS_GENSIM:
                    st.stop()
                model = load_gensim_model()
            else:
                if upload_file is None:
                    st.error('Please upload an embeddings file when "Upload own" selected.')
                    st.stop()
                else:
                    if upload_file.type in ['text/plain'] or upload_file.name.endswith('.txt') or upload_file.name.endswith('.vec'):
                        model = load_embeddings_from_text(upload_file)
                    else:
                        model = load_embeddings_from_npy(upload_file)

        if model is None or (isinstance(model, dict) and len(model) == 0):
            st.error('No embeddings loaded. Aborting.')
            st.stop()

        with st.spinner('Building vocabulary and vectors...'):
            all_words, vectors = build_vocab_and_vectors(model, seed_words, int(num_neighbors), int(max_words))

        if len(all_words) == 0:
            st.error('No words found from seeds in the vocabulary. Try different seeds or upload embeddings.')
            st.stop()

        st.success(f'Built vocabulary with {len(all_words)} words.')

        # projection
        with st.spinner('Projecting to 2D...'):
            coords2d = project_vectors(vectors, method=method, tsne_perp=tsne_perp, random_state=int(random_state))

        # clustering
        k = choose_k(len(all_words), auto=auto_k, custom_k=custom_k)
        kmeans = KMeans(n_clusters=k, random_state=int(random_state))
        clusters = kmeans.fit_predict(StandardScaler().fit_transform(vectors))

        # build dataframe for plotting & download
        df = pd.DataFrame({
            'word': all_words,
            'x': coords2d[:, 0],
            'y': coords2d[:, 1],
            'cluster': clusters
        })

        # Plotly scatter
        fig = px.scatter(df, x='x', y='y', color=df['cluster'].astype(str), hover_data=['word'], title=f'{method} projection (k={k})')
        fig.update_traces(marker=dict(size=8))

        # layout: two columns
        col1, col2 = st.columns([3, 2])
        with col1:
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader('Cluster sample')
            for cid in sorted(df['cluster'].unique()):
                members = df[df['cluster'] == cid]['word'].tolist()
                st.markdown(f'**Cluster {cid}** ({len(members)}): ' + ', '.join(members[:30]))

            st.download_button('Download CSV (coords + cluster)', data=df.to_csv(index=False).encode('utf-8'), file_name='embeddings_2d_clusters.csv', mime='text/csv')

        st.info('Tip: If t-SNE is slow, reduce Max words or use PCA only.')

        st.balloons()

