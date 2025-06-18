
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Spotify Song Clustering", page_icon="🎵", layout="wide")
st.title("🎵 Spotify Song Clustering PRO (Hierarchical Clustering)")
st.write("**Built by Ankit and Arihant 🚀**")

st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your Spotify CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    default_features = ['danceability', 'energy', 'loudness', 'speechiness',
                        'acousticness', 'instrumentalness', 'liveness',
                        'valence', 'tempo', 'duration_ms', 'popularity']
    features = st.sidebar.multiselect("Select Features for Clustering", numeric_columns, default=default_features)
    df_clean = df[features].dropna()
    st.sidebar.write(f"Total valid records: {df_clean.shape[0]}")
    subsample_size = st.sidebar.slider("Sample size", 1000, min(df_clean.shape[0], 10000), 3000)
    df_sampled = df_clean.sample(n=subsample_size, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_sampled)
    st.header("🔬 Feature Correlation Heatmap")
    fig_corr, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(X_scaled, columns=features).corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig_corr)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    st.header("🌿 Hierarchical Clustering Dendrogram")
    linked = linkage(X_scaled, method='ward')
    fig_dendro, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linked, truncate_mode='lastp', p=30, ax=ax)
    st.pyplot(fig_dendro)
    st.sidebar.header("⚙️ Clustering Control")
    k = st.sidebar.slider("Number of Clusters", 2, 10, 5)
    cluster_labels = fcluster(linked, k, criterion='maxclust')
    df_sampled['cluster'] = cluster_labels
    st.header("🎯 PCA Cluster Visualization")
    fig_pca, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette="Set2", s=60, ax=ax)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    st.pyplot(fig_pca)
    st.header("📊 Cluster Summary Table")
    cluster_summary = df_sampled.groupby('cluster')[features].mean().round(2)
    st.dataframe(cluster_summary)
    st.header("🎯 Cluster Size Distribution")
    cluster_counts = df_sampled['cluster'].value_counts().sort_index()
    fig_pie, ax = plt.subplots()
    ax.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig_pie)
    st.header("📥 Download Clustered Dataset")
    csv_export = df_sampled.to_csv(index=False).encode('utf-8')
    st.download_button("Download Clustered CSV", csv_export, "clustered_songs.csv", "text/csv")
else:
    st.warning("📂 Please upload your Spotify dataset CSV file to proceed.")
