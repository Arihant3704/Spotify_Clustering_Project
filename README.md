
# Spotify Song Clustering PRO 🎵

This project implements hierarchical clustering on Spotify songs dataset based on audio features and popularity using unsupervised machine learning.

---

## 📊 Project Summary

- Dataset Source: Kaggle Spotify Dataset (1921-2020)
- Features used: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, popularity.
- Applied: StandardScaler, PCA, Hierarchical Clustering (Ward Linkage)
- Built interactive visualization app using Streamlit.

---

## 🚀 Features of Web App

- Upload your own Spotify dataset (`tracks.csv`)
- Dynamic feature selection for clustering
- Subsampling control for memory efficiency
- Hierarchical clustering with interactive dendrogram
- PCA cluster visualization
- Cluster summary table and pie chart
- Export clustered dataset as CSV

---

## 🛠 Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/spotify-clustering-project.git
cd spotify-clustering-project
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App

```bash
streamlit run app.py
```

---

## 🌐 Deployment

This app is ready to be deployed on **Streamlit Cloud**:

- Upload files (`app.py`, `requirements.txt`, `tracks.csv`) to GitHub.
- Connect GitHub repo to [Streamlit Cloud](https://streamlit.io/).
- Deploy app with `app.py` as main file.

---

## 📂 Dataset

You can download the original dataset from Kaggle here:  
https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db

---

## 📞 Contact

Created by: **Arihant’s ML Pipeline** 🚀  
Contact for help or improvements!

---
