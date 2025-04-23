import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE

# --- Page Config ---
st.set_page_config(layout="wide")
st.title("🔍 Interactive Clustering Comparison App")

# --- 0) High-Level Notes ---
st.markdown("""
## 💡 How This App Works
1. **Data & Preprocessing**  
2. **Algorithm Controls** (DBSCAN / KMeans / Agglo / t-SNE)  
3. **Silhouette Scores & Plots**  
4. **Raw Scatter & t-SNE Visuals**  
Use the **sidebar** to interactively explore clustering behavior.
""")

# --- 1) Data & Preprocessing ---
st.markdown("### 1️⃣ Data & Preprocessing")
st.markdown("""
- **Dataset Preview**:  
  Five columns: `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1–100)`  
- **Feature Selection**:  
  We choose the **three numeric** features (Age, Income, Spending Score) to capture continuous behavior.  
- **Scaling**:  
  Standardize each feature to *zero mean* and *unit variance*  
  so no feature dominates the distance calculation.
""")

# --- Data Loading / Upload ---
st.sidebar.header("📥 Data")
uploaded = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("Mall_Customers.csv")
st.write("#### 📋 Dataset Preview", df.head())

# --- Feature Selection ---
num_cols = st.sidebar.multiselect(
    "Select numeric features", 
    options=df.select_dtypes(include="number").columns.tolist(),
    default=["Age", "Annual Income (k$)", "Spending Score (1-100)"]
)
features = df[num_cols].dropna()
st.markdown(f"**Using features:** `{num_cols}`")

# --- Scale Features ---
scaler = StandardScaler()
X = scaler.fit_transform(features)

# --- 2) Auto-Tune DBSCAN Panel ---
st.sidebar.header("🔧 Auto-Tune DBSCAN")
do_tune = st.sidebar.checkbox("Run auto-tuning", value=False)
best_eps, best_min, best_score = None, None, -1
if do_tune:
    st.sidebar.write("Tuning over grid of `eps` and `min_samples`...")
    for eps in np.arange(0.2, 1.01, 0.1):
        for m in [3,5,8,10]:
            labels = DBSCAN(eps=eps, min_samples=m).fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters < 2: continue
            score = silhouette_score(X[labels!=-1], labels[labels!=-1])
            if score > best_score:
                best_score, best_eps, best_min = score, eps, m
    st.sidebar.success(f"Best: eps={best_eps:.2f}, min_samples={best_min}, score={best_score:.3f}")

# --- 3) Algorithm Controls ---
st.sidebar.header("⚙️ Algorithm Controls")
# DBSCAN
eps = st.sidebar.slider("eps (DBSCAN)", 0.1, 2.0, best_eps or 0.5, 0.05)
min_samples = st.sidebar.select_slider("min_samples (DBSCAN)", [2,3,5,8,10], value=best_min or 5)
# K-Means
n_k = st.sidebar.slider("n_clusters (K-Means)", 2, 10, 5)
# Agglomerative
n_a = st.sidebar.slider("n_clusters (Agglomerative)", 2, 10, 5)
# t-SNE
perp = st.sidebar.slider("perplexity (t-SNE)", 5, 50, 30)

st.markdown("### 2️⃣ Sidebar Controls Reference")
st.markdown("""
| Control                     | What it Does                                                      |
|-----------------------------|--------------------------------------------------------------------|
| **eps** (DBSCAN)            | How big a point's neighborhood is                                  |
| **min_samples** (DBSCAN)    | How many neighbors before a point becomes a core                   |
| **n_clusters** (K-Means/Agglo) | Forces that many clusters; see immediate splits/merges.         |
| **perplexity** (t-SNE)      | The number of nearest neighbors you are telling t-SNE to pay attention to.
""")

# --- Fit Clustering ---
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
km = KMeans(n_clusters=n_k, random_state=42).fit(X)
ag = AgglomerativeClustering(n_clusters=n_a).fit(X)
labels_db, labels_km, labels_ag = db.labels_, km.labels_, ag.labels_

# --- 4) Silhouette Scores Table ---
def sil_score(lbls):
    uniq = set(lbls)
    if len(uniq) < 2: return np.nan
    mask = lbls != -1
    if -1 in uniq:
        return silhouette_score(X[mask], lbls[mask])
    return silhouette_score(X, lbls)

scores = pd.DataFrame({
    "Method": ["DBSCAN","K-Means","Agglomerative"],
    "Silhouette": [
        sil_score(labels_db), 
        sil_score(labels_km), 
        sil_score(labels_ag)
    ]
})
st.markdown("### 3️⃣ Silhouette Scores")
st.write("Silhouette ranges from -1 (bad) to +1 (excellent). Higher → tighter, better-separated clusters.")
st.table(scores.style.format({"Silhouette":"{:.3f}"}))

# --- Silhouette Plot for DBSCAN ---
with st.expander("▶️ Show silhouette plot for DBSCAN"):
    if len(set(labels_db)) > 1:
        mask = labels_db != -1
        s_vals = silhouette_samples(X[mask], labels_db[mask])
        fig, ax = plt.subplots()
        y_lower = 10
        y_ticks = []
        for i, cl in enumerate(np.unique(labels_db[mask])):
            cl_s = np.sort(s_vals[labels_db[mask]==cl])
            size = cl_s.shape[0]
            y_upper = y_lower + size
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cl_s)
            y_ticks.append((y_lower + y_upper)/2)
            y_lower = y_upper + 10
        ax.set_ylabel("Cluster")
        ax.set_xlabel("Silhouette Coefficient")
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(np.unique(labels_db[mask]))
        st.pyplot(fig)
    else:
        st.write("Not enough clusters for a silhouette plot.")

# --- 5) Raw Scatter + Overlays ---
st.markdown("### 4️⃣ Raw Scatter Plots with DBSCAN Overlay")
st.markdown("Shows exactly which points DBSCAN calls core vs. noise.")
fig, axs = plt.subplots(1, 2, figsize=(14,5))
# Income vs Spending
if set(["Annual Income (k$)","Spending Score (1-100)"]).issubset(features.columns):
    axs[0].scatter(
        features["Annual Income (k$)"], 
        features["Spending Score (1-100)"], 
        c=labels_db, cmap="tab10", s=30
    )
    axs[0].set_title("Income vs Spending")
else:
    axs[0].text(0.5,0.5,"Missing features",ha="center")
# Age vs Income
if set(["Age","Annual Income (k$)"]).issubset(features.columns):
    axs[1].scatter(
        features["Age"], 
        features["Annual Income (k$)"], 
        c=labels_db, cmap="tab10", s=30
    )
    axs[1].set_title("Age vs Income")
else:
    axs[1].text(0.5,0.5,"Missing features",ha="center")
st.pyplot(fig)

# --- 6) t-SNE Cluster Comparisons ---
st.markdown("### 5️⃣ t-SNE Cluster Comparisons")
st.markdown("2D projections of all three methods for easy side-by-side inspection.")
tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
proj = tsne.fit_transform(X)
fig, axs = plt.subplots(1, 3, figsize=(18,5))
for ax, (lbls, name) in zip(axs, [(labels_db,"DBSCAN"),(labels_km,"K-Means"),(labels_ag,"Agglomerative")]):
    ax.scatter(proj[:,0], proj[:,1], c=lbls, cmap="tab10", s=20)
    ax.set_title(name)
st.pyplot(fig)

st.markdown("### 📋 Algorithm Comparison Table")
st.markdown("""
| PANEL         | Coloring Scheme                               | Key Takeaway                                                                                      |
|---------------|-----------------------------------------------|---------------------------------------------------------------------------------------------------|
| **DBSCAN**    | Each color = one DBSCAN cluster; dark blue = noise | You see the clusters found before where DBSCAN found dense cores; the rest are treated as noise. |
| **K-Means**   | Each color = one of _k_ equal-sized clusters  | You’ll spot _k_ roughly even-sized groups, because K-Means forced _k_ no matter what.             |
| **Agglomerative** | Each color = one of _k_ hierarchical clusters | You see a different partition into _k_ groups, often along different density/distance boundaries than K-Means. |
""")


# --- End Notes ---
st.markdown("""
---
🎯 **Takeaways**:
- **High silhouette** (e.g. 0.766) → tight, well-separated clusters  
- **Noise points** highlight extremes (VIPs, bargains, outliers)  
- **Interactive tuning** lets you find the sweet spot for your marketing segments
""")
