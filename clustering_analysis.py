import os
import re
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from collections import Counter
import pymysql
import nltk
import ssl
# Disabling SSL verification for NLTK download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

# MySQL DB Config
MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_DATABASE_NAME = os.environ.get("MYSQL_DATABASE", "reddit_forum")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")

# Embedding Model Config
# We are using all-MiniLM-L6-v2 from sentence-transformers instead of doc2vec
# This is because the transformers-based model uses attention to better capture context from text data
# all-MiniLM-L6-v2 produces 384 dimension vectors
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Defining stopwords for later cluster keyword extraction
# Since our pre-processing already stripped apostrophes
# we will strip apostrophes from the nltk stopwords and add them back to the stopwords list
_nltk_stopwords = set(stopwords.words('english'))
STOPWORDS = _nltk_stopwords | {word.replace("'", "") for word in _nltk_stopwords}


# Database Helper Functions
def get_conn():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE_NAME,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

def load_posts(conn):
    # Return df of posts with cleaned text fields
    query = """
    SELECT id, reddit_id, subreddit, cleaned_title, cleaned_selftext, keywords, title, selftext
    FROM posts
    WHERE cleaned_title IS NOT NULL
    ORDER BY id
    """

    # Manual fetch
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    
    # Construct DataFrame from raw data
    columns = ['id', 'reddit_id', 'subreddit', 'cleaned_title', 'cleaned_selftext', 'keywords', 'title', 'selftext']
    df = pd.DataFrame(rows, columns=columns)
    df['id'] = pd.to_numeric(df['id'], errors='coerce')
    
    df['cleaned_title'] = df['cleaned_title'].fillna("")
    df['cleaned_selftext'] = df['cleaned_selftext'].fillna("")

    print(f"Loaded {len(df)} posts from MySQL DB.")
    return df

# We will later be adding vector embeddings and clusters
# So this function will prematurely create columns in the DB for both
def create_embedding_cluster_columns(conn):
    with conn.cursor() as cur:
        # Checking both columns
        for col, dtype in [
            ('embedding', 'JSON'),
            ('cluster_id', 'INT DEFAULT NULL')
        ]:
            cur.execute(f"SHOW COLUMNS FROM posts LIKE '{col}'")

            # Adding column is nonexistent
            if cur.fetchone() is None:
                cur.execute(f"ALTER TABLE posts ADD COLUMN {col} {dtype}")
                print(f"Added column '{col} to posts table.")

    conn.commit()

def store_results(conn, df, embeddings, labels):
    # Storing vector embeddings and cluster labels back to the MySQL DB and a pd df
    with conn.cursor() as cur:
        for i in range(len(df)):
            # Get the id from position 0 (first column from SQL query)
            post_id = int(df.iloc[i, 0])
            cur.execute(
                "UPDATE posts SET embedding = %s, cluster_id = %s WHERE id = %s",
                (json.dumps(embeddings[i].tolist()), int(labels[i]), post_id)
            )

    conn.commit()
    print(f"Stored vector embeddings and cluster labels for {len(df)} posts.")


# Generating Vector Emeddings
def build_documents(df):
    # We are going to combine the title and selftext into a single string for every post so they can be embedded together
    titles = df['cleaned_title'].fillna("")
    selftexts = df['cleaned_selftext'].fillna("")

    docs = (titles + ". " + selftexts).str.strip(". ").to_list()
    return docs

def embed_documents(docs):
    # Getting vector embeddings using our set model and sentence-transformers
    print(f"Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"\nEmbedding {len(docs)} documents.")
    embeddings = model.encode(docs, show_progress_bar=True) # , batch_size=64) # I believe that by default it uses batch_size=32

    print(f"Embedding shape: {embeddings.shape}\n")

    return embeddings


# Clustering vectorized posts
def find_best_k(embeddings, k_min=2, k_max=15):
    # Silhouette scores measure cluster quality by calculating how similar a point is to it's cluster
    # (+1: well-clustered -> 0: overlapping clusters -> -1: incorrect assignment)
    inertias = []
    silhouette_scores = []
    k_values = range(k_min, k_max + 1)

    # Fitting clusters
    print(f"Evaluating K = [{k_min} -> {k_max}]\n")
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)

        # Recording metrics
        inertias.append(km.inertia_)
        
        # Handle case where KMeans produces fewer clusters than requested
        unique_labels = len(np.unique(labels))
        if unique_labels < 2:
            # Silhouette score requires at least 2 clusters
            # We will use negative inertia as a fallback only if really needed
            silhouette = -km.inertia_ / 1000.0
            print(f"K = {k} | Inertia = {km.inertia_:.4f} | Silhouette = N/A (only {unique_labels} cluster found)")
        else:
            silhouette = silhouette_score(embeddings, labels)
            print(f"K = {k} | Inertia = {km.inertia_:.4f} | Silhouette = {silhouette:.4f}")
        
        silhouette_scores.append(silhouette)

    # The best index is found by the highest silhouette score
    best_idx = int(np.argmax(silhouette_scores))
    best_k = list(k_values)[best_idx]
    print(f"\nBest K = {best_k} | Silhouette = {silhouette_scores[best_idx]:.4f}")

    return best_k, list(k_values), inertias, silhouette_scores

def run_kmeans(embeddings, k):
    # Fitting final kmeans model
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    return labels, km


# Analysis Functions
def cluster_keywords(df, labels, k, top_n=10):
    # Finding the top keywords
    df = df.copy()
    df['cluster'] = labels
    cluster_keywords = {}

    # Iterating through the clusters
    for cluster_id in range(k):
        subset = df[df['cluster'] == cluster_id]
        text = " ".join(
            (subset['cleaned_title'] + " " + subset['cleaned_selftext']).to_list()
        ).lower()

        # Regex to find words and removing stopwords
        tokens = re.findall(r"[a-z]{3,}", text)
        words = [t for t in tokens if t not in STOPWORDS]

        # Using collections.Counter to find most common words
        cluster_keywords[cluster_id] = Counter(words).most_common(top_n)
    
    return cluster_keywords

def nearest_to_centroid(embeddings, labels, km, n=5):
    closest = {}

    # Iterating over clusters (centroids)
    for cluster_id, centroid in enumerate(km.cluster_centers_):
        mask = np.where(labels == cluster_id)[0]

        if len(mask) == 0:
            closest[cluster_id] = []
            continue

        # Finding distances from centroid and selecting closest n
        distances = np.linalg.norm(embeddings[mask] - centroid, axis=1)
        order = np.argsort(distances)[:n]
        closest[cluster_id] = mask[order].tolist()

    return closest


# Plotting functions
def plot_elbow_silhouettes(k_values, inertias, silhouette_scores, best_k, path):
    # Plotting both the kmeans elbow plot and the silhouettes associated
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Inertia plot
    ax1.plot(k_values, inertias, 'o-', color='red')
    ax1.axvline(best_k, color='red', alpha=0.5, label=f'K={best_k}')
    ax1.set_xlabel("K (num clusters)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Inertias")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    # Silhouette plot
    ax2.plot(k_values, silhouette_scores, 's-', color='green')
    ax2.axvline(best_k, color='red', alpha=0.5, label=f'K={best_k}')
    ax2.set_xlabel("K (num clusters)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouettes")
    ax2.grid(True, alpha=0.25)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"\nInertia/Silhouette Plot saved to {path}")

def plot_clusters_2d(embeddings, labels, k, cluster_keywords, path):
    # Using PCA to reduce embeddings to 2 dimensions and then plotting clusters
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(embeddings)

    # Getting color mapping based on k value
    cmap = plt.colormaps.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(12, 8))
    for cluster_id in range(k):
        mask = labels == cluster_id
        top_words = ", ".join(word for word, _ in cluster_keywords.get(cluster_id, [])[:5])
        label = f"Cluster {cluster_id+1}: {top_words}"
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            c=[cmap(cluster_id)],
            label=label,
            alpha=0.5,
            s=25,
            linewidths=0.25
        )
    
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(f"Reddit Post Clusters in 2D PCA Projection (K={k})")
    ax.grid(True, alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Cluster Plot saved to {path}")

def plot_posts_per_cluster(labels, k, cluster_keywords, path):
    # Bar plot of number of posts per cluster
    counts = Counter(labels)
    cluster_ids = list(range(k))
    sizes = [counts.get(cluster_id, 0) for cluster_id in cluster_ids]
    bar_labels = [", ".join(w for w, _ in cluster_keywords.get(cluster_id, [])[:3]) for cluster_id in cluster_ids]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(cluster_ids, sizes, color=plt.cm.tab10(np.linspace(0, 1, k)), edgecolor="white")
    ax.set_xticks(cluster_ids)
    ax.set_xticklabels([f"C{cluster+1}\n{bar_label}" for cluster, bar_label in zip(cluster_ids, bar_labels)], rotation=45, ha="center")

    ax.set_ylabel("Number of Posts")
    ax.set_title("Cluster Sizes with Top Keywords")

    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(size), ha="center", va="bottom")
    
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Posts per Cluster Plot saved to {path}\n")


# Main
def main():
    # Get CLI arguments
    parser = argparse.ArgumentParser(description="Cluster Posts")
    parser.add_argument("--k", type=int, default=None, help="Fixed number of clusters (skip auto-select)")
    parser.add_argument("--min-k", type=int, default=2, help="Min K for auto-select")
    parser.add_argument("--max-k", type=int, default=15, help="Max K for auto-select")
    parser.add_argument("--outdir", type=str, default="plots", help="Directory to save plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load Posts
    conn = get_conn()
    create_embedding_cluster_columns(conn)
    df = load_posts(conn)
    if df.empty:
        print(f"No posts found. Run reddit_forum_analysis.py first.")
    
    # Embed
    docs = build_documents(df)
    embeddings = embed_documents(docs)

    # Find K
    if args.k:
        k = args.k
        k_values = None
        inertias = None
        silhouettes = None
    else:
        k, k_values, inertias, silhouette_scores = find_best_k(embeddings, k_min=args.min_k, k_max=args.max_k)
    
    # Cluster posts
    labels, km = run_kmeans(embeddings, k)
    df['cluster_id'] = labels
    keywords_map = cluster_keywords(df, labels, k)
    nearest = nearest_to_centroid(embeddings, labels, km)

    # Plots
    if k_values is not None:
        plot_elbow_silhouettes(
            k_values,
            inertias,
            silhouette_scores,
            k,
            os.path.join(args.outdir, 'elbow_silhouettes.png')
        )

    plot_clusters_2d(
        embeddings,
        labels,
        k,
        keywords_map,
        os.path.join(args.outdir, 'clusters_2d.png')
    )

    plot_posts_per_cluster(
        labels,
        k,
        keywords_map,
        os.path.join(args.outdir, 'posts_per_cluster.png')
    )

    # Printing summary and centroid-closest posts
    print(f"Posts Nearest Centroids:\n")
    for cluster_id in range(k):
        cluster_df = df[df["cluster_id"] == cluster_id]
        keywords = ", ".join(word for word, _ in keywords_map.get(cluster_id, []))

        print(f"""
Cluster {cluster_id+1} ({len(cluster_df)} posts)
 - Keywords: {keywords}
 - Subreddit Occurrences: {cluster_df['subreddit'].value_counts().to_string()}

 - Posts Nearest Centroid:""")
        for idx in nearest.get(cluster_id, []):
            row = df.iloc[idx]
            title = row['cleaned_title']
            print(f"     - [r/{row['subreddit']}] {title}")
        print()

    # Storing results
    store_results(conn, df, embeddings, labels)
    conn.close()

    print("Done.")

if __name__ == "__main__":
    main()
