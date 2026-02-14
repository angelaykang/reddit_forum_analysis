import argparse
import os
import time
import json
import threading
import subprocess
from datetime import datetime

import numpy as np
import pymysql

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_DATABASE_NAME = os.environ.get("MYSQL_DATABASE", "reddit_forum")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_cmd(cmd: list[str]) -> int:
    log("RUN: " + " ".join(cmd))
    try:
        return subprocess.run(cmd, check=False).returncode
    except Exception as e:
        log(f"[ERROR] subprocess failed: {e}")
        return 1


def get_conn():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def parse_embedding(val):
    if val is None:
        return None
    if isinstance(val, (bytes, bytearray)):
        val = val.decode("utf-8", errors="ignore")
    s = str(val).strip()
    if not s:
        return None
    try:
        arr = json.loads(s)
        return np.asarray(arr, dtype=np.float32)
    except Exception:
        return None


def load_cluster_posts(conn):
    q = """
    SELECT id, subreddit, cleaned_title, cleaned_selftext, title, selftext, cluster_id, embedding
    FROM posts
    WHERE cluster_id IS NOT NULL AND embedding IS NOT NULL
    """
    with conn.cursor() as cur:
        cur.execute(q)
        rows = cur.fetchall()

    items = []
    for r in rows:
        emb = parse_embedding(r.get("embedding"))
        if emb is None:
            continue
        items.append(
            {
                "id": r.get("id"),
                "subreddit": r.get("subreddit") or "",
                "text": (
                    (r.get("cleaned_title") or r.get("title") or "")
                    + ". "
                    + (r.get("cleaned_selftext") or r.get("selftext") or "")
                ).strip(),
                "title": (r.get("cleaned_title") or r.get("title") or "").strip(),
                "cluster_id": int(r.get("cluster_id")),
                "emb": emb,
            }
        )
    return items


def compute_centroids(items, k):
    if not items:
        return None, None
    d = items[0]["emb"].shape[0]
    centroids = np.zeros((k, d), dtype=np.float32)
    counts = np.zeros((k,), dtype=np.int32)

    for it in items:
        cid = it["cluster_id"]
        if 0 <= cid < k:
            centroids[cid] += it["emb"]
            counts[cid] += 1

    for cid in range(k):
        if counts[cid] > 0:
            centroids[cid] /= float(counts[cid])

    return centroids, counts


def cluster_keywords_from_texts(texts, top_n=10):
    if not texts:
        return []
    vec = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        token_pattern=r"[a-zA-Z]{3,}",
    )
    X = vec.fit_transform(texts)
    scores = np.asarray(X.mean(axis=0)).ravel()
    idx = np.argsort(scores)[::-1][:top_n]
    inv = np.array(vec.get_feature_names_out())
    return inv[idx].tolist()


def nearest_posts_to_centroid(items, centroid, target_cluster, top_n=5):
    cluster_items = [it for it in items if it["cluster_id"] == target_cluster]
    if not cluster_items:
        return []

    M = np.stack([it["emb"] for it in cluster_items], axis=0)
    sims = cosine_similarity(M, centroid.reshape(1, -1)).ravel()
    order = np.argsort(-sims)[:top_n]
    return [cluster_items[i] for i in order]


def search_query_to_cluster(model, query_text, k):
    conn = get_conn()
    try:
        items = load_cluster_posts(conn)
    finally:
        conn.close()

    if not items:
        return None, "No clustered posts found. Run clustering_analysis.py first.", None

    centroids, counts = compute_centroids(items, k)
    if centroids is None:
        return None, "Failed to compute centroids.", None

    q_emb = model.encode([query_text])[0].astype(np.float32)
    sims = cosine_similarity(centroids, q_emb.reshape(1, -1)).ravel()

    for cid in range(k):
        if counts[cid] == 0:
            sims[cid] = -1e9

    best = int(np.argmax(sims))
    return best, None, (items, centroids, counts)


def print_cluster_view(cluster_id, items, centroids, counts, k, outdir, top_posts=5, q = None):

    print()
    print()
    print()
    print()
    print()
    print("==================================================================================================")
    print("|")
    print("|                                      Cluster Posts for", q)
    print("|")
    print("==================================================================================================")


    cluster_items = [it for it in items if it["cluster_id"] == cluster_id]
    texts = [it["text"] for it in cluster_items]
    keywords = cluster_keywords_from_texts(texts, top_n=10)

    subs = {}
    for it in cluster_items:
        subs[it["subreddit"]] = subs.get(it["subreddit"], 0) + 1
    subs_sorted = sorted(subs.items(), key=lambda x: -x[1])

    print(f"\nCluster {cluster_id+1} ({len(cluster_items)} posts)")
    print(" - Keywords:", ", ".join(keywords))
    print(" - Subreddit Occurrences:")
    for s, c in subs_sorted:
        print(f"   {s:<20} {c}")

    centroid = centroids[cluster_id]
    nearest = nearest_posts_to_centroid(items, centroid, cluster_id, top_n=top_posts)

    print("\n - Posts Nearest Centroid:")
    for it in nearest:
        title = it["title"] if it["title"] else "(no title)"
        print(f"     - [r/{it['subreddit']}] {title}")

    clusters_png = os.path.join(outdir, "clusters_2d.png")
    posts_png = os.path.join(outdir, "posts_per_cluster.png")
    print("\nGraphical representation:")
    print(f" - {clusters_png}")
    print(f" - {posts_png}\n")
    print()
    print()
    print()
    print()


def updater_loop(args, stop_event):
    interval_sec = args.min * 60
    cycle = 0

    while not stop_event.is_set():
        print()
        print()
        print()
        print()
        print()
        print("==================================================================================================")
        print("|")
        print("|                                      Data Updating.....")
        print("|")
        print("==================================================================================================")

        cycle += 1
        log(f"[UPDATE] cycle={cycle} start")

        cmd = ["python3", "reddit_forum_analysis.py", str(args.data_num)]
        for s in args.subreddit:
            cmd += ["--subreddit", s]
        if args.ocr:
            cmd += ["--ocr"]
        rc = run_cmd(cmd)
        if rc != 0:
            log(f"[ERROR] reddit_forum_analysis failed rc={rc}")

        print()
        print()
        print()
        print()
        print()
        print("==================================================================================================")
        print("|")
        print("|                                      Cluster Updating.....")
        print("|")
        print("==================================================================================================")

        if args.cluster_every > 0 and (cycle % args.cluster_every == 0):
            cmd2 = [
                "python3",
                "clustering_analysis.py",
                "--k",
                str(args.k),
                "--outdir",
                args.outdir,
            ]
            rc2 = run_cmd(cmd2)
            if rc2 != 0:
                log(f"[ERROR] clustering_analysis failed rc={rc2}")

        log(f"[SLEEP] {args.min} minutes")
        stop_event.wait(interval_sec)
        print()
        print()
        print()


def main():
    ap = argparse.ArgumentParser(description="Automation runner for Reddit forum analysis.")
    ap.add_argument("--min", type=int, required=True)
    ap.add_argument("--data_num", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--cluster-every", type=int, default=1)
    ap.add_argument("--outdir", type=str, default="plots")
    ap.add_argument("--subreddit", action="append", default=[])
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--no-bg", action="store_true")
    args = ap.parse_args()

    log(f"Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    stop_event = threading.Event()

    if not args.no_bg:
        th = threading.Thread(target=updater_loop, args=(args, stop_event), daemon=True)
        th.start()
        log("[BG] updater started")

    log("[READY] type a keyword/message to find nearest cluster. ':quit' to exit.")
    while True:
        try:

            q = input("search> ").strip()
        except (EOFError, KeyboardInterrupt):
            q = ":quit"

        if q in (":quit", ":q", "quit", "exit"):
            stop_event.set()
            log("[EXIT]")
            break
        if not q:
            continue

        cid, err, payload = search_query_to_cluster(model, q, args.k)
        if err:
            log(f"[WARN] {err}")
            log("Tip: run at least once: python3 clustering_analysis.py --k <K>")
            continue

        items, centroids, counts = payload
        log(f"[SEARCH] '{q}' -> nearest cluster {cid+1}")
        print_cluster_view(cid, items, centroids, counts, args.k, args.outdir, top_posts=5, q=q)

if __name__ == "__main__":
    main()
