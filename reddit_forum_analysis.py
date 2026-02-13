#!/usr/bin/env python3
# Reddit Forum Analysis — fetch, preprocess, store in MySQL.
# Usage: python reddit_forum_analysis.py <num_posts> [--subreddit SUB ...] [--ocr]
import os
import re
import sys
import time
import hashlib
import argparse
from datetime import datetime, timezone
from html import unescape
from collections import Counter

MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_DATABASE_NAME = os.environ.get("MYSQL_DATABASE", "reddit_forum")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
DEFAULT_SUBREDDITS = ["datasciencecareers", "datasciencejobs", "cscareerquestions", "experienceddevs"]
TOTAL_TIMEOUT_SECONDS = 400
SORT_OPTIONS = ["hot", "new", "top", "rising"]
OCR_IMAGE_TIMEOUT_SECONDS = 10
JSON_USER_AGENT = "Mozilla/5.0 (compatible; RedditForumAnalysis/1.0)"
JSON_DELAY_SECONDS = 2.0
JSON_LIMIT_PER_REQUEST = 100
try:
    import requests
except ImportError:
    requests = None
try:
    from bs4 import BeautifulSoup
    try:
        import warnings
        from bs4 import MarkupResemblesLocatorWarning
        warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
    except ImportError:
        pass
except ImportError:
    BeautifulSoup = None
try:
    import pymysql
    pymysql.cursors.DictCursor
except ImportError:
    pymysql = None


def _fetch_posts_via_json_one_page(subreddit_name, sort, after=None, limit=100):
    if requests is None:
        raise ImportError("requests required — pip install requests")
    base = "https://www.reddit.com/r/{}/{}.json".format(subreddit_name.strip().lower(), sort)
    params = {"limit": min(int(limit or 100), 100)}
    if after:
        params["after"] = after
    r = requests.get(
        base,
        params=params,
        headers={"User-Agent": JSON_USER_AGENT},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    children = (data.get("data") or {}).get("children") or []
    next_after = (data.get("data") or {}).get("after")
    posts = []
    for c in children:
        d = (c.get("data") or {})
        if d.get("promoted") or (d.get("title") or "").lower().startswith("[promoted]"):
            continue
        created = d.get("created_utc")
        if created is not None:
            created = datetime.fromtimestamp(float(created), tz=timezone.utc)
        else:
            created = datetime.now(timezone.utc)
        author = d.get("author") or "[deleted]"
        posts.append({
            "reddit_id": d.get("id") or "",
            "subreddit": subreddit_name,
            "title": (d.get("title") or ""),
            "selftext": (d.get("selftext") or ""),
            "author": author,
            "created_utc": created,
            "score": int(d.get("score") or 0),
            "num_comments": int(d.get("num_comments") or 0),
            "url": (d.get("url") or ""),
        })
    return posts, next_after


def fetch_posts_via_json(subreddit_name, num_posts, timeout_sec=400):
    seen, posts, start = set(), [], time.time()
    last_err = None
    for sort in SORT_OPTIONS:
        if len(posts) >= num_posts or (time.time() - start) > timeout_sec:
            break
        time.sleep(JSON_DELAY_SECONDS)
        after = None
        while len(posts) < num_posts and (time.time() - start) <= timeout_sec:
            try:
                batch, next_after = _fetch_posts_via_json_one_page(
                    subreddit_name, sort, after=after, limit=JSON_LIMIT_PER_REQUEST
                )
            except Exception as e:
                last_err = e
                break
            for p in batch:
                rid = p.get("reddit_id")
                if rid and rid not in seen:
                    seen.add(rid)
                    posts.append(p)
                if len(posts) >= num_posts:
                    break
            if not next_after or not batch:
                break
            after = next_after
            time.sleep(JSON_DELAY_SECONDS)
    if not posts and last_err is not None:
        raise last_err
    return posts[:num_posts]


def fetch_posts_from_subreddits_via_json(subreddit_names, num_posts, timeout_sec=400):
    if not subreddit_names:
        return []
    names = [n.strip().lower() for n in subreddit_names if n and n.strip()]
    per_sub = max(1, (num_posts + len(names) - 1) // len(names))
    timeout_per_sub = max(30, timeout_sec // len(names))
    all_posts, seen_ids = [], set()
    start = time.time()
    for name in names:
        if len(all_posts) >= num_posts:
            break
        remaining_time = max(0.0, timeout_sec - (time.time() - start))
        if remaining_time <= 0:
            break
        want = min(per_sub, num_posts - len(all_posts))
        batch_timeout = min(timeout_per_sub, remaining_time)
        batch = fetch_posts_via_json(name, want, timeout_sec=batch_timeout)
        for p in batch:
            rid = p.get("reddit_id")
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                all_posts.append(p)
            if len(all_posts) >= num_posts:
                break
    return all_posts[:num_posts]


PROMOTED = [
    r"\[promoted\]", r"\[ad\]", r"\[promo\]", r"sponsored", r"promoted post",
    r"advertisement", r"^\s*\[ad\]", r"this is an? ad(vertisement)?\b",
]


def strip_html(text):
    if not text or not isinstance(text, str):
        return ""
    text = unescape(text)
    if BeautifulSoup:
        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    else:
        text = re.sub(r"<[^>]+>", " ", text)
    return text


def remove_special(text):
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"[^\w\s.,!?\-']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_promoted(title, selftext):
    s = ((title or "") + " " + (selftext or "")).lower()
    return any(re.search(p, s, re.I) for p in PROMOTED)


def mask_username(name):
    if not name or name == "[deleted]":
        return "user_deleted"
    return "user_" + hashlib.sha256(name.encode()).hexdigest()[:8]


def extract_keywords(text, top_n=15):
    if not text or not isinstance(text, str):
        return ""
    text = remove_special(strip_html(text))
    words = re.findall(r"[a-z0-9']+", text.lower())
    sw = {
        "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "while", "for", "to", "of", "in", "on", "at",
        "by", "with", "from", "as", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that",
        "these", "those", "i", "you", "we", "they", "he", "she", "them", "his", "her", "our", "your", "their",
        "can", "could", "should", "would", "may", "might", "will", "just", "not", "no", "yes", "do", "does", "did",
        "have", "has", "had", "having", "so", "than", "too", "very", "about", "into", "over", "under", "again",
    }
    words = [w for w in words if w.isalnum() and w not in sw and len(w) > 2]
    return ",".join(w for w, _ in Counter(words).most_common(top_n))


def ocr_from_url(url):
    # requires pytesseract, Pillow, and tesseract
    if not url or not isinstance(url, str):
        return ""
    url_lower = url.lower().strip()
    if not url_lower.startswith("http"):
        return ""
    if not any(x in url_lower for x in ("i.redd.it", "imgur", ".png", ".jpg", ".jpeg", ".gif")):
        return ""
    if requests is None:
        raise ImportError("requests required for OCR — pip install requests")
    try:
        import pytesseract
        from PIL import Image
        from io import BytesIO
    except Exception as e:
        raise ImportError("pytesseract and Pillow required for OCR — pip install pytesseract Pillow") from e
    try:
        r = requests.get(url, timeout=OCR_IMAGE_TIMEOUT_SECONDS)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content))
        return pytesseract.image_to_string(img) or ""
    except Exception:
        return ""


def preprocess_post(post, use_ocr=False):
    title = (post.get("title") or "").strip()
    selftext = (post.get("selftext") or "").strip()
    if is_promoted(title, selftext):
        return None
    ct = remove_special(strip_html(title))
    cs = remove_special(strip_html(selftext))
    combined = f"{ct} {cs}".strip()
    if not combined:
        return None
    created = post.get("created_utc")
    if isinstance(created, datetime):
        created_utc = created
    else:
        try:
            created_utc = datetime.fromtimestamp(float(created), tz=timezone.utc)
        except (TypeError, ValueError):
            created_utc = datetime.now(timezone.utc)
    img_text = ""
    combined_for_keywords = combined
    if use_ocr:
        try:
            img_text = ocr_from_url(post.get("url") or "")
        except Exception:
            img_text = ""
        if img_text and img_text.strip():
            cleaned_img = remove_special(strip_html(img_text.strip()))
            if cleaned_img:
                combined_for_keywords = f"{combined} {cleaned_img}".strip()

    kw = extract_keywords(combined_for_keywords)
    return {
        "reddit_id": post.get("reddit_id", ""), "subreddit": post.get("subreddit", ""),
        "title": title, "selftext": selftext, "author_masked": mask_username(post.get("author") or "[deleted]"),
        "created_utc": created_utc, "score": int(post.get("score") or 0), "num_comments": int(post.get("num_comments") or 0),
        "url": post.get("url") or "", "cleaned_title": ct, "cleaned_selftext": cs,
        "keywords": kw, "topics": kw.replace(",", "; "),
        "image_extracted_text": img_text,
    }


def preprocess_posts(posts, use_ocr=False):
    out = []
    for p in posts:
        row = preprocess_post(p, use_ocr=use_ocr)
        if row is not None:
            out.append(row)
    return out


def _require_mysql():
    if pymysql is None:
        print("pymysql required — pip install pymysql")
        sys.exit(1)
    try:
        conn = pymysql.connect(
            host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASSWORD,
            charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor,
        )
        conn.close()
    except Exception as e:
        print("MySQL connection failed:", e)
        print("Check that the server is running and MYSQL_USER/MYSQL_PASSWORD are set.")
        sys.exit(1)


def get_conn():
    if pymysql is None:
        raise ImportError("pymysql required — pip install pymysql")
    return pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASSWORD,
        charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor,
    )


def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE_NAME}`")
        cur.execute(f"USE `{MYSQL_DATABASE_NAME}`")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id INT AUTO_INCREMENT PRIMARY KEY, reddit_id VARCHAR(20) UNIQUE NOT NULL,
                subreddit VARCHAR(100) NOT NULL, title TEXT NOT NULL, selftext TEXT,
                author_masked VARCHAR(50) NOT NULL, created_utc DATETIME NOT NULL,
                score INT DEFAULT 0, num_comments INT DEFAULT 0, url VARCHAR(500),
                cleaned_title TEXT, cleaned_selftext TEXT, keywords TEXT, topics TEXT,
                image_extracted_text TEXT,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_subreddit (subreddit), INDEX idx_created (created_utc)
            )
        """)
        # add column if we're upgrading from an older schema
        cur.execute("SHOW COLUMNS FROM posts LIKE 'image_extracted_text'")
        if cur.fetchone() is None:
            cur.execute("ALTER TABLE posts ADD COLUMN image_extracted_text TEXT")
    conn.commit()


def store_posts(conn, rows):
    if not rows:
        return 0
    sql = """
    INSERT INTO posts (reddit_id, subreddit, title, selftext, author_masked, created_utc, score, num_comments, url,
                       cleaned_title, cleaned_selftext, keywords, topics, image_extracted_text)
    VALUES (%(reddit_id)s, %(subreddit)s, %(title)s, %(selftext)s, %(author_masked)s, %(created_utc)s, %(score)s, %(num_comments)s, %(url)s,
            %(cleaned_title)s, %(cleaned_selftext)s, %(keywords)s, %(topics)s, %(image_extracted_text)s)
    ON DUPLICATE KEY UPDATE title=VALUES(title), selftext=VALUES(selftext), author_masked=VALUES(author_masked),
    score=VALUES(score), num_comments=VALUES(num_comments), cleaned_title=VALUES(cleaned_title),
    cleaned_selftext=VALUES(cleaned_selftext), keywords=VALUES(keywords), topics=VALUES(topics),
    image_extracted_text=VALUES(image_extracted_text)
    """
    with conn.cursor() as cur:
        for r in rows:
            d = {k: r.get(k, "") for k in (
                "reddit_id", "subreddit", "title", "selftext", "author_masked", "created_utc", "score", "num_comments", "url",
                "cleaned_title", "cleaned_selftext", "keywords", "topics", "image_extracted_text"
            )}
            ct = d.get("created_utc")
            if hasattr(ct, "tzinfo") and ct.tzinfo is not None:
                d["created_utc"] = ct.replace(tzinfo=None)
            cur.execute(sql, d)
    conn.commit()
    return len(rows)


def run_pipeline(num_posts, subreddits=None, use_ocr=False):
    subreddits = subreddits or DEFAULT_SUBREDDITS
    print("[1] Fetching posts from r/{}...".format(", r/".join(subreddits)))
    try:
        posts = fetch_posts_from_subreddits_via_json(subreddits, num_posts, timeout_sec=TOTAL_TIMEOUT_SECONDS)
        print("    Fetched {} posts.".format(len(posts)))
    except Exception as e:
        print("    Error:", e)
        return False, str(e)
    if not posts:
        print("    No posts returned.")
        return False, "no posts"

    print("[2] Preprocessing (clean text, filter ads, keywords{})...".format(", OCR" if use_ocr else ""))
    try:
        rows = preprocess_posts(posts, use_ocr=use_ocr)
        print("    {} posts after preprocessing.".format(len(rows)))
    except Exception as e:
        print("    Error:", e)
        return False, str(e)
    if not rows:
        print("    No posts after preprocessing.")
        return False, "no rows"

    print("[3] Writing to database...")
    try:
        conn = get_conn()
        try:
            ensure_schema(conn)
            n = store_posts(conn, rows)
            print("    {} rows inserted/updated.".format(n))
        finally:
            conn.close()
    except Exception as e:
        print("    Database error:", e)
        if "1045" in str(e) or "Access denied" in str(e):
            print("    Set MYSQL_PASSWORD if needed (e.g. export MYSQL_PASSWORD=...).")
        return False, str(e)
    print("Done.")
    return True, "ok"


def main():
    p = argparse.ArgumentParser(description="Fetch Reddit posts, preprocess, and store in MySQL.")
    p.add_argument("num_posts", type=int, help="number of posts to fetch")
    p.add_argument("--subreddit", action="append", dest="subreddits", metavar="SUB",
                    help="subreddit name(s); default: " + ", ".join(DEFAULT_SUBREDDITS))
    p.add_argument("--ocr", action="store_true", help="extract text from images via OCR (requires pytesseract, Pillow)")
    args = p.parse_args()
    _require_mysql()
    subreddits = args.subreddits if args.subreddits else DEFAULT_SUBREDDITS
    success, _ = run_pipeline(
        args.num_posts,
        subreddits=subreddits,
        use_ocr=args.ocr,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
