# Reddit Forum Analysis — Setup & Requirements (Sections 1–3)

This document contains the **setup steps and requirements for running the scripts** (per assignment 1a). The project implements **sections 1–3** (Initial Setup, Data Collection/Storage, Data Preprocessing).

---

## 1) Initial Setup

### a) Tools / Libraries

- **Assignment:** You may use requests/selenium, beautifulsoup4, and a MySQL database. Linux OS (Ubuntu) / Python script should be used. Document setup steps/requirements in the document you submit. Do not spend much time on installation; invest in concepts and improvising.
- **What we use:**
  - **requests** — Used for HTTP (fetching Reddit public .json and images for optional OCR). *(Assignment: requests.)*
  - **beautifulsoup4** — Used to remove HTML tags from post title and selftext during preprocessing. *(Assignment: beautifulsoup4.)*
  - **MySQL** — Used for storage via PyMySQL; schema in `db/schema.sql`; script creates database and table if missing. *(Assignment: MySQL database.)*
  - **Linux (Ubuntu) / Python script** — Script is Python 3, run as `python3 reddit_forum_analysis.py <num_posts>`; developed for Ubuntu/Linux.
- We do **not** use Selenium. Reddit data is fetched via **requests** to Reddit’s public .json. The assignment allows a different system if it gives career-beneficial experience; using requests + BeautifulSoup gives experience with HTTP, web scraping, and HTML/text cleaning.
- **Other:** `pymysql` (MySQL driver). Optional: `pytesseract`, `Pillow` (OCR only).

**Installation (Ubuntu/Linux):** Minimal setup per assignment.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

MySQL: `sudo apt install mysql-server` (or equivalent). Set `MYSQL_PASSWORD` (and optionally `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_DATABASE`) if needed.

### b) Resource

- **Assignment:** You may use BeautifulSoup or the Praw API to scrape Reddit. If you use a different API, document why the alternative is better than the suggested interfaces.
- **Our choice:** We use **BeautifulSoup** for preprocessing (stripping HTML from title and selftext). For **fetching** we use Reddit’s **public .json** API via **requests** (e.g. `https://www.reddit.com/r/<subreddit>/hot.json`) instead of PRAW. **Why this alternative:** (1) No Reddit API registration or approval wait; (2) same post data for our use case; (3) simpler setup and fewer dependencies; (4) experience with REST-style endpoints and pagination. Preprocessing still uses BeautifulSoup as suggested.
- **References (per assignment):**
  - Scrape Reddit on Python: https://brightdata.com/blog/web-data/how-to-scrape-reddit-python  
  - Praw API: https://medium.com/analytics-vidhya/scraping-reddit-using-python-reddit-api-wrapper-praw-5c275e34a8f4  
  - BeautifulSoup4 Guide: https://www.datacamp.com/tutorial/scraping-reddit-python-scrapy  

---

## 2) Data Collection / Storage

- **Assignment:** Select a topic on Reddit; script must take the number of posts to fetch as input, fetch them, and store them in the database after preprocessing. API has a threshold of 1000 posts or a timeout of 60 seconds. Code must handle requests of size 5000 or 400 secs by calling the API multiple times without running out of bounds, ensuring all results are fetched correctly and the request doesn’t fail. Read about timeout and max limit and modify code to handle large requests.
- **Topic selected:** Examples in assignment: r/tech, r/cybersecurity. We selected **careers in data science and tech**, with default subreddits **r/datasciencecareers**, **r/datasciencejobs**, **r/cscareerquestions**, **r/experienceddevs**. Override with `--subreddit SUB` (repeat for multiple).
- **Script input:** The script **takes the number of posts to fetch as an input** (required positional argument): `python3 reddit_forum_analysis.py <num_posts>`, e.g. `500` or `5000`.
- **Fetch and store after preprocessing:** Pipeline is **fetch → preprocess → store**. We fetch posts, preprocess them, then **store them in the database after preprocessing**; only preprocessed rows are written to the DB.
- **Large requests (5000, 400 sec):** We **call the API multiple times** (multiple pages per sort, multiple sort orders: hot, new, top, rising; multiple subreddits). Reddit’s public .json returns up to **100** items per request; we use a **30s timeout per request** (under 60s) and **400s total** (`TOTAL_TIMEOUT_SECONDS = 400`). We **do not run out of bounds** (paginate with `after`; respect total timeout). We **ensure all results are fetched correctly** (dedup by `reddit_id`). We **handle failures** (try/except, timeouts). See `reddit_forum_analysis.py`: `fetch_posts_via_json`, `fetch_posts_from_subreddits_via_json`, `JSON_LIMIT_PER_REQUEST`, `TOTAL_TIMEOUT_SECONDS`.

---

## 3) Data Preprocessing (assignment “4) Preprocess the data…”)

- **Assignment:** Preprocess by removing HTML tags, special characters, and irrelevant content (promoted messages and advertisements). Transform into a suitable format: converting timestamps, masking usernames for data privacy. Identify keywords and topics from messages and store them as additional fields in the database, along with the actual messages. For images: use Pytesseract/OCR to extract text, store as additional fields; consider this text when identifying keywords and topics.
- **Where implemented:** `reddit_forum_analysis.py` — `preprocess_post`, `preprocess_posts`; helpers: `strip_html`, `remove_special`, `is_promoted`, `mask_username`, `extract_keywords`, `ocr_from_url`.

| Requirement | How it is satisfied |
|-------------|---------------------|
| Remove HTML tags | `strip_html()` — BeautifulSoup (or regex fallback). |
| Remove special characters | `remove_special()` — keeps letters, digits, spaces, `.,!?-'`. |
| Remove promoted messages and advertisements | `is_promoted()` — patterns e.g. `[promoted]`, `[ad]`, `sponsored`, `advertisement`; such posts return `None` and are not stored. |
| Converting timestamps | `created_utc` — datetime (UTC). |
| Masking usernames (data privacy) | `mask_username()` → `author_masked` (hashed). |
| Identify keywords and topics; store as additional fields **along with the actual messages** | `extract_keywords()` → `keywords`, `topics`; DB stores these plus `title`, `selftext`, `cleaned_title`, `cleaned_selftext`. |
| Images: Pytesseract/OCR; extract text; store as additional field | `ocr_from_url()` with `--ocr`; stored in `image_extracted_text`. |
| Consider image text when identifying keywords and topics | When `--ocr` is set, image-extracted text is appended to `combined_for_keywords` before `extract_keywords()`. |

**Pytesseract:** https://pypi.org/project/pytesseract/ — used as in PyPI docs; optional dependency.

---

## 4) Forum Analysis & Clustering Algorithms

- **Assignment:** Convert messages into vector embeddings, cluster documents based on content, identify keywords per cluster, and visualize K clusters with message content.
- **Where implemented:** `clustering_analysis.py` reads pre-processed posts from the MySQL DB, generates embeddings, clusters with KMeans, and produces visualizations.
- **Embedding Model choice:** We use **sentence-transformers** (`all-MiniLM-L6-v2`) instead of doc2vec. Why?
  - (1) It produces 384-dimensional dense vectors that capture semantic meaning more accurately, especially on short texts.
  - (2) It is pretrained on 1B+ sentence pairs.
  - (3) The transformers-based attention mechanism can better capture context when compared to doc2vec.
- **References:**
  - sentence-transformers: https://www.sbert.net/
  - all-MiniLM-L6-v2 model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- **Algorithm:** **KMeans** (scikit-learn). We chose hard clustering over soft clustering because it produces clearer visualizations and our centroid analysis is best enabled with KMeans. The topics of posts amongst our forums are broad, vast, and use technical jargon, which could greatly hinder the effectiveness of soft clustering.
- **Selecting K:** When `--k` is not provided, the script evaluates K = 2 through 15 using both the **inertia** and **silhouette analysis**, automatically selecting the K with the highest silhouette score.
- **Keyword extraction:** For each cluster, (title + selftext) is tokenized, NLTK stopwords (including apostrophe-stripped variants) are filtered out, and the most frequent terms are reported.
- **Visualization (3 plots saved to `plots/`):**
  - `elbow_silhouettes.png` — Elbow curve (inertia vs. K) and silhouette score vs. K, with chosen K marked.
  - `clusters_2d.png` — PCA projection of embeddings to 2D, colored by cluster, with top keywords per cluster in the legend.
  - `posts_per_cluster.png` — Bar chart of post counts per cluster with top keywords labeled.
- **DB storage:** Cluster assignments (`cluster_id`) and embedding vectors (`embedding`) are written back to the `posts` table.
- **References:**
  - Scikit-Learn Clustering: https://scikit-learn.org/stable/modules/clustering.html
  - NLTK: https://www.nltk.org/

---

## 5) Running the Scripts

**Requirements:** Python 3.8+, `pip install -r requirements.txt`, MySQL running. No Reddit API credentials needed.

**Basic data collection run:**
```bash
python3 reddit_forum_analysis.py 500
python3 reddit_forum_analysis.py 5000
```

**Options:**
```bash
python3 reddit_forum_analysis.py 500 --subreddit cscareerquestions --subreddit experienceddevs
python3 reddit_forum_analysis.py 500 --ocr
```
- `--subreddit SUB` — Subreddit(s) to use (repeat for multiple).
- `--ocr` — Extract text from post images via Pytesseract (requires `pytesseract`, `Pillow`, and system Tesseract, e.g. `sudo apt install tesseract-ocr`).

**Clustering analysis:**
```bash
# Search for best K with silhouette analysis (K=2-15)
python3 clustering_analysis.py

# Choose a specific K
python3 clustering_analysis.py --k 6

# Input K search range and output path
python3 clustering_analysis.py --min-k 3 --max-k 10 --outdir results/plots
```
- `--k K` - Use a fixed number of clusters (skips auto-selection).
- `--min-k` / `--max-k` - Range for automatic K selection (default 2–15).
- `--outdir DIR` - Directory for saved plots (default: `plots/`).

**MySQL (optional):**
```bash
export MYSQL_HOST="localhost" MYSQL_USER="root" MYSQL_PASSWORD="your_password" MYSQL_DATABASE="reddit_forum"
# Or: mysql -u root -p < db/schema.sql
```

