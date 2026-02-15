# DSCI 560: Lab 5

### Project Overview

This lab builds a simple Reddit analysis pipeline: collect posts, clean and store them in MySQL, embed and cluster them, and visualize the clusters. Then you wrap it in an automation + interactive CLI so the database updates on a schedule, and you can type a query to find the closest cluster and see representative posts and plots.

### Team: pylovers

| Name | USC ID |
|------|--------|
| Dylan Chen | 6984540266 |
| Angela Kang | 8957777203 |
| Vincent-Daniel Yun | 4463771151 |

### Dependencies
    requests>=2.28.0
    beautifulsoup4>=4.12.0
    pymysql>=1.1.0
    numpy>=1.24.0
    pandas>=2.0.0
    matplotlib>=3.7.0
    scikit-learn>=1.3.0
    sentence-transformers>=2.2.0
    nltk>=3.8.0

### Tools and Libraries
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
Please check [detailed detup](https://github.com/angelaykang/reddit_forum_analysis/blob/main/SETUP.md) for detailed setup instructions.
<br><br>

# Automation Run Command
This script runs an automated loop that periodically pulls new Reddit posts into MySQL and reruns clustering, so the database stays up to date. In parallel, it provides an interactive CLI where your query is embedded and matched to the nearest cluster centroid, then prints that cluster’s summary, representative posts, and plot paths.

    python3 main.py --min 1 --data_num 500 --k 6 --cluster-every 1 --outdir plot


`--min 1`: Background update interval in minutes. Every 1 minute, the script reruns data collection (and clustering depending on --cluster-every).

`--data_num 500`: Number of Reddit posts to fetch each update cycle (passed to reddit_forum_analysis.py).

`--k 6`: Number of clusters to use for K-means (passed to clustering_analysis.py --k), and also the number of centroids used for interactive query matching.

`--cluster-every 1`: Run clustering once every N update cycles. 1 means cluster every cycle; 2 would mean cluster every other cycle, etc.

`--outdir plot`: Output directory where clustering plots are saved (e.g., plot/clusters_2d.png, plot/posts_per_cluster.png).


<br><br>

# Manually Run Commands for Collecting data and clustering analysis
**Requirements:** Python 3.8+, `pip install -r requirements.txt`, MySQL running. No Reddit API credentials needed.

### Data Collection
**Basic data collection run:**
```bash
python3 reddit_forum_analysis.py 500
python3 reddit_forum_analysis.py 5000
```
**Expected screen when you run** `python3 reddit_forum_analysis.py 500`
    
    [1] Fetching posts from r/datasciencecareers, r/datasciencejobs, r/cscareerquestions, r/experienceddevs...
        Fetched 500 posts.
    [2] Preprocessing (clean text, filter ads, keywords)...
        499 posts after preprocessing.
    [3] Writing to database...
        499 rows inserted/updated.
    Done.
<br>

**Options:**
```bash
python3 reddit_forum_analysis.py 500 --subreddit cscareerquestions --subreddit experienceddevs
python3 reddit_forum_analysis.py 500 --ocr
```
- `--subreddit SUB` — Subreddit(s) to use (repeat for multiple).
- `--ocr` — Extract text from post images via Pytesseract (requires `pytesseract`, `Pillow`, and system Tesseract, e.g. `sudo apt install tesseract-ocr`).
<br><br><br>

### Clustering analysis
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

**Expected screen when you run** `python3 clustering_analysis.py --k 6`

    Loaded 501 posts from MySQL DB.
    Loading model: all-MiniLM-L6-v2
    
    Embedding 501 documents.
    Batches: 100%|███████████████████| 16/16 [00:02<00:00,  7.73it/s]
    Embedding shape: (501, 384)
    
    Cluster Plot saved to plots/clusters_2d.png
    Posts per Cluster Plot saved to plots/posts_per_cluster.png
    
    Posts Nearest Centroids:
    
    
    Cluster 1 (59 posts)
     - Keywords: code, like, work, time, use, software, https, would, using, tools
     - Subreddit Occurrences: subreddit
    experienceddevs       43
    cscareerquestions     13
    datasciencejobs        2
    datasciencecareers     1
    
     - Posts Nearest Centroid:
         - [r/experienceddevs] Hot take for discussion strong architecture patterns work equally well for AI and Juniors.
         - [r/experienceddevs] It isn't the tool, but the hands why the AI displacement narrative gets it backwards
         - [r/experienceddevs] AI-assisted coding and the true Bottlenecks in Software Development
         - [r/cscareerquestions] Is it foolish to avoid using AI coding agents?
         - [r/experienceddevs] Do you think there will be a breaking point where decreasing code quality becomes a problem, outside of engineering?
    
    
    Cluster 2 (96 posts)
     - Keywords: job, like, years, work, career, get, experience, time, really, people
     - Subreddit Occurrences: subreddit
    cscareerquestions     51
    datasciencecareers    16
    experienceddevs       16
    datasciencejobs       13
    .
    .
    .

