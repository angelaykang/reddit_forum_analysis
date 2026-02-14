# DSCI 560: Lab 5 - Data Collection for Domain-Specific Chatbot

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
<br><br><br>

## Manual Running the Scripts for Collecting data and clustering analysis
**Requirements:** Python 3.8+, `pip install -r requirements.txt`, MySQL running. No Reddit API credentials needed.

### Data Collection
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
