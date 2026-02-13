-- Reddit forum posts schema (optional manual setup; script can create DB/table automatically)
-- Run: mysql -u root -p < db/schema.sql

CREATE DATABASE IF NOT EXISTS reddit_forum;
USE reddit_forum;

CREATE TABLE IF NOT EXISTS posts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    reddit_id VARCHAR(20) UNIQUE NOT NULL,
    subreddit VARCHAR(100) NOT NULL,
    title TEXT NOT NULL,
    selftext TEXT,
    author_masked VARCHAR(50) NOT NULL,
    created_utc DATETIME NOT NULL,
    score INT DEFAULT 0,
    num_comments INT DEFAULT 0,
    url VARCHAR(500),
    cleaned_title TEXT,
    cleaned_selftext TEXT,
    keywords TEXT,
    topics TEXT,
    image_extracted_text TEXT,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_subreddit (subreddit),
    INDEX idx_created (created_utc)
);
