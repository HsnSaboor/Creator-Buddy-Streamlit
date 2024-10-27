import os
import streamlit as st
import asyncio
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from lxml import html
from urllib.parse import parse_qs, urlparse
import re
import requests
from PIL import Image
from io import BytesIO
import pytesseract
from colorthief import ColorThief
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from youtube_comment_downloader import YoutubeCommentDownloader
from textblob import TextBlob
import pandas as pd
import random

# Install Playwright once
os.system("pip install playwright")
os.system("playwright install")

st.set_page_config(page_title="YouTube Video Analyzer", page_icon="📊")

# Title and Instructions
st.title("📊 YouTube Video Analyzer")
st.write("Analyze YouTube video details, comments sentiment, and thumbnail color patterns.")
st.write("Enter a YouTube video URL or video ID to begin:")

# Video Input
video_input = st.text_input("YouTube Video URL or ID:")
if video_input.startswith("http"):
    parsed_url = urlparse(video_input)
    video_id = parse_qs(parsed_url.query).get("v", [None])[0]
else:
    video_id = video_input

# Helper functions
def analyze_thumbnail(video_id):
    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    response = requests.get(thumbnail_url)
    thumbnail_img = Image.open(BytesIO(response.content))

    # Display thumbnail
    st.image(thumbnail_img, caption="Thumbnail")

    # Extract color palette and dominant color
    color_thief = ColorThief(BytesIO(response.content))
    dominant_color = color_thief.get_color(quality=1)
    palette = color_thief.get_palette(color_count=6)
    
    # Display color palette
    with st.expander("Thumbnail Color Analysis"):
        st.write("**Dominant Color**:", dominant_color)
        st.write("**Color Palette:**")
        cols = st.columns(6)
        for idx, color in enumerate(palette):
            cols[idx].color_picker("", color=color, disabled=True)

    # Text extraction from thumbnail
    text = pytesseract.image_to_string(thumbnail_img)
    st.write("**Extracted Text from Thumbnail**")
    st.write(text.strip())

    # Generate heatmap data from thumbnail
    heatmap_data = generate_heatmap_data(thumbnail_img)
    st.pyplot(plot_heatmap(heatmap_data))

    return dominant_color, palette, text.strip()

def generate_heatmap_data(img):
    img = img.resize((50, 50))  # Resize for processing speed
    img_data = np.array(img)
    heatmap_data = np.mean(img_data[:, :, :3], axis=2)  # Average over RGB channels
    return heatmap_data

def plot_heatmap(data):
    plt.figure(figsize=(6, 6))
    sns.heatmap(data, cmap='viridis', square=True, cbar=False)
    plt.axis('off')
    return plt.gcf()

def get_comment_sentiment(comment_text):
    analysis = TextBlob(comment_text)
    return "Positive" if analysis.sentiment.polarity > 0 else "Negative"

def sentiment_summary(comments):
    sentiments = [get_comment_sentiment(comment["text"]) for comment in comments]
    pos_count = sentiments.count("Positive")
    neg_count = sentiments.count("Negative")
    return pos_count, neg_count

async def fetch_video_details(video_id):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(f"https://www.youtube.com/watch?v={video_id}", wait_until="domcontentloaded")
        content = await page.content()
        tree = html.fromstring(content)
        
        # Extract video details
        title = tree.xpath('//h1[@class="style-scope ytd-watch-metadata"]/yt-formatted-string/text()')
        title = title[0].strip() if title else "Title Not Found"
        
        views, publish_time, tags = "Views Not Found", "Publish Date Not Found", []
        info = tree.xpath('//yt-formatted-string[@id="info"]')
        if info:
            info_text = info[0].text_content().strip()
            views, publish_time, *tags = info_text.split("  ")
            tags = [tag.strip() for tag in tags if tag.strip()]
        
        likes = "Likes Not Found"
        like_button = await page.query_selector('//button[@title="I like this"]')
        if like_button:
            likes_attr = await like_button.get_attribute('aria-label')
            match = re.search(r'(\d[\d,]*)', likes_attr)
            likes = match.group(1) if match else likes
        
        # Description
        description = " ".join(element.text_content().strip() for element in tree.xpath('//yt-formatted-string[@id="description"]//span'))
        
        await browser.close()
        return title, int(views.replace(",", "")), publish_time, tags, int(likes.replace(",", "")), description

# Analyze comments
def analyze_comments(video_id):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments(video_id)
    beautified_comments = [{"author": comment['author'], "text": re.sub(r'<.*?>', '', comment['text'])} for comment in comments]
    comment_count = len(beautified_comments)

    # Sentiment summary
    pos_count, neg_count = sentiment_summary(beautified_comments)
    st.write("**Comment Sentiment Summary**")
    st.metric("Positive Comments", pos_count)
    st.metric("Negative Comments", neg_count)
    
    # Visualize comment sentiment distribution
    fig, ax = plt.subplots()
    ax.bar(['Positive', 'Negative'], [pos_count, neg_count], color=['#4CAF50', '#F44336'])
    st.pyplot(fig)

    return beautified_comments, comment_count

# Main Execution if Video ID is provided
if video_id:
    title, views, publish_time, tags, likes, description = asyncio.run(fetch_video_details(video_id))
    
    # Display video metadata
    st.header("Video Details")
    col1, col2, col3 = st.columns(3)
    col1.metric("Views", f"{views:,}")
    col2.metric("Likes", f"{likes:,}")
    col3.metric("Published Date", publish_time)
    st.subheader("Title")
    st.write(title)
    
    # Description in expander
    with st.expander("Video Description"):
        st.write(description)
    
    # Tags display
    with st.expander("Tags"):
        for tag in tags:
            st.markdown(f"[{tag}](https://www.youtube.com/hashtag/{tag[1:]})")

    # Thumbnail Analysis
    dominant_color, palette, thumbnail_text = analyze_thumbnail(video_id)
    
    # Comments Analysis
    beautified_comments, comment_count = analyze_comments(video_id)
    
    # Like-View and Comment-View Ratios as Progress Bars
    st.subheader("Engagement Ratios")
    like_view_ratio = likes / views if views else 0
    comment_view_ratio = comment_count / views if views else 0
    
    st.progress(int(like_view_ratio * 100), text="Like to View Ratio")
    st.progress(int(comment_view_ratio * 100), text="Comment to View Ratio")

    # Display comments
    st.subheader("Sample Comments")
    st.write("Below is a random selection of comments from the video:")
    sample_comments = random.sample(beautified_comments, min(5, len(beautified_comments)))
    for comment in sample_comments:
        st.write(f"**{comment['author']}**: {comment['text']}")