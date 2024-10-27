import os
import streamlit as st
import asyncio
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from lxml import html
from urllib.parse import parse_qs, urlparse
import random
import re
import logging
import pandas as pd
from textblob import TextBlob
from youtube_comment_downloader import YoutubeCommentDownloader
import requests
from PIL import Image
import pytesseract
from colorthief import ColorThief
from io import BytesIO
import matplotlib.pyplot as plt

# Install Playwright using os.system (this should only be done once)
os.system("pip install playwright")
os.system("playwright install")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define Streamlit elements
st.title("YouTube Video Analysis")
st.write("Enter a YouTube video URL or ID to analyze")

# YouTube URL or ID input
video_input = st.text_input("YouTube Video URL or ID:")
if video_input.startswith("http"):
    parsed_url = urlparse(video_input)
    query_params = parse_qs(parsed_url.query)
    video_id = query_params.get("v", [None])[0]
else:
    video_id = video_input

# Helper functions
def convert_yt_redirect_to_normal_link(redirect_url):
    parsed_url = urlparse(redirect_url)
    query_params = parse_qs(parsed_url.query)
    return query_params.get("q", [''])[0]

def convert_yt_watch_to_full_link(watch_url):
    parsed_url = urlparse(watch_url)
    query_params = parse_qs(parsed_url.query)
    video_id = query_params.get('v', [''])[0]
    return f"https://www.youtube.com/watch?v={video_id}"

def convert_hashtag_to_link(hashtag):
    return f"[{hashtag}](https://www.youtube.com/hashtag/{hashtag[1:]})"

def analyze_thumbnail(video_id):
    # Download thumbnail
    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    thumbnail_response = requests.get(thumbnail_url)
    thumbnail_image = Image.open(BytesIO(thumbnail_response.content))
    st.image(thumbnail_image, caption="Video Thumbnail")

    # Analyze colors
    color_thief = ColorThief(BytesIO(thumbnail_response.content))
    dominant_color = color_thief.get_color(quality=1)
    palette = color_thief.get_palette(color_count=6)

    # Display color palette
    st.write("**Thumbnail Dominant Color**:", dominant_color)
    st.write("**Color Palette:**")
    for color in palette:
        st.color_picker("", color=color, key=f"color_{color}", disabled=True)

    # Extract text
    thumbnail_text = pytesseract.image_to_string(thumbnail_image)
    st.write("**Thumbnail Text Analysis:**", thumbnail_text.strip())

    return dominant_color, palette, thumbnail_text.strip()

def get_comment_sentiment(comment_text):
    analysis = TextBlob(comment_text)
    return "Positive" if analysis.sentiment.polarity > 0 else "Negative"

def analyze_comments_sentiment(beautified_comments, comment_count):
    positive_comments = sum(1 for comment in beautified_comments if get_comment_sentiment(comment['text']) == "Positive")
    return "Positive" if positive_comments > (comment_count / 2) else "Negative"

async def extract_video_data(video_id):
    async with async_playwright() as p:
        browser_type = "chromium"
        browser = await p[browser_type].launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        video_url = f"https://www.youtube.com/watch?v={video_id}"
        await page.goto(video_url, wait_until="domcontentloaded")

        # Extract video details
        content = await page.content()
        tree = html.fromstring(content)

        # Title
        title_element = tree.xpath('//h1[@class="style-scope ytd-watch-metadata"]/yt-formatted-string')
        title = title_element[0].text_content().strip() if title_element else "Title not found"
        st.header(title)

        # Publish time and views
        views, publish_time, tags = "Views not found", "Publish time not found", []
        info_element = tree.xpath('//yt-formatted-string[@id="info"]')
        if info_element:
            info_text = info_element[0].text_content().strip()
            views, publish_time, *tags = info_text.split('  ')
            tags = [tag.strip() for tag in tags if tag.strip()]

        st.write("**Publish Time:**", publish_time)
        st.write("**Views:**", views)

        # Likes
        likes = "Likes not found"
        like_button_selector = '//button[contains(@class, "yt-spec-button-shape-next") and @title="I like this"]'
        likes_element = await page.query_selector(like_button_selector)
        if likes_element:
            aria_label = await likes_element.get_attribute('aria-label')
            if aria_label:
                match = re.search(r'(\d[\d,]*)', aria_label)
                if match:
                    likes = int(match.group(1).replace(',', ''))
        st.write("**Likes:**", likes)

        # Description
        description_elements = tree.xpath('//ytd-text-inline-expander[@id="description-inline-expander"]//yt-attributed-string[@user-input=""]//span')
        description = " ".join(element.text_content().strip() for element in description_elements)
        st.write("**Description:**", description)

        # Comments analysis
        downloader = YoutubeCommentDownloader()
        comments = downloader.get_comments(video_id)
        beautified_comments = [{"author": comment['author'], "text": re.sub(r'<.*?>', '', comment['text'])} for comment in comments]
        comment_count = len(beautified_comments)
        st.write("**Number of Comments:**", comment_count)
        
        overall_sentiment = analyze_comments_sentiment(beautified_comments, comment_count)
        st.write("**Comment Sentiment Analysis:**", overall_sentiment)

        # Analyze Thumbnail
        dominant_color, palette, thumbnail_text = analyze_thumbnail(video_id)

        # Progress bars
        st.write("**Comment to View Ratio:**")
        st.progress(comment_count / int(views) if views != "Views not found" else 0)
        st.write("**Like to View Ratio:**")
        st.progress(int(likes) / int(views) if views != "Likes not found" and views != "Views not found" else 0)

# Run the analysis if video ID is provided
if video_id:
    asyncio.run(extract_video_data(video_id)