import asyncio
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from lxml import html, etree
from urllib.parse import parse_qs, urlparse
import random
import re
import logging
import pandas as pd
from itertools import islice
import xml.etree.ElementTree as ET
import datetime
import streamlit as st
from textblob import TextBlob
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
import requests
from PIL import Image
import pytesseract
from groq import Groq
from colorthief import ColorThief
import json
import os
from io import BytesIO
import math
from typing import List, Dict
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

os.system('pip install playwright')
os.system('playwright install')

client = Groq(
    api_key='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'
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Predefined user agents, resolutions, and browser configurations
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59 Safari/537.36"
]

RESOLUTIONS = [
    {"width": 1024, "height": 768},
    {"width": 1280, "height": 720},
    {"width": 1366, "height": 768},
    {"width": 1440, "height": 900},
    {"width": 1600, "height": 900}
]

BROWSERS = ["chromium"]

def extract_topics(text):
    system_message = """
    You are a topic extraction model. Your task is to extract the main and niche topics from the given text.
    The main topic should be a broad category that encompasses the overall content of the text.
    The niche topic should be a more specific sub-category within the main topic.
    The third topic should be a specific topic or theme that is mentioned in the text.
    Use only the description, title, and thumbnail text to determine the third topic.
    If two videos are in the same niche, they should have the same main and niche topics.
    Return the main, niche, and third topics as a JSON object with the following structure:
    {
        "main_topic": "main topic",
        "niche_topic": "niche topic",
        "third_topic": "third topic"
    }
    Example:
    Input: "Text: The video discusses the latest advancements in quantum computing and its applications in cryptography."
    Output: {"main_topic": "Technology", "niche_topic": "Quantum Computing", "third_topic": "Cryptography"}
    """
    user_message = f"Text: {text}"

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        model="llama-3.1-8b-instant",
        temperature=0.3,  # Lower temperature for less creativity
        max_tokens=1024,
        top_p=0.9,  # More deterministic output
        stop=None,
        stream=False,
        response_format={"type": "json_object"}
    )

    return json.loads(chat_completion.choices[0].message.content)

def parse_svg_heatmap(heatmap_svg, video_duration_seconds, svg_width=1000, svg_height=1000):
    if not heatmap_svg or heatmap_svg.strip() == "":
        logging.error("SVG heatmap content is empty or None")
        return []

    try:
        tree = ET.ElementTree(ET.fromstring(heatmap_svg))
        root = tree.getroot()
    except ET.ParseError as e:
        logging.error(f"Failed to parse SVG heatmap: {e}")
        logging.error(f"SVG content: {heatmap_svg}")
        return []

    heatmap_points = []

    for g in root.findall('.//{http://www.w3.org/2000/svg}g'):
        for defs in g.findall('.//{http://www.w3.org/2000/svg}defs'):
            for path in defs.findall('.//{http://www.w3.org/2000/svg}path'):
                d_attr = path.attrib.get('d', '')

                coordinates = re.findall(r'[MC]([^MC]+)', d_attr)

                for segment in coordinates:
                    points = segment.strip().replace(',', ' ').split()
                    for i in range(0, len(points), 2):
                        x = float(points[i])
                        y = float(points[i + 1])

                        duration_seconds = (x / svg_width) * video_duration_seconds
                        attention = 100 - (y / svg_height) * 100

                        heatmap_points.append({'Attention': attention, 'duration': duration_seconds})

    return heatmap_points

def analyze_heatmap_data(heatmap_points: List[Dict[str, float]], threshold: float = 2.25) -> Dict[str, any]:
    if not heatmap_points or not all(isinstance(point, dict) and 'Attention' in point and 'duration' in point for point in heatmap_points):
        return {}

    total_attention = sum(point['Attention'] for point in heatmap_points)
    average_attention = total_attention / len(heatmap_points)

    significant_rises = []
    significant_falls = []
    rise_start = None
    fall_start = None

    for i, point in enumerate(heatmap_points):
        attention = point['Attention']
        duration = point['duration']

        if attention > average_attention + threshold:
            if rise_start is None:
                rise_start = duration
            if i == len(heatmap_points) - 1 or heatmap_points[i + 1]['Attention'] <= average_attention + threshold:
                significant_rises.append({'start': rise_start, 'end': duration})
                rise_start = None

        if attention < average_attention - threshold:
            if fall_start is None:
                fall_start = duration
            if i == len(heatmap_points) - 1 or heatmap_points[i + 1]['Attention'] >= average_attention - threshold:
                significant_falls.append({'start': fall_start, 'end': duration})
                fall_start = None

    return {
        'average_attention': average_attention,
        'significant_rises': significant_rises,
        'significant_falls': significant_falls,
        'total_rises': len(significant_rises),
        'total_falls': len(significant_falls)
    }

def fetch_transcript(video_id: str):
    try:
        return YouTubeTranscriptApi.get_transcript(video_id)
    except TranscriptsDisabled:
        print("Transcripts are disabled for this video.")
        return None
    except Exception as e:
        print(f"Error fetching default transcript: {e}")

        try:
            available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            if 'en' in available_transcripts:
                return available_transcripts.find_transcript(['en']).fetch()
            else:
                first_transcript = next(available_transcripts).fetch()
                print("No English transcript found, using the first available.")
                return first_transcript
        except Exception as fallback_error:
            print(f"Error fetching alternative transcripts: {fallback_error}")
            return None

def get_significant_transcript_sections(transcript: List[Dict[str, any]], analysis_data: Dict[str, any]):
    significant_sections = {'rises': [], 'falls': []}

    for rise in analysis_data['significant_rises']:
        rise_transcript = [entry for entry in transcript if rise['start'] <= entry['start'] <= rise['end']]
        significant_sections['rises'].append(rise_transcript)

    for fall in analysis_data['significant_falls']:
        fall_transcript = [entry for entry in transcript if fall['start'] <= entry['start'] <= fall['end']]
        significant_sections['falls'].append(fall_transcript)

    return significant_sections

def duration_to_seconds(duration):
    parts = duration.split(':')
    if len(parts) == 2:  # MM:SS
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    elif len(parts) == 3:  # HH:MM:SS
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds

def convert_yt_redirect_to_normal_link(redirect_url):
    parsed_url = urlparse(redirect_url)
    query_params = parse_qs(parsed_url.query)
    return query_params.get('q', [''])[0]

def convert_yt_watch_to_full_link(watch_url):
    parsed_url = urlparse(watch_url)
    query_params = parse_qs(parsed_url.query)
    video_id = query_params.get('v', [''])[0]
    return f"https://www.youtube.com/watch?v={video_id}"

def convert_hashtag_to_link(hashtag):
    return f"[{hashtag}](https://www.youtube.com/hashtag/{hashtag[1:]})"

def analyze_thumbnail(video_id):
    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    thumbnail_response = requests.get(thumbnail_url)
    thumbnail_image = Image.open(BytesIO(thumbnail_response.content))

    color_thief = ColorThief(BytesIO(thumbnail_response.content))
    dominant_color = color_thief.get_color(quality=1)
    palette = color_thief.get_palette(color_count=6)

    thumbnail_text = pytesseract.image_to_string(thumbnail_image)

    return dominant_color, palette, thumbnail_text.strip()

def get_comment_sentiment(comment_text):
    analysis = TextBlob(comment_text)
    return "Positive" if analysis.sentiment.polarity > 0 else "Negative"

def analyze_comments_sentiment(beautified_comments, comment_count):
    positive_comments = sum(1 for comment in beautified_comments if get_comment_sentiment(comment['text']) == "Positive")
    overall_sentiment = "Positive" if positive_comments > (comment_count / 2) else "Negative"
    return overall_sentiment

def normalize_youtube_url(url):
    parsed_url = urlparse(url)
    video_id = None

    if parsed_url.netloc == 'youtu.be':
        video_id = parsed_url.path[1:]
    elif parsed_url.netloc in ['www.youtube.com', 'youtube.com', 'm.youtube.com']:
        query_params = parse_qs(parsed_url.query)
        video_id = query_params.get('v', [''])[0]

    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    else:
        raise ValueError("Invalid YouTube URL")

async def extract_video_data(video_url):
    logging.info(f"Extracting video data for video URL: {video_url}")
    normalized_url = normalize_youtube_url(video_url)
    video_id = parse_qs(urlparse(normalized_url).query)['v'][0]

    async with async_playwright() as p:
        browser_type = random.choice(BROWSERS)
        browser = await getattr(p, browser_type).launch(
            headless=True,
            args=["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage", "--disable-extensions", "--disable-plugins"]
        )

        context = await browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            viewport=random.choice(RESOLUTIONS),
            locale="en-US",
            ignore_https_errors=True,
            java_script_enabled=True,
            bypass_csp=True
        )

        await context.route("**/*", lambda route: route.abort() if route.request.resource_type in ["video", "audio", "font"] else route.continue_())

        page = await context.new_page()

        await page.goto(normalized_url, wait_until="domcontentloaded", timeout=60000)

        if "m.youtube.com" in page.url:
            await page.goto(normalized_url.replace("m.youtube.com", "www.youtube.com"), wait_until="networkidle")

        expand_selector = 'tp-yt-paper-button#expand'

        try:
            await page.wait_for_selector(expand_selector, timeout=8000)
            expand_button = await page.query_selector(expand_selector)
            if expand_button:
                await expand_button.click()
        except PlaywrightTimeoutError:
            logging.warning("Expand button not found.")

        await page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
        await page.wait_for_timeout(10000)

        content = await page.content()
        tree = html.fromstring(content)

        for elem in tree.xpath('//head | //style | //script'):
            elem.getparent().remove(elem)

        title_element = tree.xpath('//h1[@class="style-scope ytd-watch-metadata"]/yt-formatted-string')
        title = title_element[0].text_content().strip() if title_element else "Title not found"

        info_element = tree.xpath('//yt-formatted-string[@id="info"]')
        if info_element:
            info_text = info_element[0].text_content().strip()
            views, publish_time, *tags = info_text.split('  ')
            tags = [tag.strip() for tag in tags if tag.strip()]
        else:
            views, publish_time, tags = "Views not found", "Publish time not found", []

        if views != "Views not found":
            views = int(re.sub(r'\D', '', views))

        like_button_selector = '//button[contains(@class, "yt-spec-button-shape-next") and @title="I like this"]'
        likes_element = await page.query_selector(like_button_selector)
        likes = "Likes not found"
        if likes_element:
            aria_label = await likes_element.get_attribute('aria-label')
            if aria_label:
                match = re.search(r'(\d[\d,]*)', aria_label)
                if match:
                    likes = int(match.group(1).replace(',', ''))

        heatmap_svg = await extract_heatmap_svgs(page)

        duration_element = tree.xpath('//span[@class="ytp-time-duration"]')
        duration = duration_element[0].text_content().strip() if duration_element else "Duration not found"
        duration_to_seconds_value = duration_to_seconds(duration)

        heatmap_points = []
        heatmap_analysis = {}
        significant_transcript_sections = {}

        if heatmap_svg and heatmap_svg != "Heatmap SVG not found":
            heatmap_points = parse_svg_heatmap(heatmap_svg, duration_to_seconds_value)
            heatmap_analysis = analyze_heatmap_data(heatmap_points)

            transcript = fetch_transcript(video_id)
            significant_transcript_sections = get_significant_transcript_sections(transcript, heatmap_analysis) if transcript else {}

        description_elements = tree.xpath('//ytd-text-inline-expander[@id="description-inline-expander"]//yt-attributed-string[@user-input=""]//span[@class="yt-core-attributed-string yt-core-attributed-string--white-space-pre-wrap"]')
        description_parts = []
        links = set()

        for element in description_elements:
            text_content = element.text_content().strip()
            if text_content and text_content not in description_parts:
                description_parts.append(text_content)
            for link in element.xpath('.//a'):
                link_text = link.text_content().strip()
                link_href = link.get('href')
                if link_text and link_href and link_href not in links:
                    if link_href.startswith('https://www.youtube.com/redirect'):
                        link_href = convert_yt_redirect_to_normal_link(link_href)
                    elif link_href.startswith('/watch?'):
                        link_href = convert_yt_watch_to_full_link(link_href)
                    elif link_href.startswith('/hashtag/'):
                        link_text = convert_hashtag_to_link(link_text)  # This line is incorrect, fix it
                        link_href = f"https://www.youtube.com{link_href}"
                    description_parts.append(f"[{link_text}]({link_href})")
                    links.add(link_href)

        description = ' '.join(description_parts)

        comments_xpath = "//div[@id='leading-section']//h2[@id='count']//yt-formatted-string[@class='count-text style-scope ytd-comments-header-renderer']/span[1]/text()"
        comment_count_element = tree.xpath(comments_xpath)
        if comment_count_element:
            comment_count_text = comment_count_element[0]
            logging.info(f"Extracted comment count text: {comment_count_text}")
            if comment_count_text.isdigit():
                comment_count = int(comment_count_text)
            else:
                comment_count = int(''.join(filter(str.isdigit, comment_count_text)))
        else:
            logging.warning("Comment count element not found")
            comment_count = 1

        print(f"Total number of comments: {comment_count}")

        comments = await extract_comments(video_id)
        beautified_comments = [{"author": comment['author'], "text": re.sub(r'<.*?>', '', comment['text'])} for comment in comments]

        overall_sentiment = analyze_comments_sentiment(beautified_comments, comment_count)

        comment_to_view_ratio = comment_count / views * 100 if views else 0
        like_to_views_ratio = likes / views * 100 if views else 0
        comment_to_like_ratio = comment_count / likes * 100 if likes else 0

        logging.info(f"Getting AI Analysis for video ID: {video_id}")

        dominant_color, palette, thumbnail_text = analyze_thumbnail(video_id)

        # Extract topics from title, description, and thumbnail text
        combined_text = f"{title}\n{description}\n{thumbnail_text}"
        topics = extract_topics(combined_text)

        # Ensure topics is a dictionary with the expected keys
        main_topic = topics.get('main_topic', 'N/A')
        niche_topic = topics.get('niche_topic', 'N/A')
        third_topic = topics.get('third_topic', 'N/A')

        logging.info(f"Creating Output Markdown for video ID: {video_id}")

        markdown_content = f"""
# {title}

## Video Statistics

- **Video ID:** {video_id}
- **Title:** {title}
- **Views:** {views}
- **Publish Time:** {publish_time}
- **Tags:** {', '.join(tags)}
- **Likes:** {likes}
- **Duration:** {duration}
- **Topics:** Main: {main_topic}, Niche: {niche_topic}, Third: {third_topic}
- **Description:** {description}
- **Duration in Seconds:** {duration_to_seconds_value}

## Overall Sentiment

- **Overall Comment Sentiment:** {overall_sentiment}

## Thumbnail Analysis

- **Dominant Color:** {dominant_color}
- **Palette Colors:** {palette}
- **Thumbnail Text:** {thumbnail_text}

## Ratios

- **Comment to View Ratio:** {comment_to_view_ratio:.2f}%
- **Views to Like Ratio:** {like_to_views_ratio:.2f}%
- **Comment to Like Ratio:** {comment_to_like_ratio:.2f}%

## Links

{', '.join(links)}

"""

        if heatmap_svg and heatmap_svg != "Heatmap SVG not found":
            markdown_content += f"""
## Heatmap Analysis

- **Average Attention:** {heatmap_analysis['average_attention']:.2f}%

- **Total Rises:** {heatmap_analysis['total_rises']}
- **Total Falls:** {heatmap_analysis['total_falls']}

- **Significant Rises:**
  {',  '.join([f"{rise['start']}s to {rise['end']}s" for rise in heatmap_analysis['significant_rises']])}

- **Significant Falls:**
  {',  '.join([f"{fall['start']}s to {fall['end']}s" for fall in heatmap_analysis['significant_falls']])}

## Significant Transcript Sections

### Rises
{', '.join([f"- **{rise['start']}s to {rise['end']}s**: {', '.join([entry['text'] for entry in rise_transcript])}" for rise, rise_transcript in zip(heatmap_analysis['significant_rises'], significant_transcript_sections['rises'])])}

### Falls
{', '.join([f"- **{fall['start']}s to {fall['end']}s**: {', '.join([entry['text'] for entry in fall_transcript])}" for fall, fall_transcript in zip(heatmap_analysis['significant_falls'], significant_transcript_sections['falls'])])}

## Heatmap SVG

- **Heatmap Graph Points:** {heatmap_points}

```svg
{heatmap_svg}

"""
        else:
            markdown_content += """
## Heatmap Analysis

- **Heatmap not available for this video.**

"""

        markdown_content += f"""
## Comments

- **Total No of Comments:** {comment_count}

| Author               | Comment
"""
        for comment in beautified_comments:
            markdown_content += f"""
| {comment['author']} | {comment['text']}
"""

        return markdown_content

# Sidebar for user input
st.sidebar.title("YouTube Video Analysis")
video_url = st.sidebar.text_input("Enter YouTube video URL")
analyze_button = st.sidebar.button("Analyze Video")

# Main layout structure
st.title("YouTube Video Data Analysis")
st.write("This app extracts and visualizes insights from YouTube videos.")

if analyze_button and video_url:
    # Extract video details here
    video_id = video_url.split("v=")[1]
    
    st.subheader("Video Details")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display video thumbnail
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        st.image(thumbnail_url, caption="Video Thumbnail", width=200)
        
    with col2:
        # Display video metadata
        st.write("**Video URL:**", video_url)
        st.write("**Video ID:**", video_id)

    # Display Heatmap Data with Graph
    def plot_attention_heatmap(heatmap_points):
        # Create DataFrame from heatmap points
        heatmap_df = pd.DataFrame(heatmap_points)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(heatmap_df.pivot("duration", "Attention"), cmap="YlGnBu", ax=ax)
        st.pyplot(fig)

    # Example heatmap data (replace with real data extraction)
    heatmap_points = parse_svg_heatmap(heatmap_svg, video_duration_seconds)

    st.subheader("Attention Heatmap")
    plot_attention_heatmap(heatmap_points)

    # Display Transcript Analysis with Highlights
    st.subheader("Transcript Analysis")

    # Assuming `significant_sections` is a dict with 'rises' and 'falls'
    significant_sections = get_significant_transcript_sections(transcript, analysis_data)

    # Display each section
    with st.expander("Significant Rises in Attention"):
        for rise in significant_sections['rises']:
            st.write(" ".join([entry['text'] for entry in rise]))

    with st.expander("Significant Falls in Attention"):
        for fall in significant_sections['falls']:
            st.write(" ".join([entry['text'] for entry in fall]))

    # Display Comments Sentiment Analysis
    def plot_sentiment_pie(positive, negative):
        labels = ['Positive', 'Negative']
        sizes = [positive, negative]
        colors = ['#4CAF50', '#FF5252']
        explode = (0.1, 0)  

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=140)
        st.pyplot(fig1)

    positive_comments = 120  # Replace with actual analysis result
    negative_comments = 80  # Replace with actual analysis result

    st.subheader("Comment Sentiment Analysis")
    plot_sentiment_pie(positive_comments, negative_comments)

    # Display Dominant Colors in Thumbnail
    st.subheader("Thumbnail Color Analysis")

    dominant_color, palette, thumbnail_text = analyze_thumbnail(video_id)
    st.write("**Dominant Color:**", dominant_color)

    st.write("**Color Palette:**")
    for color in palette:
        st.color_picker(label=f"Color", value=f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}')

# Helper functions
def parse_svg_heatmap(heatmap_svg, video_duration_seconds, svg_width=1000, svg_height=1000):
    # Implementation of SVG parsing
    pass

def get_significant_transcript_sections(transcript, analysis_data):
    # Implementation of transcript section extraction
    pass

def analyze_thumbnail(video_id):
    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    thumbnail_response = requests.get(thumbnail_url)
    thumbnail_image = Image.open(BytesIO(thumbnail_response.content))

    color_thief = ColorThief(BytesIO(thumbnail_response.content))
    dominant_color = color_thief.get_color(quality=1)
    palette = color_thief.get_palette(color_count=6)

    thumbnail_text = pytesseract.image_to_string(thumbnail_image)

    return dominant_color, palette, thumbnail_text.strip()

def get_comment_sentiment(comment_text):
    analysis = TextBlob(comment_text)
    return "Positive" if analysis.sentiment.polarity > 0 else "Negative"

def analyze_comments_sentiment(beautified_comments, comment_count):
    positive_comments = sum(1 for comment in beautified_comments if get_comment_sentiment(comment['text']) == "Positive")
    overall_sentiment = "Positive" if positive_comments > (comment_count / 2) else "Negative"
    return overall_sentiment

def fetch_transcript(video_id: str):
    try:
        return YouTubeTranscriptApi.get_transcript(video_id)
    except TranscriptsDisabled:
        print("Transcripts are disabled for this video.")
        return None
    except Exception as e:
        print(f"Error fetching default transcript: {e}")

        try:
            available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            if 'en' in available_transcripts:
                return available_transcripts.find_transcript(['en']).fetch()
            else:
                first_transcript = next(available_transcripts).fetch()
                print("No English transcript found, using the first available.")
                return first_transcript
        except Exception as fallback_error:
            print(f"Error fetching alternative transcripts: {fallback_error}")
            return None

def extract_comments(video_id):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(f'https://www.youtube.com/watch?v={video_id}', sort_by=SORT_BY_RECENT)
    return list(islice(comments, 100))  # Limit to 100 comments for example

def main():
    if analyze_button and video_url:
        video_id = video_url.split("v=")[1]
        transcript = fetch_transcript(video_id)
        comments = extract_comments(video_id)
        beautified_comments = [{"author": comment['author'], "text": re.sub(r'<.*?>', '', comment['text'])} for comment in comments]
        comment_count = len(beautified_comments)
        overall_sentiment = analyze_comments_sentiment(beautified_comments, comment_count)

        st.subheader("Comments")
        for comment in beautified_comments:
            st.write(f"**{comment['author']}:** {comment['text']}")

        st.subheader("Overall Sentiment")
        st.write(f"**Overall Comment Sentiment:** {overall_sentiment}")

if __name__ == "__main__":
    main()