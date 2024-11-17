import asyncio
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from lxml import html, etree
from urllib.parse import parse_qs, urlparse
import random
import regex as re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from fastcounter import Counter
import pandas as pd
from itertools import islice
import xml.etree.ElementTree as ET
from textblob import TextBlob
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
import requests
import cv2
import pytesseract
from groq import Groq
from colorthief import ColorThief
import orjson
from io import BytesIO
import math
from typing import List, Dict, Optional, Any
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, Transcript, TranscriptList
import spacy
import dask.dataframe as dd
import uvloop
import streamlit as st
from PIL import Image  # Importing Image from PIL

os.system('playwright install')

# Configure uvloop for faster asyncio event loops
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

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

def detect_ctas(Transcript):
    # List of common CTA phrases and patterns
    cta_patterns = [
        r'\bsubscribe\b', r'\bfollow\b', r'\bcheck\s+the\s+link\b', r'\bclick\s+here\b',
        r'\bvisit\s+our\s+website\b', r'\blearn\s+more\b', r'\bwatch\s+the\s+video\b',
        r'\bjoin\s+our\s+community\b', r'\bsign\s+up\b', r'\bdownload\s+now\b',
        r'\bget\s+started\b', r'\bcontact\s+us\b', r'\blike\s+and\s+share\b',
        r'\bleave\s+a\s+comment\b', r'\bsupport\s+us\b', r'\bbuy\s+now\b',
        r'\border\s+today\b', r'\bexplore\s+more\b', r'\bsee\s+details\b',
        r'\bread\s+more\b', r'\bfind\s+out\s+more\b', r'\bdm\s+us\b', r'\bswipe\s+up\b',
        r'\buse\s+the\s+link\b', r'\bshop\s+now\b', r'\btry\s+it\s+now\b',
        r'\bget\s+it\s+now\b', r'\bwatch\s+now\b', r'\bjoin\s+now\b', r'\bstart\s+now\b',
        r'\bcheck\s+it\s+out\b', r'\bsee\s+more\b', r'\bview\s+more\b', r'\bcheck\s+out\b',
        r'\bget\s+more\b', r'\bwatch\s+more\b', r'\bjoin\s+us\b', r'\bfollow\s+us\b',
        r'\blike\s+us\b', r'\bsave\s+it\b', r'\bshare\s+it\b', r'\bcomment\s+below\b',
        r'\blike\s+this\s+video\b', r'\bshare\s+this\s+video\b', r'\bcomment\s+now\b',
        r'\blike\s+now\b', r'\bshare\s+now\b', r'\bcomment\s+on\s+this\b',
        r'\blike\s+on\s+this\b', r'\bshare\s+on\s+this\b', r'\bcomment\s+on\s+this\s+video\b',
        r'\blike\s+on\s+this\s+video\b', r'\bshare\s+on\s+this\s+video\b',
        r'\bcomment\s+on\s+this\s+post\b', r'\blike\s+on\s+this\s+post\b',
        r'\bshare\s+on\s+this\s+post\b', r'\bcomment\s+on\s+this\s+story\b',
        r'\blike\s+on\s+this\s+story\b', r'\bshare\s+on\s+this\s+story\b',
        r'\bcomment\s+on\s+this\s+reel\b', r'\blike\s+on\s+this\s+reel\b',
        r'\bshare\s+on\s+this\s+reel\b', r'\bcomment\s+on\s+this\s+tiktok\b',
        r'\blike\s+on\s+this\s+tiktok\b', r'\bshare\s+on\s+this\s+tiktok\b',
        r'\bcomment\s+on\s+this\s+instagram\b', r'\blike\s+on\s+this\s+instagram\b',
        r'\bshare\s+on\s+this\s+instagram\b'
    ]
    
    # Detect CTA phrases in the description
    detected_ctas = []
    for pattern in cta_patterns:
        matches = re.findall(pattern, Transcript)
        detected_ctas.extend(matches)
    
    # Count the occurrences of each CTA
    cta_counts = Counter(detected_ctas)
    
    # Convert the counts to a list of tuples
    cta_list = [(cta, count) for cta, count in cta_counts.items()]
    
    return cta_list

def analyze_keywords(title, description):
    # Combine title and description
    text = title.lower() + " " + description.lower()
    
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    # Count the frequency of each keyword
    keyword_counts = Counter(filtered_tokens)
    
    # Get the top 5 most frequent keywords
    top_keywords = keyword_counts.most_common(5)
    
    return top_keywords

def calculate_readability_score(description):
    # Tokenize the description into sentences and words
    sentences = sent_tokenize(description)
    words = word_tokenize(description)
    
    # Calculate the number of sentences and words
    num_sentences = len(sentences)
    num_words = len(words)
    
    # Calculate the number of syllables (simplified for demonstration)
    def count_syllables(word):
        return len([char for char in word if char.lower() in 'aeiou'])
    
    num_syllables = sum(count_syllables(word) for word in words)
    
    # Calculate the Flesch-Kincaid readability score
    if num_sentences == 0 or num_words == 0:
        return 0, "Invalid description"
    
    readability_score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    
    # Determine the readability rating
    if readability_score >= 90:
        rating = "Easy"
    elif readability_score >= 50:
        rating = "Medium"
    else:
        rating = "Hard"
    
    return readability_score, rating

def get_readability_tips(rating):
    tips = []
    if rating == "Easy":
        tips.append("The description is already very accessible. Keep up the good work!")
    elif rating == "Medium":
        tips.append("Consider simplifying some complex sentences to improve accessibility.")
        tips.append("Use shorter sentences and more common words to make the description easier to read.")
    elif rating == "Hard":
        tips.append("The description is quite difficult to read. Simplifying it could improve accessibility.")
        tips.append("Break down complex ideas into simpler sentences.")
        tips.append("Use more common words and avoid jargon.")
    
    return tips

def get_keyword_insights(top_keywords, title, description):
    # Extract the top keywords
    keywords = [keyword for keyword, count in top_keywords]
    
    # Provide insights on the relevance of these keywords
    insights = []
    for keyword in keywords:
        if keyword in title:
            insights.append(f"'{keyword}' is a strong keyword as it appears in the title.")
        elif keyword in description:
            insights.append(f"'{keyword}' is a relevant keyword as it appears in the description.")
        else:
            insights.append(f"'{keyword}' is less relevant as it does not appear in the title or description.")
    
    # Suggestion on improving SEO
    suggestion = "To improve SEO, consider including more relevant keywords in the title and description. " \
                 "Ensure that the top keywords are prominently featured in both the title and description."
    
    return insights, suggestion

# Topic extraction with exception handling
def extract_topics(text: str) -> Dict[str, str]:
    try:
        system_message = """
        You are a topic extraction model. Your task is to extract the main and niche topics from the given text.
        The main topic should be a broad category that encompasses the overall content of the text.
        The niche topic should be a more specific sub-category within the main topic.
        If two videos are in the same niche, they should have the same main and niche topics.
        The third topic can be different, as long as it still falls within the same niche.
        Return the main and niche topics as a JSON object with the following structure:
        {
            "main_topic": "main topic",
            "niche_topic": "niche topic",
            "third_topic": "third topic"
        }
        Note: Please ensure that the response is formatted as a JSON object.
        """
        
        user_message = f"Text: {text}"

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
            response_format={"type": "json_object"},  # Corrected response format
        )

        # Ensure the content is a valid JSON and return the extracted topics
        return orjson.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        print(f"Error extracting topics: {e}")
        return {"main_topic": "N/A", "niche_topic": "N/A", "third_topic": "N/A"}

# Summarize video content with exception handling
def summarize_video_content(text: str) -> Dict[str, str]:
    try:
        system_message = """
        You are a video summary model. Your task is to provide a concise summary of the main content of the video described in the text.
        The summary should capture the primary focus of the video, key points discussed, and any important conclusions or takeaways.
        Ensure the summary is brief but informative, covering the main idea and any notable details.
        Return the summary as a JSON object with the following structure:
        {
            "video_summary": "summary text"
        }
        """
        user_message = f"Text: {text}"

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
            response_format={"type": "json_object"},  # Corrected response format
        )

        # Ensure the content is a valid JSON and return the video summary
        return orjson.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        print(f"Error summarizing video content: {e}")
        return {"video_summary": "Summary not available"}

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

def analyze_heatmap_data(heatmap_points: List[Dict[str, float]], threshold: float = 1.35) -> Dict[str, any]:
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

from typing import List, Dict, Any, Optional
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

def fetch_transcript(video_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch the transcript for a YouTube video, prioritizing English transcripts.
    If no English transcript is available, translate an available transcript to English.

    Args:
        video_id (str): The YouTube video ID.

    Returns:
        Optional[List[Dict[str, Any]]]: The transcript as a list of dictionaries, or None if not found.
    """
    try:
        # List all available transcripts for the video
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Prioritize English transcripts
        english_transcript = None
        for transcript in transcript_list:
            if transcript.language_code == 'en':
                english_transcript = transcript
                if not transcript.is_generated:  # Prefer manual transcripts
                    break
        
        # If no English transcript, try translating a non-generated transcript to English
        if not english_transcript:
            for transcript in transcript_list:
                if not transcript.is_generated:  # Try translating manual transcript
                    try:
                        english_transcript = transcript.translate('en')
                        break
                    except Exception as e:
                        print(f"Error translating transcript: {e}")
        
        # If no manual or translated English transcript, use an auto-generated transcript
        if not english_transcript:
            for transcript in transcript_list:
                if transcript.is_generated:
                    try:
                        english_transcript = transcript.translate('en')
                        break
                    except Exception as e:
                        print(f"Error translating auto-generated transcript: {e}")
        
        # Fetch and return the transcript data if available
        if english_transcript:
            return english_transcript.fetch()
        else:
            print("No English transcript available or translatable for this video.")
            return None

    except TranscriptsDisabled:
        print("Transcripts are disabled for this video.")
        return None
    except NoTranscriptFound:
        print("No transcripts found for this video.")
        return None
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None

def get_significant_transcript_sections(transcript: Optional[List[Dict[str, any]]], analysis_data: Dict[str, any]) -> Dict[str, List[List[Dict[str, any]]]]:
    if not transcript:
        print("Transcript is unavailable. Returning empty significant sections.")
        return {'rises': [], 'falls': []}  # Ensure 'rises' and 'falls' keys are present

    significant_sections = {'rises': [], 'falls': []}

    for rise in analysis_data.get('significant_rises', []):
        rise_transcript = [entry for entry in transcript if rise['start'] <= entry['start'] <= rise['end']]
        significant_sections['rises'].append(rise_transcript)

    for fall in analysis_data.get('significant_falls', []):
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
    thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"
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

def calculate_watch_time(views, duration_seconds):
    # Calculate total watch time in seconds
    total_watch_time_seconds = views * duration_seconds
    
    # Convert total watch time to minutes
    total_watch_time_minutes = total_watch_time_seconds / 60
    
    # Convert total watch time to hours
    total_watch_time_hours = total_watch_time_minutes / 60
    
    # Format total watch time in hours with appropriate suffix
    if total_watch_time_hours >= 1e9:  # Greater than or equal to 1 billion
        total_watch_time_formatted = f"{total_watch_time_hours / 1e9:.2f}B hours"
    elif total_watch_time_hours >= 1e6:  # Greater than or equal to 1 million
        total_watch_time_formatted = f"{total_watch_time_hours / 1e6:.2f}M hours"
    elif total_watch_time_hours >= 1e3:  # Greater than or equal to 1 thousand
        total_watch_time_formatted = f"{total_watch_time_hours / 1e3:.2f}K hours"
    else:
        total_watch_time_formatted = f"{total_watch_time_hours:.2f} hours"
    
    # Calculate watch time per user in seconds
    watch_time_per_user_seconds = duration_seconds
    
    # Convert watch time per user to minutes
    watch_time_per_user_minutes = watch_time_per_user_seconds / 60
    
    # Convert watch time per user to hours if greater than 60 minutes
    if watch_time_per_user_minutes > 60:
        watch_time_per_user_hours = watch_time_per_user_minutes / 60
        watch_time_per_user_formatted = f"{watch_time_per_user_hours:.2f} hours"
    else:
        watch_time_per_user_formatted = f"{watch_time_per_user_minutes:.2f} minutes"
    
    return {
        "total_watch_time": total_watch_time_formatted,
        "watch_time_per_user": watch_time_per_user_formatted
    }

async def extract_video_data(video_id):
    logging.info(f"Extracting video data for video ID: {video_id}")
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

        video_url = f"https://www.youtube.com/watch?v={video_id}"
        await page.goto(video_url, wait_until="domcontentloaded", timeout=60000)

        if "m.youtube.com" in page.url:
            await page.goto(video_url.replace("m.youtube.com", "www.youtube.com"), wait_until="networkidle")

        expand_selector = 'tp-yt-paper-button#expand'

        try:
            await page.wait_for_selector(expand_selector, timeout=20000)
            expand_button = await page.query_selector(expand_selector)
            if expand_button:
                await expand_button.click()
        except PlaywrightTimeoutError:
            logging.warning("Expand button not found.")

        await page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
        await page.wait_for_timeout(8000)

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

        heatmap_points = parse_svg_heatmap(heatmap_svg, duration_to_seconds_value)

        heatmap_analysis = analyze_heatmap_data(heatmap_points)

        # Ensure heatmap_analysis has default values
        average_attention = heatmap_analysis.get('average_attention', 0)
        total_rises = heatmap_analysis.get('total_rises', 0)
        total_falls = heatmap_analysis.get('total_falls', 0)
        significant_rises = heatmap_analysis.get('significant_rises', [])
        significant_falls = heatmap_analysis.get('significant_falls', [])

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

        # Parallel execution of comment extraction and thumbnail analysis
        comments_task = asyncio.create_task(extract_comments(video_id))
        thumbnail_task = asyncio.create_task(asyncio.to_thread(analyze_thumbnail, video_id))

        comments = await comments_task
        beautified_comments = [{"author": comment['author'], "text": re.sub(r'<.*?>', '', comment['text'])} for comment in comments]

        dominant_color, palette, thumbnail_text = await thumbnail_task

        overall_sentiment = analyze_comments_sentiment(beautified_comments, comment_count)

        comment_to_view_ratio = comment_count / views * 100 if views else 0
        like_to_views_ratio = likes / views * 100 if views else 0
        comment_to_like_ratio = comment_count / likes * 100 if likes else 0
        # Calculate the engagement rate
        engagement_rate = (likes + comment_count) / views * 100 if views else 0

        logging.info(f"Getting AI Analysis for video ID: {video_id}")

        # Fetch transcript
        transcript = fetch_transcript(video_id)

        # Check if transcript is available
        if transcript is None or len(transcript) == 0:
            print(f"No transcript available for video {video_id}. Skipping transcript analysis.")
            # Assign default values to avoid KeyError
            significant_transcript_sections = {'rises': [], 'falls': []}
        else:
            # Proceed with transcript-related logic if transcript is available
            significant_transcript_sections = get_significant_transcript_sections(transcript, heatmap_analysis)

        # Ensure significant_transcript_sections has default keys
        rises_sections = significant_transcript_sections.get('rises', [])
        falls_sections = significant_transcript_sections.get('falls', [])

        # Extract topics
        combined_text = f"{title}\n{description}\n{' '.join([entry['text'] for entry in transcript[:500]])}\n{thumbnail_text}\n{tags}" if transcript else f"{title}\n{description}"
        topics = extract_topics(combined_text)

        top_keywords = analyze_keywords(title, description)
        insights, suggestion = get_keyword_insights(top_keywords, title, description)

        readability_score, rating = calculate_readability_score(description)
        tips = get_readability_tips(rating)

        # Ensure topics are properly retrieved
        main_topic = topics.get('main_topic', 'N/A')
        niche_topic = topics.get('niche_topic', 'N/A')
        third_topic = topics.get('third_topic', 'N/A')

        # Get summary
        summary_response = summarize_video_content(combined_text)
        summary = summary_response.get("video_summary", "Summary not available")

        cta_list = detect_ctas(description)

        watch_time = calculate_watch_time(views, duration_to_seconds_value) if duration_to_seconds_value else "Duration not available"

        logging.info(f"Creating Output json for video ID: {video_id}")

        # Enhancing output for better readability with newlines in long text fields
        output_json = {
                "title": title,
                "video_statistics": {
                    "video_id": video_id,
                    "title": title,
                    "views": views,
                    "publish_time": publish_time,
                    "tags": tags,
                    "likes": likes,
                    "duration": duration,
                    "topics": {
                        "main_topic": main_topic,
                        "niche_topic": niche_topic,
                        "third_topic": third_topic
                    },
                    "summary": "\n".join(summary.split(". ")),  # Adding newlines after sentences in summary
                    "description": "\n".join(description.split("\n")),  # Retaining line breaks in the description
                    "duration_in_seconds": duration_to_seconds_value,
                    "watch_time": {
                        "total_watch_time": watch_time["total_watch_time"],
                        "watch_time_per_user": watch_time["watch_time_per_user"]
                    },
                    "ctas": cta_list
                },
                "overall_sentiment": {
                    "comment_sentiment": overall_sentiment
                },
                "thumbnail_analysis": {
                    "dominant_color": dominant_color,
                    "palette_colors": palette,
                    "thumbnail_text": thumbnail_text
                },
                "engagement_stats": {
                    "engagement_rate": engagement_rate,
                    "comment_to_view_ratio": comment_to_view_ratio,
                    "views_to_like_ratio": like_to_views_ratio,
                    "comment_to_like_ratio": comment_to_like_ratio,
                    "suggestions": suggestion,
                    "keyword_insights": insights,
                    "top_keywords": {keyword: count for keyword, count in top_keywords},
                    "readability_score": {
                        "score": readability_score,
                        "rating": rating,
                        "tips": tips
                    }
                },
                "links": "\n".join(links),  # Adding newlines between links for better visibility
                "heatmap_analysis": {
                    "average_attention": average_attention,
                    "total_rises": total_rises,
                    "total_falls": total_falls,
                    "significant_rises": significant_rises,
                    "significant_falls": significant_falls,
                    "heatmap_svg": {
                        "graph_points": heatmap_points,
                        "svg": heatmap_svg
                    }
                },
                "significant_transcript_sections": {
                    "rises": [
                        {"start": rise["start"], "end": rise["end"], "text": entry["text"]}
                        for rise, transcript in zip(significant_rises, rises_sections)
                        for entry in transcript
                    ],
                    "falls": [
                        {"start": fall["start"], "end": fall["end"], "text": entry["text"]}
                        for fall, transcript in zip(significant_falls, falls_sections)
                        for entry in transcript
                    ]
                },
                "transcript": transcript,  # Add transcript data
                "comments": {
                    "total_comments": comment_count,
                    "list": beautified_comments
                }
            }

            # Converting JSON to string with proper indentation
        output_json_str = orjson.dumps(output_json, option=orjson.OPT_INDENT_2).decode('utf-8')

        with open(f'{video_id}_data.json', 'w', encoding='utf-8') as json_file:
            json_file.write(output_json_str)

        logging.info("Extraction and json file creation completed successfully.")

        return output_json

async def extract_heatmap_svgs(page):
    # Wait for the network to be idle to ensure all resources have loaded
    try:
        await page.wait_for_load_state('networkidle')
        logging.info("Network idle state reached")
    except Exception as e:
        logging.error(f"Timeout waiting for network idle: {e}")
        return f"Timeout waiting for network idle: {e}"

    # Wait for the heatmap container to be visible
    try:
        await page.wait_for_selector('div.ytp-heat-map-container', state='hidden', timeout=30000)
    except Exception as e:
        logging.error(f"Timeout waiting for heatmap container: {e}")
        return f"Timeout waiting for heatmap container: {e}"

    heatmap_container = await page.query_selector('div.ytp-heat-map-container')
    if heatmap_container:
        heatmap_container_html = await heatmap_container.inner_html()
    else:
        return "Heatmap container not found"

    tree = html.fromstring(heatmap_container_html)

    heatmap_elements = tree.xpath('//div[@class="ytp-heat-map-chapter"]/svg')

    if not heatmap_elements:
        return "Heatmap SVG not found"

    total_width = sum(get_pixel_value(elem.attrib['width']) for elem in heatmap_elements)
    total_height = max(get_pixel_value(elem.attrib['height']) for elem in heatmap_elements)

    combined_svg = etree.Element('svg', {
        'xmlns': 'http://www.w3.org/2000/svg',
        'width': f'{total_width}px',
        'height': f'{total_height}px',
        'viewBox': f'0 0 {total_width} {total_height}'
    })

    current_x = 0
    for elem in heatmap_elements:
        width = get_pixel_value(elem.attrib['width'])
        height = get_pixel_value(elem.attrib['height'])

        group = etree.SubElement(combined_svg, 'g', {
            'transform': f'translate({current_x}, 0)'
        })

        for child in elem.getchildren():
            group.append(child)

        current_x += width

    combined_svg_str = etree.tostring(combined_svg, pretty_print=True).decode('utf-8')

    if not combined_svg_str or combined_svg_str.strip() == "":
        logging.error("Combined SVG heatmap content is empty or None")
        return "Combined SVG heatmap content is empty or None"

    return combined_svg_str

# Helper function to get pixel value
def get_pixel_value(value):
    if 'px' in value:
        return int(value.replace('px', ''))
    elif '%' in value:
        # Assuming the parent container's width is 1000px for simplicity
        return int(float(value.replace('%', '')) * 10)
    else:
        raise ValueError(f"Unsupported width/height format: {value}")

async def extract_comments(video_id, limit=100):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(f'https://www.youtube.com/watch?v={video_id}', sort_by=SORT_BY_RECENT)
    return list(islice(comments, limit))

def beautify_output(input_text):
    # Remove leading and trailing whitespace
    input_text = input_text.strip()

    # Split the input text into sections
    sections = re.split(r'\n\s*\n', input_text)

    # Initialize the json output
    json_output = []

    # Process each section
    for section in sections:
        # Remove leading and trailing whitespace from each section
        section = section.strip()

        # Check if the section is a title
        if section.startswith('#'):
            json_output.append(section)
            continue

        # Check if the section is a description
        if section.startswith('**Description:**'):
            json_output.append('## Description')
            description_lines = section.split('\n')[1:]
            for line in description_lines:
                json_output.append(f"- {line.strip()}")
            continue

        # Check if the section is a list of links
        if section.startswith('## Links'):
            json_output.append('## Links')
            links = section.split('\n')[1:]
            for link in links:
                json_output.append(f"- {link.strip()}")
            continue

        # Check if the section is a heatmap SVG
        if section.startswith('## Heatmap SVG'):
            json_output.append('## Heatmap SVG')
            svg_content = section.split('\n', 1)[1]
            json_output.append(f"```svg\n{svg_content}\n```")
            continue

        # Check if the section is a list of comments
        if section.startswith('## Comments'):
            json_output.append('## Comments')
            comments = eval(section.split('\n', 1)[1])
            for comment in comments:
                json_output.append(f"### {comment['author']}")
                json_output.append(f"{comment['text']}")
            continue

        # If the section is a list of details
        if section.startswith('**'):
            json_output.append('## Video Details')
            details = section.split('\n')
            for detail in details:
                json_output.append(f"- {detail.strip()}")
            continue

    # Join the json output into a single string
    return '\n'.join(json_output)

# Streamlit App
def main():
    st.title("YouTube Video Data Extractor")
    video_id = st.text_input("Enter the YouTube video ID:")

    if st.button("Extract Data"):
        if video_id:
            st.write("Extracting data...")
            data = asyncio.run(extract_video_data(video_id))
            st.json(data)
        else:
            st.write("Please enter a valid YouTube video ID.")

if __name__ == "__main__":
    main()
