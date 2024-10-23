import asyncio
import pandas as pd
import logging
import random
import re
import zipfile
from itertools import islice
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
from lxml import html, etree
from urllib.parse import parse_qs, urlparse
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

# Install Playwright using os.system (this should only be done once)
os.system("pip install playwright")
# Install the necessary browsers for Playwright
os.system("playwright install")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
]

RESOLUTIONS = [
    {"width": 1024, "height": 768},
    {"width": 1280, "height": 720},
    {"width": 1366, "height": 768},
    {"width": 1440, "height": 900},
    {"width": 1600, "height": 900}
]

BROWSERS = ["chromium"]

# Helper Functions
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

async def process_videos(video_ids):
    # Create a BytesIO stream for the ZIP file
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for index, video_id in enumerate(video_ids):
            progress_text = f"Processing Video ID: {video_id}"
            st.write(progress_text)
            markdown_content = await extract_video_data(video_id)
            zip_file.writestr(f"{video_id}.md", markdown_content)
            st.progress((index + 1) / len(video_ids))  # Update progress bar
    
    zip_buffer.seek(0)  # Move to the beginning of the BytesIO stream
    return zip_buffer.getvalue()

async def extract_video_data(video_id):
    logging.info(f"Extracting video data for video ID: {video_id}")
    # Use Playwright to gather data
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

        # Block unnecessary resources
        await context.route("**/*", lambda route: route.abort() if route.request.resource_type in ["video", "audio", "font"] else route.continue_())

        page = await context.new_page()
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        await page.goto(video_url, wait_until="networkidle")

        try:
            if "m.youtube.com" in page.url:
                await page.goto(video_url.replace("m.youtube.com", "www.youtube.com"), wait_until="networkidle")

            selector = 'tp-yt-paper-button#expand'
            await page.wait_for_selector(selector, timeout=5000)

            expand_button = await page.query_selector(selector)
            if expand_button:
                await expand_button.click()

            content = await page.content()
            tree = html.fromstring(content)

            # Remove unwanted elements
            for elem in tree.xpath('//head | //style | //script'):
                elem.getparent().remove(elem)

            # Extract Title
            title_element = tree.xpath('//h1[@class="style-scope ytd-watch-metadata"]/yt-formatted-string')
            title = title_element[0].text_content().strip() if title_element else "Title not found"

            # Extract Views, Publish Time, and Tags
            info_element = tree.xpath('//yt-formatted-string[@id="info"]')
            if info_element:
                info_text = info_element[0].text_content().strip()
                views, publish_time, *tags = info_text.split('  ')
                tags = [tag.strip() for tag in tags if tag.strip()]
            else:
                views, publish_time, tags = "Views not found", "Publish time not found", []

            # Extract Likes
            like_button_selector = '//button[contains(@class, "yt-spec-button-shape-next") and @title="I like this"]'
            likes_element = await page.query_selector(like_button_selector)
            likes = "0"
            if likes_element:
                aria_label = await likes_element.get_attribute('aria-label')
                if aria_label:
                    match = re.search(r'(\d[\d,]*)', aria_label)
                    likes = match.group(1) if match else "0"

            # Extract Heatmap SVG
            heatmap_svg = await extract_heatmap_svgs(page)

            # Extract Description
            description = await extract_description(tree)

            # Extract Duration
            duration_element = tree.xpath('//span[@class="ytp-time-duration"]')
            duration = duration_element[0].text_content().strip() if duration_element else "Duration not found"

            # Extract Comments
            comments = await extract_comments(video_id)
            beautified_comments = [{"author": comment['author'], "text": re.sub(r'<.*?>', '', comment['text'])} for comment in comments]

            # Create Markdown content
            markdown_content = create_markdown_content(video_id, title, views, publish_time, tags, likes, duration, description, heatmap_svg, beautified_comments)

            logging.info(f"Markdown content created for video ID: {video_id}")

            return markdown_content

        except Exception as e:
            logging.error(f"An error occurred while extracting data for video ID {video_id}: {e}")
            return ""

        finally:
            await browser.close()

async def extract_comments(video_id, limit=20):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments(video_id, sort=SORT_BY_RECENT)
    return list(islice(comments, limit))

async def extract_heatmap_svgs(page):
    await page.wait_for_load_state('networkidle')
    logging.info("Network idle state reached")

    try:
        await page.wait_for_selector('div.ytp-heat-map-container', state='hidden', timeout=10000)
    except Exception as e:
        return f"Timeout waiting for heatmap container: {e}"

    heatmap_container = await page.query_selector('div.ytp-heat-map-container')
    if heatmap_container:
        heatmap_container_html = await heatmap_container.inner_html()
    else:
        return "Heatmap container not found"

    # Parse the HTML content
    tree = html.fromstring(heatmap_container_html)

    # Find all SVG elements within the heatmap container
    heatmap_elements = tree.xpath('//svg')

    if not heatmap_elements:
        return "Heatmap SVG not found"

    # Calculate the total width and height of the SVG elements
    svg_data = []
    for idx, svg in enumerate(heatmap_elements):
        svg_str = etree.tostring(svg).decode()
        svg_data.append(f"SVG Heatmap {idx+1}:\n" + svg_str)

    return "\n\n".join(svg_data)

def create_markdown_content(video_id, title, views, publish_time, tags, likes, duration, description, heatmap_svg, comments):
    comments_section = '\n\n'.join([f"**{comment['author']}**: {comment['text']}" for comment in comments])

    return f"""# Video Data for {video_id}

## Title: {title}
## Views: {views}
## Publish Time: {publish_time}
## Tags: {', '.join(tags) if tags else 'No Tags'}
## Likes: {likes}
## Duration: {duration}
## Description: {description}

## Heatmap SVGs
{heatmap_svg}

## Comments
{comments_section if comments else "No comments available."}
"""

# Streamlit UI
st.title("YouTube Video Data Extractor")
st.write("Enter Video IDs (comma-separated):")
video_ids_input = st.text_area("Video IDs", "")
video_ids = [vid.strip() for vid in video_ids_input.split(',') if vid.strip()]

if st.button("Extract Data"):
    if video_ids:
        zip_file = asyncio.run(process_videos(video_ids))
        
        # Cache the zip file data
        st.cache_data(zip_file, allow_output_mutation=True)

        # Provide download button
        st.download_button(
            label="Download Zip File",
            data=zip_file,
            file_name="youtube_video_data.zip",
            mime="application/zip"
        )
    else:
        st.warning("Please enter valid Video IDs.")
