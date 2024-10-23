import asyncio
import os
import zipfile
import logging
import random
import re
from concurrent.futures import ProcessPoolExecutor
import streamlit as st
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from lxml import html, etree
from urllib.parse import parse_qs, urlparse
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT

# Install Playwright using os.system (this should only be done once)
os.system("pip install playwright")
# Install the necessary browsers for Playwright
os.system("playwright install")

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

        # Block unnecessary resources
        await context.route("**/*", lambda route: route.abort() if route.request.resource_type in ["video", "audio", "font"] else route.continue_())

        page = await context.new_page()

        video_url = f"https://www.youtube.com/watch?v={video_id}"
        await page.goto(video_url, wait_until="networkidle")

        logging.info(f"Starting Extraction Process for video ID: {video_id}")

        if "m.youtube.com" in page.url:
            await page.goto(video_url.replace("m.youtube.com", "www.youtube.com"), wait_until="networkidle")

        try:
            content = await page.content()
            tree = html.fromstring(content)

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

            if likes_element:
                aria_label = await likes_element.get_attribute('aria-label')
                likes = re.search(r'(\d[\d,]*)', aria_label).group(1) if aria_label else "0"
            else:
                likes = "0"

            # Extract Heatmap SVG
            heatmap_svg = await extract_heatmap_svgs(page)

            # Ensure unique description and links to avoid repetition
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
                            link_text = convert_hashtag_to_link(link_text)
                            link_href = f"https://www.youtube.com{link_href}"
                        description_parts.append(f"[{link_text}]({link_href})")
                        links.add(link_href)

            description = ' '.join(description_parts)

            # Extract Duration
            duration_element = tree.xpath('//span[@class="ytp-time-duration"]')
            duration = duration_element[0].text_content().strip() if duration_element else "Duration not found"

            comments = await extract_comments(video_id)
            beautified_comments = [{"author": comment['author'], "text": re.sub(r'<.*?>', '', comment['text'])} for comment in comments]

            # Create Markdown content
            markdown_content = f"""# {title}

## Video Statistics

- **Video ID:** {video_id}
- **Title:** {title}
- **Views:** {views}  
- **Publish Time:** {publish_time}  
- **Tags:** {', '.join(tags)}  
- **Likes:** {likes}  
- **Duration:** {duration}  
- **Description:** {description}  

## Links
{','.join(links)}

## Heatmap SVG
![Heatmap]({heatmap_svg})

## Comments

- **Total No of Comments:** {len(beautified_comments)}

| Author               | Comment                                                                 |
|----------------------|-------------------------------------------------------------------------|
"""
            for comment in beautified_comments:
                markdown_content += f"| {comment['author']} | {comment['text']} |\n"

            # Save Markdown content to a file with the correct video_id
            with open(f'{title}_data.md', 'w', encoding='utf-8') as md_file:
                md_file.write(markdown_content)

            logging.info("Extraction and Markdown file creation completed successfully.")
            return f'{title}_data.md'

        except PlaywrightTimeoutError:
            logging.error(f"Failed to extract data for video ID: {video_id}")
            return None

        finally:
            await context.close()
            await browser.close()

async def extract_heatmap_svgs(page):
    # Wait for the network to be idle to ensure all resources have loaded
    await page.wait_for_load_state('networkidle')

    heatmap_container = await page.query_selector('div.ytp-heat-map-container')
    if heatmap_container:
        heatmap_container_html = await heatmap_container.inner_html()
    else:
        return "Heatmap container not found"

    # Parse the HTML content
    tree = html.fromstring(heatmap_container_html)
    heatmap_elements = tree.xpath('//svg')

    if not heatmap_elements:
        return "Heatmap SVG not found"

    # Extract and return the SVG content
    heatmap_svg_content = etree.tostring(heatmap_elements[0], pretty_print=True, encoding='unicode')
    return heatmap_svg_content

async def extract_comments(video_id, limit=20):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments(video_id)
    return list(islice(comments, limit))

def run_extraction(video_id):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(extract_video_data(video_id))

def main():
    st.title("YouTube Video Data Extractor")
    
    video_id = st.text_input("Enter YouTube Video ID", value="")
    
    if st.button("Extract Data"):
        if video_id:
            logging.info(f"Button pressed for video ID: {video_id}")
            with st.spinner("Extracting data..."):
                try:
                    # Run the extraction in a separate process to avoid blocking Streamlit
                    with ProcessPoolExecutor() as executor:
                        markdown_file = executor.submit(run_extraction, video_id).result()

                    if markdown_file:
                        st.success(f"Data extracted successfully! Markdown file created: {markdown_file}")
                        with open(markdown_file, 'rb') as file:
                            st.download_button("Download Markdown", file, file_name=markdown_file)
                    else:
                        st.error("Failed to extract data. Please check the Video ID.")

                except Exception as e:
                    logging.error(f"An error occurred during extraction: {e}")
                    st.error("An unexpected error occurred. Please try again.")

if __name__ == "__main__":
    main()
