import asyncio
import pandas as pd
import logging
import random
import re
import zipfile
import os
from itertools import islice
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
from pathlib import Path
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from lxml import html, etree
from urllib.parse import parse_qs, urlparse
import streamlit as st

# Install Playwright using os.system
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
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    tasks = [extract_video_data(video_id, output_dir) for video_id in video_ids]
    await asyncio.gather(*tasks)

    # Create a zip file of all markdown files
    zip_filename = "video_data.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zip_file:
        for md_file in output_dir.glob("*.md"):
            zip_file.write(md_file, md_file.name)
    return zip_filename

async def extract_video_data(video_id, output_dir):
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

        selector = 'tp-yt-paper-button#expand'

        try:
            await page.wait_for_selector(selector, timeout=5000)
            expand_button = await page.query_selector(selector)
            if expand_button:
                await expand_button.click()
                print(f"Successfully clicked the button with selector: {selector}")

            content = await page.content()
            tree = html.fromstring(content)

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

            if likes_element:
                # Get the aria-label attribute from the like button
                aria_label = await likes_element.get_attribute('aria-label')
                
                if aria_label:
                    match = re.search(r'(\d[\d,]*)', aria_label)
                    likes = match.group(1) if match else "0"
                else:
                    likes = "0"
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

            # XPath to specifically target the first <span> within the yt-formatted-string under the comments section
            comments_xpath = "//div[@id='leading-section']//h2[@id='count']//yt-formatted-string/span[1]/text()"
            comment_count_element = tree.xpath(comments_xpath)
            comment_count = comment_count_element[0] if comment_count_element else None

            # Print the result
            print(f"Total number of comments: {comment_count}")

            comments = await extract_comments(video_id)
            beautified_comments = [{"author": comment['author'], "text": re.sub(r'<.*?>', '', comment['text'])} for comment in comments]
            
            logging.info(f"Creating Output Markdown for video ID: {video_id}")

            # Create Markdown content
            markdown_content = f""" {title}

        # Video Statistics

        - **Video ID:** {video_id}
        - **Title:** {title}
        - **Views:** {views}  
        - **Publish Time:** {publish_time}  
        - **Tags:** {', '.join(tags)}  
        - **Likes:** {likes}  
        - **Duration:** {duration}  
        - **Description:** {description}  
        - **Heatmap SVG:** ![Heatmap SVG]({heatmap_svg if heatmap_svg else 'No heatmap available'})

        ## Comments
        {''.join([f"- {comment['author']}: {comment['text']}\n" for comment in beautified_comments]) if beautified_comments else "No comments available."}
        """
            # Write Markdown content to file
            md_file_path = output_dir / f"{video_id}.md"
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                md_file.write(markdown_content)
            logging.info(f"Markdown file created for video ID: {video_id}")

        except PlaywrightTimeoutError:
            logging.error(f"Timeout while extracting data for video ID: {video_id}")
        except Exception as e:
            logging.error(f"An error occurred while extracting data for video ID {video_id}: {e}")

        await browser.close()

async def extract_comments(video_id, limit=20):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments(video_id, sort=SORT_BY_RECENT)
    return list(islice(comments, limit))

async def extract_heatmap_svgs(page):
    # Wait for the network to be idle to ensure all resources have loaded
    try:
        await page.wait_for_load_state('networkidle')
        logging.info("Network idle state reached")
    except Exception as e:
        return f"Timeout waiting for network idle: {e}"

        # Wait for the heatmap container to be visible
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

    # Calculate the total width and height of the combined SVG
    total_width = sum(get_pixel_value(elem.attrib['width']) for elem in heatmap_elements)
    total_height = max(get_pixel_value(elem.attrib['height']) for elem in heatmap_elements)

    # Create a new SVG element to hold the combined SVGs
    combined_svg = etree.Element('svg', {
        'xmlns': 'http://www.w3.org/2000/svg',
        'width': f'{total_width}px',
        'height': f'{total_height}px',
        'viewBox': f'0 0 {total_width} {total_height}'
    })

    # Position each SVG element horizontally
    current_x = 0
    for elem in heatmap_elements:
        # Get the width and height of the current SVG
        width = get_pixel_value(elem.attrib['width'])
        height = get_pixel_value(elem.attrib['height'])

        # Create a new group element to hold the current SVG and position it
        group = etree.SubElement(combined_svg, 'g', {
            'transform': f'translate({current_x}, 0)'
        })

        # Copy the current SVG's children to the new group
        for child in elem.getchildren():
            group.append(child)

        # Move the x position for the next SVG
        current_x += width

    # Convert the combined SVG to a string
    combined_svg_str = etree.tostring(combined_svg, pretty_print=True).decode('utf-8')

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


def main():
    st.title("YouTube Video Data Extractor")
    video_ids = st.text_area("Enter YouTube Video IDs (one per line):").splitlines()
    if st.button("Extract Data"):
        video_ids = list(filter(None, video_ids))  # Filter out empty lines
        if video_ids:
            zip_file = asyncio.run(process_videos(video_ids))
            st.success(f"Data extraction complete! Download your zip file [here](./{zip_file}).")
        else:
            st.warning("Please enter at least one video ID.")

if __name__ == "__main__":
    main()
