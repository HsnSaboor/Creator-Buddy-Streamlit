import asyncio
import datetime
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from lxml import html, etree
from urllib.parse import parse_qs, urlparse
import random
import re
import streamlit as st
import logging
import pandas as pd
from itertools import islice
from textblob import TextBlob
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
import requests
from PIL import Image
import pytesseract
from colorthief import ColorThief
from io import BytesIO
import math

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

def get_luminance(color):
    # Calculate luminance for the given color (RGB values in the range 0-255)
    r, g, b = [channel / 255.0 for channel in color]
    r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def contrast_score(color1, color2):
    # Calculate the contrast ratio between two colors
    luminance1 = get_luminance(color1)
    luminance2 = get_luminance(color2)
    contrast_ratio = (luminance1 + 0.05) / (luminance2 + 0.05) if luminance1 > luminance2 else (luminance2 + 0.05) / (luminance1 + 0.05)
    # Scale contrast ratio (1 to 21) to a score out of 10
    score = min(10, (contrast_ratio - 1) * (10 / 20))
    return round(score, 1)  # Return score rounded to 1 decimal place

# Assuming analyze_thumbnail is defined as shown before
def analyze_thumbnail(video_id):
    # Download and save thumbnail
    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    thumbnail_response = requests.get(thumbnail_url)
    thumbnail_image = Image.open(BytesIO(thumbnail_response.content))

    # Analyze thumbnail colors using ColorThief
    color_thief = ColorThief(BytesIO(thumbnail_response.content))
    dominant_color = color_thief.get_color(quality=1)
    palette = color_thief.get_palette(color_count=6)

    # Calculate contrast score between dominant color and each color in the palette
    contrast_scores = [contrast_score(dominant_color, color) for color in palette]

    # Average the contrast scores to get an overall score
    averagecontrast = round(sum(contrast_scores) / len(contrast_scores), 1)

    # Extract text from thumbnail using Tesseract OCR
    thumbnail_text = pytesseract.image_to_string(thumbnail_image)

    # Return average_contrast_score so it's accessible when the function is called
    return dominant_color, palette, averagecontrast, thumbnail_text.strip()

# Calculate sentiment for each comment
def get_comment_sentiment(comment_text):
    analysis = TextBlob(comment_text)
    return "Positive" if analysis.sentiment.polarity > 0 else "Negative"

# Apply sentiment analysis to each comment and determine overall sentiment
def analyze_comments_sentiment(beautified_comments, comment_count):
    positive_comments = sum(1 for comment in beautified_comments if get_comment_sentiment(comment['text']) == "Positive")
    overall_sentiment = "Positive" if positive_comments > (comment_count / 2) else "Negative"
    return overall_sentiment

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
        await page.goto(video_url, wait_until="domcontentloaded", timeout=60000)


        logging.info(f"Starting Extraction Process for video ID: {video_id}")

        if "m.youtube.com" in page.url:
            await page.goto(video_url.replace("m.youtube.com", "www.youtube.com"), wait_until="networkidle")

        expand_selector = 'tp-yt-paper-button#expand'

        try:
            await page.wait_for_selector(expand_selector, timeout=8000)
            expand_button = await page.query_selector(expand_selector)
            if expand_button:
                await expand_button.click()
                print(f"Successfully clicked the button with selector: {expand_selector}")
        except PlaywrightTimeoutError:
            logging.warning("Expand button not found.")

        # Scroll to the bottom to load comments
        await page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
        await page.wait_for_timeout(8000)  # Wait to ensure comments load

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

        # Convert views to integer
        if views != "Views not found":
            views = int(re.sub(r'\D', '', views))  # Remove non-digit characters and convert to int

        # Extract Likes
        like_button_selector = '//button[contains(@class, "yt-spec-button-shape-next") and @title="I like this"]'
        likes_element = await page.query_selector(like_button_selector)
        likes = "Likes not found"
        if likes_element:
            aria_label = await likes_element.get_attribute('aria-label')
            if aria_label:
                match = re.search(r'(\d[\d,]*)', aria_label)
                if match:
                    likes = int(match.group(1).replace(',', ''))  # Remove commas and convert to int

        # Extract Heatmap SVG
        heatmap_svg = await extract_heatmap_svgs(page)

        # Extract Description and Links
        description_elements = tree.xpath('//ytd-text-inline-expander[@id="description-inline-expander"]//yt-attributed-string[@user-input=""]//span[@class="yt-core-attributed-string yt-core-attributed-string--white-space-pre-wrap"]')
        description_parts = []
        links = set()  # Use a set to prevent duplicate links

        for element in description_elements:
            text_content = element.text_content().strip()
            if text_content and text_content not in description_parts:  # Prevent text duplication
                description_parts.append(text_content)
            for link in element.xpath('.//a'):
                link_text = link.text_content().strip()
                link_href = link.get('href')
                if link_text and link_href and link_href not in links:  # Prevent link duplication
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

        # Extract Comments
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

        # Apply sentiment analysis to each comment and determine overall sentiment
        overall_sentiment = analyze_comments_sentiment(beautified_comments, comment_count)

        # Assuming `views`, `likes`, and `comment_count` are already converted to integers
        comment_to_view_ratio = comment_count / views * 100 if views else 0
        like_to_views_ratio = likes / views * 100 if views else 0
        comment_to_like_ratio = comment_count / likes * 100 if likes else 0


        logging.info(f"Creating Output Markdown for video ID: {video_id}")

        # Analyze Thumbnail
        dominant_color, palette, averagecontrast, thumbnail_text = analyze_thumbnail(video_id)

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

## Overall Sentiment

- **Overall Comment Sentiment:** {overall_sentiment}

## Thumbnail Analysis

- **Dominant Color:** {dominant_color}
- **Palette Colors:** {palette}
- **Thumbnail Text:** {thumbnail_text}
- **Contrast Score:** {averagecontrast}

## Ratios

- **Comment to View Ratio:** {comment_to_view_ratio:.2f}%
- **Views to Like Ratio:** {like_to_views_ratio:.2f}%
- **Comment to Like Ratio:** {comment_to_like_ratio:.2f}%

## Links

{', '.join(links)}

 ## Heatmap SVG
```svg
{heatmap_svg}

## Comments

- **Total No of Comments:** {comment_count}

| Author               | Comment                                                                 |
|----------------------|-------------------------------------------------------------------------|

"""
        
        for comment in beautified_comments:
            markdown_content += f"| {comment['author']} | {comment['text']} |\n"

    # Save Markdown content to a file with the correct video_id
    with open(f'{video_id}_data.md', 'w', encoding='utf-8') as md_file:  # Using f-string for correct formatting
        md_file.write(markdown_content)

    logging.info("Extraction and Markdown file creation completed successfully.")
   
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



async def extract_comments(video_id, limit=100):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(f'https://www.youtube.com/watch?v={video_id}', sort_by=SORT_BY_RECENT)
    return list(islice(comments, limit))

def beautify_output(input_text):
    # Remove leading and trailing whitespace
    input_text = input_text.strip()
    
    # Split the input text into sections
    sections = re.split(r'\n\s*\n', input_text)
    
    # Initialize the Markdown output
    markdown_output = []
    
    # Process each section
    for section in sections:
        # Remove leading and trailing whitespace from each section
        section = section.strip()
        
        # Check if the section is a title
        if section.startswith('#'):
            markdown_output.append(section)
            continue
        
        # Check if the section is a description
        if section.startswith('**Description:**'):
            markdown_output.append('## Description')
            description_lines = section.split('\n')[1:]
            for line in description_lines:
                markdown_output.append(f"- {line.strip()}")
            continue
        
        # Check if the section is a list of links
        if section.startswith('## Links'):
            markdown_output.append('## Links')
            links = section.split('\n')[1:]
            for link in links:
                markdown_output.append(f"- {link.strip()}")
            continue
        
        # Check if the section is a heatmap SVG
        if section.startswith('## Heatmap SVG'):
            markdown_output.append('## Heatmap SVG')
            svg_content = section.split('\n', 1)[1]
            markdown_output.append(f"```svg\n{svg_content}\n```")
            continue
        
        # Check if the section is a list of comments
        if section.startswith('## Comments'):
            markdown_output.append('## Comments')
            comments = eval(section.split('\n', 1)[1])
            for comment in comments:
                markdown_output.append(f"### {comment['author']}")
                markdown_output.append(f"{comment['text']}")
            continue
        
        # If the section is a list of details
        if section.startswith('**'):
            markdown_output.append('## Video Details')
            details = section.split('\n')
            for detail in details:
                markdown_output.append(f"- {detail.strip()}")
            continue
    
    # Join the Markdown output into a single string
    return '\n'.join(markdown_output)

if __name__ == "__main__":
    st.title("YouTube Video Analyzer")

# User input for video ID
    video_id = st.text_input("Enter YouTube Video ID:")

if st.button("Analyze Video"):
    start_time = datetime.datetime.time()  # Capture start time

    with st.spinner("Extracting data..."):
        try:
            asyncio.run(extract_video_data(video_id))
            # Load the extracted data from the saved Markdown file
            with open(f"{video_id}_data.md", "r") as f:
                markdown_content = f.read()
            # Display the Markdown content using Streamlit
            st.markdown(markdown_content)

            # Calculate and display execution time
            end_time = datetime.datetime.time()
            execution_time = end_time - start_time
            logging.info(f"Video analysis for '{video_id}' completed in {execution_time:.2f} seconds")
            st.success(f"Analysis complete! Execution time: {execution_time:.2f} seconds")

            # Create a download button for the Markdown file
            st.download_button(
                label="Download Report",
                data=markdown_content,
                file_name=f"{video_id}_report.md",
                mime="text/markdown"
            )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
