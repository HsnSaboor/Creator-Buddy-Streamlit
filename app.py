import streamlit as st
import asyncio
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from lxml import html, etree
from urllib.parse import parse_qs, urlparse
import random
import re
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
    r, g, b = [channel / 255.0 for channel in color]
    r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def contrast_score(color1, color2):
    luminance1 = get_luminance(color1)
    luminance2 = get_luminance(color2)
    contrast_ratio = (luminance1 + 0.05) / (luminance2 + 0.05) if luminance1 > luminance2 else (luminance2 + 0.05) / (luminance1 + 0.05)
    score = min(10, (contrast_ratio - 1) * (10 / 20))
    return round(score, 1)

def analyze_thumbnail(video_id):
    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    thumbnail_response = requests.get(thumbnail_url)
    thumbnail_image = Image.open(BytesIO(thumbnail_response.content))
    color_thief = ColorThief(BytesIO(thumbnail_response.content))
    dominant_color = color_thief.get_color(quality=1)
    palette = color_thief.get_palette(color_count=6)
    contrast_scores = [contrast_score(dominant_color, color) for color in palette]
    average_contrast = round(sum(contrast_scores) / len(contrast_scores), 1)
    thumbnail_text = pytesseract.image_to_string(thumbnail_image)
    return dominant_color, palette, average_contrast, thumbnail_text.strip()

def get_comment_sentiment(comment_text):
    analysis = TextBlob(comment_text)
    return "Positive" if analysis.sentiment.polarity > 0 else "Negative"

def analyze_comments_sentiment(beautified_comments, comment_count):
    positive_comments = sum(1 for comment in beautified_comments if get_comment_sentiment(comment['text']) == "Positive")
    overall_sentiment = "Positive" if positive_comments > (comment_count / 2) else "Negative"
    return overall_sentiment

async def extract_video_data(video_id):
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
            await page.wait_for_selector(expand_selector, timeout=5000)
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
        duration_element = tree.xpath('//span[@class="ytp-time-duration"]')
        duration = duration_element[0].text_content().strip() if duration_element else "Duration not found"
        comments_xpath = "//div[@id='leading-section']//h2[@id='count']//yt-formatted-string[@class='count-text style-scope ytd-comments-header-renderer']/span[1]/text()"
        comment_count_element = tree.xpath(comments_xpath)
        if comment_count_element:
            comment_count_text = comment_count_element[0]
            if comment_count_text.isdigit():
                comment_count = int(comment_count_text)
            else:
                comment_count = int(''.join(filter(str.isdigit, comment_count_text)))
        else:
            comment_count = 1
        comments = await extract_comments(video_id)
        beautified_comments = [{"author": comment['author'], "text": re.sub(r'<.*?>', '', comment['text'])} for comment in comments]
        overall_sentiment = analyze_comments_sentiment(beautified_comments, comment_count)
        comment_to_view_ratio = comment_count / views * 100 if views else 0
        like_to_views_ratio = likes / views * 100 if views else 0
        like_to_comment_ratio = likes / comment_count * 100 if likes else 0
        dominant_color, palette, average_contrast, thumbnail_text = analyze_thumbnail(video_id)
        
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

- **Overall Sentiment:** {overall_sentiment}

## Thumbnail Analysis

- **Dominant Color:** {dominant_color}
- **Palette Colors:** {palette}
- **Thumbnail Text:** {thumbnail_text}

## Ratios

- **Comment to View Ratio:** {comment_to_view_ratio:.2f}%
- **Views to Like Ratio:** {like_to_views_ratio:.2f}%
- **Comment to Like Ratio:** {like_to_comment_ratio:.2f}%

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
    with open(f'{title}_data.md', 'w', encoding='utf-8') as md_file:  # Using f-string for correct formatting
        md_file.write(markdown_content)

    logging.info("Extraction and Markdown file creation completed successfully.")
   
async def extract_heatmap_svgs(page):
    """
    Extract heatmap SVG from the given page.
    
    Args:
    page: The page object to extract heatmap SVG from.
    
    Returns:
    The extracted heatmap SVG as a string.
    """
    try:
        await page.wait_for_load_state('networkidle')
        logging.info("Network idle state reached")
    except Exception as e:
        return f"Timeout waiting for network idle: {e}"
    
    try:
        await page.wait_for_selector('div.ytp-heat-map-container', state='hidden', timeout=10000)
    except Exception as e:
        return f"Timeout waiting for heatmap container: {e}"
    
    heatmap_container = await page.query_selector('div.ytp-heat-map-container')
    if heatmap_container:
        heatmap_container_html = await heatmap_container.inner_html()
    else:
        return "Heatmap container not found"
    
    tree = html.fromstring(heatmap_container_html)
    heatmap_elements = tree.xpath('//svg')
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
    return combined_svg_str

def get_pixel_value(value):
    """
    Convert the given value to pixels.
    
    Args:
    value: The value to convert.
    
    Returns:
    The converted value in pixels.
    """
    if 'px' in value:
        return int(value.replace('px', ''))
    elif '%' in value:
        return int(float(value.replace('%', '')) * 10)
    else:
        raise ValueError(f"Unsupported width/height format: {value}")

async def extract_comments(video_id, limit=20):
    """
    Extract comments from the given video.
    
    Args:
    video_id: The ID of the video to extract comments from.
    limit: The maximum number of comments to extract (default: 20).
    
    Returns:
    A list of comments.
    """
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(f'https://www.youtube.com/watch?v={video_id}', sort_by=SORT_BY_RECENT)
    return list(islice(comments, limit))

def beautify_output(input_text):
    """
    Beautify the given input text.
    
    Args:
    input_text: The input text to beautify.
    
    Returns:
    The beautified text.
    """
    input_text = input_text.strip()
    sections = re.split(r'\n\s*\n', input_text)
    markdown_output = []
    for section in sections:
        section = section.strip()
        if section.startswith('#'):
            markdown_output.append(section)
            continue
        if section.startswith('Description:'):
            markdown_output.append('## Description')
            description_lines = section.split('\n')[1:]
            for line in description_lines:
                markdown_output.append(f"- {line.strip()}")
            continue
        if section.startswith('## Links'):
            markdown_output.append('## Links')
            links = section.split('\n')[1:]
            for link in links:
                markdown_output.append(f"- {link.strip()}")
            continue
        if section.startswith('## Heatmap SVG'):
            markdown_output.append('## Heatmap SVG')
            svg_content = section.split('\n', 1)[1]
            markdown_output.append(f"svg\n{svg_content}\n")
            continue
        if section.startswith('## Comments'):
            markdown_output.append('## Comments')
            comments = eval(section.split('\n', 1)[1])
            for comment in comments:
                markdown_output.append(f"### {comment['author']}")
                markdown_output.append(f"{comment['text']}")
            continue
        if section.startswith('**'):
            markdown_output.append('## Video Details')
            details = section.split('\n')
            for detail in details:
                markdown_output.append(f"- {detail.strip()}")
            continue
    return '\n'.join(markdown_output)

def main():
    st.title("YouTube Video Data Extractor")
    video_id = st.text_input("Enter the YouTube video ID:")
    if st.button("Extract Data"):
        with st.spinner("Extracting data..."):
            markdown_content = asyncio.run(extract_video_data(video_id))
            st.markdown(markdown_content)
            st.download_button(
                label="Download Markdown File",
                data=markdown_content,
                file_name=f"{video_id}_data.md",
                mime="text/markdown"
            )

if __name__ == "__main__":
    main()
