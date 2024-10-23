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

# Install the necessary browsers for Playwrighte
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

# Streamlit UI
st.title("YouTube Video Data Extractor")
st.write("Upload a CSV file containing a column named 'video_id'.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    
    if 'video_id' not in df.columns:
        st.error("The CSV must contain a column named 'video_id'.")
    else:
        video_ids = df['video_id'].dropna().unique()
        st.write(f"Processing {len(video_ids)} video IDs...")

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
                # Block unnecessary resources
                await context.route("**/*", lambda route: route.abort() if route.request.resource_type in ["video", "audio", "font"] else route.continue_())

                page = await context.new_page()

                video_url = f"https://www.youtube.com/watch?v={video_id}"
                await page.goto(video_url, wait_until="networkidle")

                logging.info(f"Starting Extarction Process for video ID: {video_id}")

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
                            # Print the full aria-label text for debugging purposes
                            print(f"Full aria-label text: {aria_label}")
                            
                            # Use regex to extract only the number from the aria-label (e.g., "like this video along with 2,550,860 other people")
                            match = re.search(r'(\d[\d,]*)', aria_label)
                            if match:
                                # Replace commas and get the numeric value
                                likes = match.group(1)
                            else:
                                likes = "0"  # Default if no valid number is found
                        else:
                            likes = "0"  # Default if the aria-label attribute is missing
                    else:
                        likes = "0"  # Default if the like button itself is missing

                    # Extract Heatmap SVG
                    heatmap_svg = await extract_heatmap_svgs(page)

                    # Ensure unique description and links to avoid repetition
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


                    # XPath to specifically target the first <span> within the yt-formatted-string under the comments section
                    comments_xpath = "//div[@id='leading-section']//h2[@id='count']//yt-formatted-string/span[1]/text()"
                    
                    # Extract the number of comments using the refined XPath expression
                    comment_count_element = tree.xpath(comments_xpath)

                    # Extract the text content of the element
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

        ## Links
        {','.join(links)}

        ## Heatmap SVG
        ![Heatmap]({heatmap_svg})

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

                except PlaywrightTimeoutError:
                    print(f"Failed to find the button with selector: {selector}")

                await context.close()
                await browser.close()

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


        async def extract_comments(video_id, limit=20):
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

        # Process videos and download zip file
zip_file_path = asyncio.run(process_videos(video_ids))
st.success("Processing completed!")
with open(zip_file_path, "rb") as f:
    st.download_button("Download Markdown Files", f, file_name=zip_file_path)
        
        # Display output in tabs
tab_names = list(md_files.keys())
tabs = st.tabs(tab_names)
for tab, (title, md_content) in zip(tabs, md_files.items()):
    with tab:
        st.markdown(md_content)
