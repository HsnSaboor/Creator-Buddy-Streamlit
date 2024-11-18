import asyncio
from browser_manager import initialize_browser
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from lxml import html, etree
from urllib.parse import parse_qs, urlparse
import random
import regex as re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
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
import os
import math
from typing import List, Dict, Optional, Any
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, Transcript, TranscriptList
import spacy
import dask.dataframe as dd
import uvloop
import streamlit as st
from PIL import Image

nltk.download('punkt')
nltk.download('stopwords')

os.system('playwright install')

# Configure uvloop for faster asyncio event loops
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

client = Groq(
    api_key='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'
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def extract_video_data(video_id, page):
    logging.info(f"Extracting video data for video ID: {video_id}")
    
    try:
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
    except Exception as e:
        logging.error(f"Error during video extraction: {e}")
        return {"error": str(e)}

def main():
    # Initialize the browser before starting the Streamlit app
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        browser, context, page = loop.run_until_complete(initialize_browser())
    except Exception as e:
        logging.error(f"Error during browser initialization: {e}")
        return

    # Start the Streamlit app
    st.title("YouTube Video Data Extractor")
    video_id = st.text_input("Enter the YouTube video ID:")

    if st.button("Extract Data"):
        if video_id:
            st.write("Extracting data...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                data = loop.run_until_complete(extract_video_data(video_id, page))
                st.json(data)
            except Exception as e:
                st.error(f"Error during extraction: {e}")
            finally:
                loop.close()
        else:
            st.write("Please enter a valid YouTube video ID.")

    # Close the browser when the Streamlit app is closed
    st.stop()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(close_browser())
    except Exception as e:
        logging.error(f"Error closing browser: {e}")
    finally:
        loop.close()

if __name__ == "__main__":
    main()
