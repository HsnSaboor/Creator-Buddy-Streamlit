import asyncio
import logging
import random
from playwright.async_api import async_playwright

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

# Global variables to store browser and context
browser = None
context = None
page = None

async def initialize_browser():
    global browser, context, page
    if browser is None:
        logging.info("Initializing browser and caching YouTube...")
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

            # Cache YouTube by running a sample video
            sample_video_url = "https://www.youtube.com/watch?v=kXTyejeVwr8"
            await page.goto(sample_video_url, wait_until="domcontentloaded", timeout=60000)

            # Ensure the page is fully loaded
            await page.wait_for_load_state('networkidle')

            logging.info("YouTube cached successfully.")
    return browser, context, page

async def close_browser():
    global browser, context, page
    if browser:
        await context.close()
        await browser.close()
        browser = None
        context = None
        page = None
