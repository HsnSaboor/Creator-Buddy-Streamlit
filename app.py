import streamlit as st
import subprocess
import os
import asyncio
from playwright.async_api import async_playwright

# Async function to run Playwright test
async def run_playwright_test():
    """Run Playwright to test the Streamlit app"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Set headless=True if you don't need a visible browser
        page = await browser.new_page()

        try:
            # Navigate to Playwright's website for testing
            await page.goto("http://playwright.dev")

            # Check the title of the page
            title = await page.title()

            # Write the title to Streamlit
            st.write(f"Page Title: {title}")

            # Take a screenshot
            await page.screenshot(path="playwright_test_screenshot.png")
            st.write("Screenshot taken and saved as 'playwright_test_screenshot.png'.")
        
        except Exception as e:
            st.write(f"Error during Playwright test: {e}")
        
        finally:
            # Close the browser
            await browser.close()

# Streamlit App UI
st.title("Streamlit App with Playwright Testing")
st.write("This app runs a Playwright test on itself asynchronously.")

# Button to start the Playwright test
if st.button("Run Playwright Test"):
    st.write("Running Playwright test...")

    # Run the async Playwright test
    asyncio.run(run_playwright_test())
