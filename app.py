import streamlit as st
import subprocess
import os
import asyncio
from playwright.async_api import async_playwright

# Async function to run Playwright test
async def run_playwright_test():
    """Run Playwright to test the Streamlit app and take a screenshot"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)  # Set headless=True if you want headless browsing
        page = await browser.new_page()

        try:
            # Navigate to a webpage (e.g., Playwright's homepage)
            await page.goto("http://playwright.dev")

            # Check the title of the page
            title = await page.title()
            st.write(f"Page Title: {title}")

            # Take a screenshot
            screenshot_path = "playwright_test_screenshot.png"
            await page.screenshot(path=screenshot_path)
            st.write("Screenshot taken and saved as 'playwright_test_screenshot.png'.")

            # Display the screenshot in the app
            with open(screenshot_path, "rb") as file:
                screenshot_data = file.read()
                st.image(screenshot_data, caption="Screenshot", use_column_width=True)

                # Provide a download button to download the screenshot
                st.download_button(
                    label="Download Screenshot",
                    data=screenshot_data,
                    file_name="playwright_test_screenshot.png",
                    mime="image/png"
                )

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
