import os
import streamlit as st
import asyncio
from playwright.async_api import async_playwright

os.system("playwright install")

st.write("Starting the test…")

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto("http://playwright.dev")
        title = await page.title()
        st.write(title)
        await browser.close()
        return title

if __name__ == '__main__':
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)
    title=loop.run_until_complete(main())
    print(title)
