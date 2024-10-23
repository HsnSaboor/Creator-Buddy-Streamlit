import streamlit as st
import subprocess
import os
import asyncio
from playwright.async_api import async_playwright
import platform
import psutil

# Async function to run Playwright test
async def run_playwright_test():
    """Run Playwright to test the Streamlit app and take a screenshot."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Set headless=True for headless browsing
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

# Function to get machine specifications
def get_machine_specs():
    """Returns a dictionary with machine specifications."""
    specs = {
        "Platform": platform.system(),
        "Platform Release": platform.release(),
        "Platform Version": platform.version(),
        "Architecture": platform.machine(),
        "Processor": platform.processor(),
        "CPU Cores": psutil.cpu_count(logical=True),
        "Memory (GB)": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "Disk Usage (Total)": f"{round(psutil.disk_usage('/').total / (1024 ** 3), 2)} GB",
        "Disk Usage (Used)": f"{round(psutil.disk_usage('/').used / (1024 ** 3), 2)} GB",
        "Disk Usage (Free)": f"{round(psutil.disk_usage('/').free / (1024 ** 3), 2)} GB"
    }
    
    # Handling shared CPU environments
    try:
        shared_cpu = os.environ.get('CPU')
        if shared_cpu:
            specs['Shared CPU Environment'] = "Yes"
        else:
            specs['Shared CPU Environment'] = "No"
    except Exception as e:
        specs['Shared CPU Environment'] = f"Unknown ({e})"
    
    return specs

# Function to run fastfetch and return its output
def run_neofetch():
    """Run neofetch and return its output."""
    try:
        # Run the fastfetch command and capture the output
        result = subprocess.run(['neofetch'], capture_output=True, text=True)
        return result.stdout
    except FileNotFoundError:
        return "neofetch is not installed. Please install it to see system information."
    except Exception as e:
        return f"Error running neofetch: {e}"

# Streamlit App UI
st.title("Streamlit App with Playwright Testing and System Specs")
st.write("This app runs a Playwright test and displays the machine specs.")

# Button to start the Playwright test
if st.button("Run Playwright Test"):
    st.write("Running Playwright test...")
    
    # Run the async Playwright test
    asyncio.run(run_playwright_test())

# Display Machine Specifications
st.subheader("Machine Specifications")
specs = get_machine_specs()
for key, value in specs.items():
    st.write(f"**{key}**: {value}")

# Display output of fastfetch
st.subheader("Fastfetch Output")
neofetch_output = run_neofetch()
st.text_area("Neofetch Output", value=neofetch_output, height=300)

