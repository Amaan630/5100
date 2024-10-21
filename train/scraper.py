import asyncio
from pyppeteer import launch

async def scrape_twitter():
    # Launch the browser
    browser = await launch()
    
    try:
        # Create a new page
        page = await browser.newPage()
        
        # Navigate to Twitter
        print("Navigating to Twitter...")
        await page.goto('https://twitter.com')
        
        # Wait for the page to load
        await page.waitForSelector('body')
        
        # Get the page title
        title = await page.title()
        print(f"Page title: {title}")
        
        # You can add more scraping logic here
        
    finally:
        # Close the browser
        await browser.close()

# Run the scraper
if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(scrape_twitter())