import re
import os
import csv
import asyncio

from pyppeteer import launch
from pyppeteer_stealth import stealth

from constants import SEARCH_QUERIES


# Web scraping twitter is difficult because 
# they have a bunch of anti bot measures.
# So some things we have to do are:
# 
# => mask user agent, puppeteer metadata, etc (done by stealth)
# => log in with a real account
# => run it without headless mode
# 
# Even with these measures, it seems like my account
# got soft banned after scraping only like 50 or so
# tweets.
# 
# Will check again after a few days to see if it's
# working again.


async def handle_login(page, email, username, password):
    print("Handling login...")
    
    # enter email
    await page.waitForSelector('input[autocomplete="username"]')
    await page.type('input[autocomplete="username"]', email)
    await page.keyboard.press('Enter')
    
    # check for and handle username step if it appears
    try:
        username_selector = 'input[data-testid="ocfEnterTextTextInput"]'
        await page.waitForSelector(username_selector, {'timeout': 5000})
        print("Username step detected, entering username...")
        await page.type(username_selector, username)
        await page.keyboard.press('Enter')
    except:
        print("No username step required, proceeding to password...")
    
    # wait for and enter password
    await page.waitForSelector('input[name="password"]')
    await page.type('input[name="password"]', password)
    await page.keyboard.press('Enter')
    
    # wait for login to complete
    await page.waitForNavigation()
    print("Login completed")

async def setup_browser():
    browser = await launch(
        executablePath='/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
        headless=False,
        args=['--no-sandbox', '--disable-setuid-sandbox', '--enable-javascript', 
              '--disable-blink-features=AutomationControlled']
    )
    
    page = await browser.newPage()
    await stealth(page)
    await page.setJavaScriptEnabled(True)
    
    return browser, page

async def collect_tweets(page, max_tweets=10):
    tweets = []
    consecutive_empty_scrolls = 0
    max_empty_scrolls = 3  # because at one point, the tweets stop loading
    
    while len(tweets) < max_tweets and consecutive_empty_scrolls < max_empty_scrolls:
        print(f"Scrolling to bottom of page, {len(tweets)} tweets collected")
        previous_tweet_count = len(tweets)
        
        await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
        await page.waitForSelector('[data-testid="primaryColumn"]', {'timeout': 60000})
        
        # add a small delay to allow tweets to load
        await page.waitFor(1000)
        
        new_tweets = await page.evaluate('''
            () => {
                const tweetElements = document.querySelectorAll('article[data-testid="tweet"]');
                return Array.from(tweetElements).map(tweet => {
                    const textElement = tweet.querySelector('div[data-testid="tweetText"]');
                    return textElement ? textElement.innerText : '';
                });
            }
        ''')
        
        for tweet_text in new_tweets:
            if tweet_text and tweet_text not in [t['text'] for t in tweets]:
                hashtags = ' '.join(re.findall(r'#(\w+)', tweet_text))
                if hashtags:    
                    tweets.append({'text': tweet_text, 'tags': hashtags})
                    print(f"Collected {len(tweets)} tweets")
                
            if len(tweets) >= max_tweets:
                break
        
        # check if we got any new tweets in this scroll
        if len(tweets) == previous_tweet_count:
            consecutive_empty_scrolls += 1
            print(f"No new tweets found. Attempt {consecutive_empty_scrolls}/{max_empty_scrolls}")
        else:
            consecutive_empty_scrolls = 0
    
    print(f"Finished collecting {len(tweets)} tweets")
    return tweets

async def save_tweets_to_csv(tweets, filename='tweets.csv'):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['text', 'tags'])
        writer.writeheader()
        writer.writerows(tweets)
    print(f"Saved {len(tweets)} tweets to {filename}")

async def scrape_twitter(email, username, password, search_query = None):
    browser = None
    try:
        browser, page = await setup_browser()
        
        # navigate to twitter login
        print("Navigating to Twitter...")
        await page.goto('https://twitter.com/login', {
            'waitUntil': 'networkidle0',
            'timeout': 60000
        })
        
        # handle login
        await handle_login(page, email, username, password)
        
        # navigate to search query
        if search_query:
            await page.goto(f'https://twitter.com/search?q={search_query}&src=typed_query&f=live', {
                'waitUntil': 'networkidle2',
                'timeout': 60000
            })
        else:
            await page.goto('https://twitter.com/', {
                'waitUntil': 'networkidle2',
                'timeout': 60000
            })
        
        # collect tweets
        tweets = await collect_tweets(page)
        
        # save tweets
        await save_tweets_to_csv(tweets, f"{search_query}.csv")
        
    finally:
        if browser:
            await browser.close()

# run the scraper
if __name__ == "__main__":
    EMAIL = os.getenv("TWITTER_EMAIL")
    USERNAME = os.getenv("TWITTER_USERNAME")
    PASSWORD = os.getenv("TWITTER_PASSWORD")
    
    for search_query in SEARCH_QUERIES:
        asyncio.get_event_loop().run_until_complete(
            scrape_twitter(EMAIL, USERNAME, PASSWORD, search_query)
        )
