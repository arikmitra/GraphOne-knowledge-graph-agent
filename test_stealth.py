"""
Phase V smoke test: fetch a bot-detection test page through the stealth
browser pool and save the rendered HTML for visual inspection.
 
Run with:  python test_stealth.py
Then open the saved stealth_check.html in a normal browser and check that
rows like "WebDriver", "Chrome", "Plugins Length" read green/pass.
"""
import asyncio
from src.scrapers.anti_bot import StealthBrowserPool
 
async def main():
    async with StealthBrowserPool(pool_size=1, headless=True) as pool:
        html = await pool.fetch_rendered(
            "https://bot.sannysoft.com", wait_until="networkidle"
        )
        print("fetched", len(html) if html else 0, "bytes")
        with open("stealth_check.html", "w", encoding="utf-8") as f:
            f.write(html or "")
        print("saved to stealth_check.html")
 
 
if __name__ == "__main__":
    asyncio.run(main())