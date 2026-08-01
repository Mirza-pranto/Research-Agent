import os
import requests
from bs4 import BeautifulSoup
from typing import List, Optional
from dotenv import load_dotenv
from ddgs import DDGS
from schemas import ExtractedSource

load_dotenv()

# Initialize Firecrawl SDK if an API key is available
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

firecrawl_app = None
if FIRECRAWL_API_KEY:
    try:
        from firecrawl import Firecrawl
        firecrawl_app = Firecrawl(api_key=FIRECRAWL_API_KEY)
    except ImportError:
        print("[WARNING] firecrawl-py package not installed. Falling back to BeautifulSoup.")


def scrape_with_firecrawl(url: str) -> Optional[str]:
    """
    Uses Firecrawl API to convert a URL into clean, LLM-ready Markdown content.
    Handles dynamic JS, anti-bot protection, and main content extraction.
    """
    if not firecrawl_app:
        return None

    try:
        # Firecrawl v2 API call for Markdown formatting
        scrape_result = firecrawl_app.scrape(
            url, 
            formats=["markdown"],
            only_main_content=True
        )
        if isinstance(scrape_result, dict):
            return scrape_result.get("markdown", "")
        elif hasattr(scrape_result, "markdown"):
            return scrape_result.markdown
    except Exception as e:
        print(f"[Firecrawl Error] Failed to scrape {url}: {e}")
    return None


def scrape_with_bs4(url: str, timeout: int = 10) -> str:
    """
    Fallback local HTML scraper using requests and BeautifulSoup.
    Strips non-essential elements (scripts, styles, navs) and returns cleaned text.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Strip non-text elements
        for script in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            script.decompose()

        # Get plain text
        text = soup.get_text(separator="\n")
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = "\n".join(chunk for chunk in chunks if chunk)

        # Truncate to avoid context window explosion (max ~3,000 characters)
        return cleaned_text[:3000]

    except Exception as e:
        print(f"[BS4 Error] Failed to fetch {url}: {e}")
        return ""


def execute_web_search(queries: List[str], max_results_per_query: int = 2, deep_scrape: bool = True) -> List[ExtractedSource]:
    """
    Performs DuckDuckGo search and optionally deep-scrapes target URLs using Firecrawl/BS4.
    """
    sources: List[ExtractedSource] = []
    ddgs = DDGS()

    for q in queries:
        try:
            results = list(ddgs.text(q, max_results=max_results_per_query))
            for res in results:
                title = res.get("title", "")
                url = res.get("href", "")
                snippet = res.get("body", "")

                detailed_content = snippet

                # Optional Deep Scraping step
                if deep_scrape and url:
                    print(f"--- [DEEP SCRAPE] Fetching full content for: {url} ---")
                    
                    # Try Firecrawl first (LLM-optimized Markdown)
                    content = scrape_with_firecrawl(url)
                    
                    # Fallback to BeautifulSoup if Firecrawl is unavailable or fails
                    if not content:
                        content = scrape_with_bs4(url)

                    if content:
                        # Combine original snippet with deep scraped body
                        detailed_content = f"{snippet}\n\n[Full Page Summary/Content]:\n{content[:2000]}"

                source_obj = ExtractedSource(
                    title=title,
                    url=url,
                    snippet=detailed_content
                )
                sources.append(source_obj)

        except Exception as err:
            print(f"[Search Error] Failed query '{q}': {err}")

    return sources