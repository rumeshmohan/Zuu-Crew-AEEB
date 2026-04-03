import json
import re
import sys
import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, List, Set
from collections import deque
from urllib.parse import urljoin

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from markdownify import markdownify as md


class PrimeLandsWebCrawler:
    """
    Async web crawler using Playwright for JavaScript-rendered content.

    Performs BFS traversal up to a configurable depth and page limit,
    extracts property metadata, and saves results as JSONL.

    Args:
        base_url: Root URL to crawl (all links must start with this).
        max_depth: Maximum BFS depth from start URLs.
        exclude_patterns: URL substrings that should be skipped.
        jsonl_path: File path where crawled documents are saved as JSONL.
        max_pages: Maximum number of pages to crawl before stopping.
    """

    def __init__(
        self,
        base_url: str,
        max_depth: int,
        exclude_patterns: List[str],
        jsonl_path: str,
        max_pages: int = 50,
    ) -> None:
        self.base_url = base_url
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.exclude_patterns = exclude_patterns
        self.jsonl_path = jsonl_path
        self.visited: Set[str] = set()
        self.documents: List[Dict[str, Any]] = []

        Path(self.jsonl_path).parent.mkdir(parents=True, exist_ok=True)

    def should_crawl(self, url: str) -> bool:
        """
        Determine whether a URL should be crawled.

        Args:
            url: The URL to evaluate.

        Returns:
            True if the URL is eligible for crawling, False otherwise.
        """
        if url in self.visited:
            return False
        if not url.startswith(self.base_url):
            return False
        for pattern in self.exclude_patterns:
            if pattern in url:
                return False
        if re.search(r'\.(jpg|jpeg|png|gif|pdf|zip|exe)$', url, re.IGNORECASE):
            return False
        return True

    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Extract page content and property metadata from a parsed HTML page.

        Attempts clean extraction from known content containers first,
        falling back to raw body text if the result is too short.

        Args:
            soup: Parsed HTML of the page.
            url: URL of the page (used for property_id extraction).

        Returns:
            Dict with keys: 'title', 'url', 'metadata', 'content', 'links'.
        """
        clean_soup = BeautifulSoup(str(soup), 'html.parser')

        for tag in clean_soup(["script", "style", "noscript", "iframe", "svg", "header", "footer", "nav"]):
            tag.decompose()

        content_node = (
            clean_soup.find("div", class_="container") or
            clean_soup.find("div", class_="page-content") or
            clean_soup.find("main") or
            clean_soup.body
        )

        if content_node:
            raw_md = md(str(content_node), heading_style="ATX")
            clean_lines = [line.strip() for line in raw_md.splitlines() if len(line.strip()) > 5]
            clean_content = "\n".join(clean_lines)
        else:
            clean_content = ""

        if len(clean_content) < 50:
            print("      Warning: Clean extraction yielded short content — using raw text fallback.")
            clean_content = clean_soup.get_text(separator="\n", strip=True)

        search_text = soup.get_text(separator=' ', strip=True).lower()

        metadata: Dict[str, Any] = {
            "property_id": url.split('/')[-2] if len(url.split('/')) > 2 else None,
            "address":     None,
            "price":       "Contact for Price",
            "bedrooms":    None,
            "bathrooms":   None,
            "sqft":        None,
            "amenities":   None,
            "agent":       "Prime Lands",
        }

        price_match = re.search(r'([\d\,\.]+)\s*(lkr|rs|rupees)', search_text, re.IGNORECASE)
        if price_match:
            metadata['price'] = f"{price_match.group(1)} {price_match.group(2).upper()}"

        beds_match = re.search(r'(\d+)\s*(beds?|bedrooms?)', search_text, re.IGNORECASE)
        if beds_match:
            metadata['bedrooms'] = beds_match.group(1)

        baths_match = re.search(r'(\d+)\s*(baths?|bathrooms?)', search_text, re.IGNORECASE)
        if baths_match:
            metadata['bathrooms'] = baths_match.group(1)

        size_match = re.search(r'([\d\,\.]+)\s*(sqft|sq\.?\s*ft|perch|perches)', search_text, re.IGNORECASE)
        if size_match:
            metadata['sqft'] = size_match.group(0).strip()

        perch_match = re.search(r'(\d+)\s*(perch|perches)', search_text, re.IGNORECASE)
        if perch_match:
            metadata['sqft'] = f"{perch_match.group(1)} perches"

        amenities_match = re.findall(
            r'(swimming pool|gym|security|parking|garden|playground|clubhouse)',
            search_text, re.IGNORECASE
        )
        if amenities_match:
            metadata['amenities'] = list({a.lower() for a in amenities_match})

        links = []
        for a in soup.find_all('a', href=True):
            href = a.get('href', '')
            if href.startswith('/'):
                href = urljoin(self.base_url, href)
            href = href.split('#')[0].split('?')[0]
            if href.startswith(self.base_url) and href != url:
                links.append(href)

        return {
            "title":    soup.title.string.strip() if soup.title else "Untitled",
            "url":      url,
            "metadata": metadata,
            "content":  clean_content,
            "links":    list(set(links)),
        }

    async def crawl_async(
        self,
        start_urls: List[str],
        request_delay: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """
        Perform async BFS crawl using Playwright.

        Args:
            start_urls: List of URLs to begin crawling from.
            request_delay: Seconds to wait between page requests.

        Returns:
            List of extracted document dictionaries.
        """
        queue: deque = deque([(url, 0) for url in start_urls])

        # File opened once for the entire crawl to avoid repeated open/close
        # cycles causing file-locking errors on Windows.
        with open(self.jsonl_path, 'w', encoding='utf-8') as jsonl_file:

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                page.set_default_timeout(45000)

                while queue:
                    if len(self.documents) >= self.max_pages:
                        print(f"Reached max pages limit ({self.max_pages}). Stopping crawl.")
                        break

                    url, depth = queue.popleft()

                    if depth > self.max_depth or not self.should_crawl(url):
                        continue

                    try:
                        print(f"[{depth}] {url}")
                        self.visited.add(url)

                        await page.goto(url, wait_until="domcontentloaded")

                        try:
                            await page.wait_for_selector(
                                "div.container, div.page-content, main, div.property-details",
                                state="attached",
                                timeout=8000,
                            )
                            await page.wait_for_timeout(2000)
                        except Exception:
                            await page.wait_for_timeout(3000)

                        html = await page.content()
                        soup = BeautifulSoup(html, 'html.parser')

                        doc_data = self.extract_content(soup, url)
                        doc_data['depth_level'] = depth

                        if len(doc_data['content']) >= 50:
                            self.documents.append(doc_data)
                            # Flush immediately so data is persisted if the crawl crashes mid-run.
                            jsonl_file.write(json.dumps(doc_data, ensure_ascii=False) + '\n')
                            jsonl_file.flush()
                            print(f"   Saved ({len(doc_data['content'])} chars)")
                        else:
                            print(f"   Skipped — content too short (<50 chars).")

                        if depth < self.max_depth:
                            links_added = 0
                            queued_urls = {item[0] for item in queue}
                            for link in doc_data['links']:
                                if link not in self.visited and link not in queued_urls:
                                    queue.append((link, depth + 1))
                                    links_added += 1
                            if links_added > 0:
                                print(f"   Added {links_added} new URLs to queue.")

                        await asyncio.sleep(request_delay)

                    except Exception as e:
                        print(f"   Error crawling {url}: {str(e)[:120]}")
                        continue

                await browser.close()

        return self.documents

    def crawl(
        self,
        start_urls: List[str],
        request_delay: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """
        Synchronous entry point for crawling, compatible with Windows and Jupyter.

        On Windows, runs the async crawler in a dedicated thread to avoid
        event loop conflicts with Jupyter's existing loop.

        Args:
            start_urls: List of URLs to begin crawling from.
            request_delay: Seconds to wait between page requests.

        Returns:
            List of extracted document dictionaries.
        """
        if sys.platform == 'win32':
            results_container: List[List[Dict[str, Any]]] = []

            def runner() -> None:
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self.crawl_async(start_urls, request_delay))
                    results_container.append(result)
                finally:
                    loop.close()

            thread = threading.Thread(target=runner)
            thread.start()
            thread.join()
            return results_container[0] if results_container else []

        return asyncio.run(self.crawl_async(start_urls, request_delay))


__all__ = ['PrimeLandsWebCrawler']