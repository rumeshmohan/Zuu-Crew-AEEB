import argparse
import json
import time
import logging
from pathlib import Path
from typing import Any
from playwright.sync_api import sync_playwright, Page

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SCRAPER_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRAPER_DIR.parent
OUTPUT_PATH  = PROJECT_ROOT / "data" / "catalog.json"

TARGET_ITEMS = 10000
WAIT_TIME    = 2.0

SELECTOR_CARD        = ".product-card, .item-card, [class*='catalogueV2'], .catalogueV2-item"
SELECTOR_NAME        = ".catalogueV2heading, .product-title, h2, h3"
SELECTOR_PRICE       = ".catalogueV2converted, .CatalogueV2price, .price, .product-price"
SELECTOR_LINK        = "a"
SELECTOR_DESC        = ".catalogueV2desc, .product-desc, .description, .subtitle, p.summary"
SELECTOR_UNAVAILABLE = ".catalogue-sold-out, .catalogueSoldOut, .out-of-stock"

SEED_URLS = [
    "https://www.kapruka.com/online/grocery",
    "https://www.kapruka.com/online/electronics",
    "https://www.kapruka.com/online/clothing",
    "https://www.kapruka.com/online/home_lifestyle",
    "https://www.kapruka.com/online/cosmetics",
    "https://www.kapruka.com/online/ayurvedic",
    "https://www.kapruka.com/online/household",
    "https://www.kapruka.com/online/toys",
    "https://www.kapruka.com/online/sports",
    "https://www.kapruka.com/online/cakes",
    "https://www.kapruka.com/online/chocolates",
    "https://www.kapruka.com/online/flowers",
    "https://www.kapruka.com/online/personalized_gifts",
    "https://www.kapruka.com/online/perfumes",
    "https://www.kapruka.com/online/books",
    "https://www.kapruka.com/online/handbags",
    "https://www.kapruka.com/online/shoes",
    "https://www.kapruka.com/online/watches",
    "https://www.kapruka.com/online/jewelry",
    "https://www.kapruka.com/online/stationery",
]


def _make_browser_page(p: Any) -> tuple[Any, Any, Any]:
    """Launch a headless Chromium browser and return (browser, context, page)."""
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(
        viewport={"width": 1280, "height": 900},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    )
    return browser, context, context.new_page()


def parse_products_chunk(
    page: Page, category: str, id_counter: int, seen_names: set[str]
) -> list[dict[str, Any]]:
    """Extract all new product dicts from the current page, skipping already-seen names."""
    cards = page.query_selector_all(SELECTOR_CARD)
    products = []
    local_id = id_counter

    for card in cards:
        try:
            name_el = card.query_selector(SELECTOR_NAME)
            name = name_el.inner_text().strip() if name_el else None
            if not name:
                continue

            dedup_name = name.lower().replace(" ", "")
            if dedup_name in seen_names:
                continue

            price_el = card.query_selector(SELECTOR_PRICE)
            price = price_el.inner_text().strip() if price_el else None
            if not price:
                continue

            anchor = card.query_selector(SELECTOR_LINK)
            href = anchor.get_attribute("href") if anchor else ""
            if href and href.startswith("/"):
                href = f"https://www.kapruka.com{href}"

            desc_el = card.query_selector(SELECTOR_DESC)
            description = desc_el.inner_text().strip() if desc_el else f"Buy {name} online from Kapruka in Sri Lanka."

            unavail_el = card.query_selector(SELECTOR_UNAVAILABLE)
            availability = "Out of Stock" if unavail_el else "In Stock"

            clean_cat = category.split("/")[-1].split("?")[0] if "/" in category else category
            for main_cat in ["cakes", "grocery", "cosmetics", "clothing", "electronics", "home", "household"]:
                if main_cat in category.lower():
                    clean_cat = main_cat
                    break

            seen_names.add(dedup_name)
            products.append({
                "id":           f"KAP_{local_id}",
                "category":     clean_cat,
                "name":         name,
                "price":        price,
                "description":  description,
                "availability": availability,
                "product_url":  href or "",
            })
            local_id += 1

        except Exception:
            continue

    return products


def save_catalog(catalog: list[dict[str, Any]]) -> None:
    """Persist the catalog list to OUTPUT_PATH as formatted JSON."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)


def scrape_massive_catalog(fresh: bool = False) -> None:
    """Crawl Kapruka category pages and accumulate products until TARGET_ITEMS is reached."""
    if fresh and OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
        catalog, seen_names, id_counter = [], set(), 1
        logger.info("🗑️ Starting fresh crawl...")
    elif OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        seen_names = {p["name"].lower().replace(" ", "") for p in catalog}
        id_counter = len(catalog) + 1
        logger.info(f"♻️ Resuming: {len(catalog)} items loaded. Starting with ID: KAP_{id_counter}")
    else:
        catalog, seen_names, id_counter = [], set(), 1

    with sync_playwright() as p:
        browser, context, page = _make_browser_page(p)

        queue   = SEED_URLS.copy()
        visited = set()

        while queue and len(catalog) < TARGET_ITEMS:
            current_url = queue.pop(0)
            base_url = current_url.split("?")[0]
            if base_url in visited:
                continue
            visited.add(base_url)

            logger.info(f"🕸️ Spider mapping: {base_url}")

            try:
                page.goto(base_url, timeout=60_000, wait_until="domcontentloaded")
                time.sleep(WAIT_TIME)

                new_links_found = 0
                for link in page.query_selector_all("a"):
                    href = link.get_attribute("href")
                    if href and ("/online/" in href or "/price/" in href):
                        full_link  = href if href.startswith("http") else f"https://www.kapruka.com{href}"
                        clean_link = full_link.split("?")[0]
                        if clean_link not in visited and clean_link not in queue:
                            queue.append(clean_link)
                            new_links_found += 1

                if new_links_found > 0:
                    logger.info(f"   - 🧭 Discovered {new_links_found} hidden category pages!")

                for _ in range(4):
                    page.evaluate("window.scrollBy(0, 800);")
                    time.sleep(0.5)

                slug      = base_url.split("/")[-1]
                new_items = parse_products_chunk(page, slug, id_counter, seen_names)

                if new_items:
                    catalog    += new_items
                    id_counter += len(new_items)
                    logger.info(f"   - 🛒 Extracted {len(new_items)} items. Total: {len(catalog):,} / {TARGET_ITEMS:,}")
                    save_catalog(catalog)
                else:
                    logger.info("   - No new items on this page. Moving to next in queue...")

            except Exception as e:
                logger.warning(f"   - Error on {base_url}: {e}")
                continue

        browser.close()

    print(f"✅ Final total: {len(catalog):,} unique products collected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    scrape_massive_catalog(fresh=parser.parse_args().fresh)