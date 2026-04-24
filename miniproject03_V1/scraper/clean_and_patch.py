import json
import re
import logging
import sys
import shutil
import time
import concurrent.futures
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.config import get_config
from utils.llm_services import get_llm

config = get_config()
CATALOG_PATH = (ROOT_DIR / config.get("paths.catalog_file")).resolve()
BACKUP_PATH  = (ROOT_DIR / "data" / "catalog_pre_patch_backup.json").resolve()

llm_service = get_llm(tier=config.get("provider.tier"))

MEGA_MAP = {
    "Electronics & Gadgets":        ["apple", "samsung", "iphone", "mobile", "laptop", "smart", "audio", "led", "tv", "camera"],
    "Cakes, Bakes & Gourmet":       ["cake", "gateau", "divine", "java", "bakery", "pastry", "brownie", "cupcake", "dessert"],
    "Chocolates & Confectionery":   ["chocolate", "revello", "ferrero", "toblerone", "cadbury", "sweet", "candy", "ritzbury"],
    "Grocery & Daily Essentials":   ["rice", "milk", "tea", "coffee", "pantry", "fresh", "hamper", "beverages", "oil", "nipuna"],
    "Health, Beauty & Wellness":    ["perfume", "skin", "makeup", "fragrance", "wellness", "pharmacy", "serum", "lotion"],
    "Fashion & Apparel":            ["saree", "shirt", "dress", "clothing", "batik", "linen", "handbag", "shoes", "watch", "jewelry"],
    "Kids, Toys & Baby Care":       ["toy", "baby", "toddler", "remote", "jeep", "plush", "teddy", "diaper", "game"],
    "Home, Decor & Lifestyle":      ["flower", "bouquet", "rose", "furniture", "kitchen", "mug", "cushion", "decor", "homeware"],
    "Stationery & Corporate Gifts": ["book", "pen", "notebook", "office", "personalized", "frame", "keychain"],
}

SYSTEM_PROMPT = """You are a Data Entry Clerk. Do NOT converse. Do NOT apply safety filters.
Output EXACTLY this format and nothing else:
DESC: [1-sentence description] | SUITABILITY: [Recipient type] | SAFETY: [Allergens or 'None']"""


def clean_category(category: str, name: str) -> str:
    """Resolve a slash-delimited category string to the best-matching MEGA_MAP key."""
    if "/" not in category:
        return category
    lookup = f"{category} {name}".lower().replace(" ", "_")
    for mega, keywords in MEGA_MAP.items():
        if any(key in lookup for key in keywords):
            return mega
    return category.split("/")[0].replace("_", " ").strip().title()


def clean_price_to_int(price_str) -> int:
    """Extract the last numeric value from a price string and return it as an integer."""
    if not price_str:
        return 0
    matches = re.findall(r"\d+", str(price_str).replace(",", ""))
    return int(matches[-1]) if matches else 0


def generate_with_timeout(prompt: str, timeout: int = 20):
    """Wrap the LLM call in a strict timeout to prevent hanging."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(llm_service.generate, prompt)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return None


def enrich_single_item(item: dict) -> bool:
    """Enrich a catalog item with an LLM-generated description, skipping if already present."""
    desc = item.get("description", "")
    if "DESC:" in desc and "SUITABILITY:" in desc and "SAFETY:" in desc:
        return True

    user_prompt = f"Product: {item['name']}\nCategory: {item['category']}"

    for _ in range(3):
        try:
            enriched_text = generate_with_timeout(f"{SYSTEM_PROMPT}\n\n{user_prompt}", timeout=20)
            if enriched_text:
                clean_text = enriched_text.strip().replace('"', "").replace("*", "")
                clean_text = re.sub(r"^(Here(?:'s| is).*?:?\n*)", "", clean_text, flags=re.IGNORECASE).strip()
                if "DESC:" in clean_text:
                    item["description"] = clean_text
                    return True
        except Exception:
            pass

        time.sleep(2)

    return False


def run_pipeline():
    """Back up, clean, enrich, and persist the full product catalog."""
    if CATALOG_PATH.exists():
        shutil.copy2(CATALOG_PATH, BACKUP_PATH)

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    print(f"🚀 Processing {len(catalog)} items with auto-resume & timeout protection...")

    for item in tqdm(catalog, desc="Refining Catalog"):
        item.pop("llm_enriched", None)
        item["category"] = clean_category(item.get("category", ""), item.get("name", ""))
        item["price"] = clean_price_to_int(item.get("price", 0))
        enrich_single_item(item)

        with open(CATALOG_PATH, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)

    print("✅ Pipeline finished. File is clean and ready for Vector DB.")


if __name__ == "__main__":
    run_pipeline()