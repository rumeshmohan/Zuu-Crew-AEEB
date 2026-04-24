# memory/vector_db.py

import os
import sys
import json
import time
import requests
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from io import BytesIO
from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.models import (
    Distance,
    PointStruct,
    PayloadSchemaType,
    VectorParams,
)
from openai import OpenAI
from utils.config import get_config, get_api_key

# ── Paths & Config ────────────────────────────────────────────────────────────
current_dir = Path(__file__).resolve().parent
root_dir    = current_dir.parent

config = get_config()

CATALOG_PATH    = os.path.join(root_dir, "data", "catalog.json")
COLLECTION_NAME = config.get("rag.collection_name", "kapruka_catalog")
BATCH_SIZE      = config.get("rag.batch_size", 50)
PROVIDER        = config.get("provider.default", "groq")

# ── Embedding mode resolution ─────────────────────────────────────────────────
# embedding.provider controls the backend:
#   "clip"           → local CLIP model via HuggingFace transformers
#   anything else    → OpenAI-compatible API (cohere, openai, openrouter, gemini…)
#
# embedding.clip_mode (only when provider = clip) controls which vectors are built:
#   "text_only"      → CLIP text encoder only  → single "text" named vector
#   "text_image"     → CLIP text + image encoder → dual "text" + "image" named vectors

EMBEDDING_PROVIDER = config.get("embedding.provider", "clip")
CLIP_MODE          = config.get("embedding.clip_mode", "text_image")   # text_only | text_image

USE_CLIP = EMBEDDING_PROVIDER == "clip"

print(
    f"🔧 Embedding backend : {EMBEDDING_PROVIDER}"
    + (f" [{CLIP_MODE}]" if USE_CLIP else "")
)


# ══════════════════════════════════════════════════════════════════════════════
# API Embedding Helpers  (defined early — used during client setup below)
# ══════════════════════════════════════════════════════════════════════════════

def _get_api_vector_size(model_name: str) -> int:
    """Return the embedding dimension for the given API model name."""
    name = model_name.lower()
    if "small" in name or "ada" in name:
        return 1536
    elif "large" in name:
        return 3072
    elif "cohere" in name or "embed-english" in name or "embed-multilingual" in name:
        return 1024
    elif "gemini" in name:
        return 768
    return 1536


# ── Browser-like headers to avoid 403s from Kapruka CDN ──────────────────────
IMAGE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer":         "https://www.kapruka.com/",
    "Accept":          "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ══════════════════════════════════════════════════════════════════════════════
# API Client Setup
# ══════════════════════════════════════════════════════════════════════════════

def _get_base_url(provider_name: str) -> str | None:
    """Return the OpenAI-compatible base URL for the given provider, or None for default OpenAI."""
    urls = {
        "ollama":      "http://localhost:11434/v1",
        "openrouter":  "https://openrouter.ai/api/v1",
        "cohere":      "https://api.cohere.ai/compatibility/v1",
        "groq":        "https://api.groq.com/openai/v1",
        "gemini":      "https://generativelanguage.googleapis.com/v1beta/openai/",
    }
    return urls.get(provider_name)


# Chat client — always needed for LLM auto-tagging during ingest
try:
    _chat_api_key = "ollama-local" if PROVIDER == "ollama" else get_api_key(PROVIDER)
except ValueError as e:
    raise ValueError(f"API setup error: {e}")

chat_client = OpenAI(api_key=_chat_api_key, base_url=_get_base_url(PROVIDER))
CHAT_MODEL  = config.get_model(PROVIDER, "general")

# Embedding client — only needed when NOT using CLIP
embedding_client: OpenAI | None = None
EMBEDDING_MODEL: str            = ""
EMBEDDING_DIM:   int            = 0

if not USE_CLIP:
    try:
        _emb_api_key = get_api_key(EMBEDDING_PROVIDER)
    except ValueError as e:
        raise ValueError(f"Embedding API key error: {e}")
    embedding_client = OpenAI(api_key=_emb_api_key, base_url=_get_base_url(EMBEDDING_PROVIDER))
    EMBEDDING_MODEL  = config.get_model(EMBEDDING_PROVIDER, config.get("embedding.tier", "default"), is_embedding=True)
    EMBEDDING_DIM    = _get_api_vector_size(EMBEDDING_MODEL)
    print(f"   Model  : {EMBEDDING_MODEL}  ({EMBEDDING_DIM}-dim)")

# Qdrant client
DB_PATH       = os.path.join(root_dir, "qdrant_db")
qdrant_client = QdrantClient(path=DB_PATH)


def generate_api_embedding(text: str) -> list[float]:
    """Return the embedding vector for a single text string via the configured API provider."""
    assert embedding_client is not None, "embedding_client is None — provider is set to 'clip'."
    try:
        response = embedding_client.embeddings.create(input=text, model=EMBEDDING_MODEL)
        return response.data[0].embedding
    except Exception as e:
        print(f"⚠️ Embedding API error: {e}")
        return [0.0] * EMBEDDING_DIM


# ══════════════════════════════════════════════════════════════════════════════
# CLIP Embedding Helpers
# ══════════════════════════════════════════════════════════════════════════════

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_DIM      = 512   # fixed output dimension for this CLIP model

_clip_model:     CLIPModel | None     = None
_clip_processor: CLIPProcessor | None = None


def _get_clip():
    """Load CLIP model and processor once, then cache in module-level globals."""
    global _clip_model, _clip_processor
    if _clip_model is None:
        print("🔄 Loading CLIP model (first run may download ~600 MB)...")
        _clip_model     = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        _clip_model.eval()
        print("✅ CLIP model loaded.")
    return _clip_model, _clip_processor


def clip_text_embedding(text: str) -> list[float]:
    """
    Encode a text string with CLIP's text encoder.
    Returns a normalised 512-dim vector.
    """
    model, processor = _get_clip()
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,   # CLIP's hard token limit
    )
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        if hasattr(emb, "pooler_output"):
            emb = emb.pooler_output
        elif not isinstance(emb, torch.Tensor):
            emb = emb[0]
    emb = emb / emb.norm(dim=-1, keepdim=True)   # L2 normalise
    return emb[0].tolist()


def clip_image_embedding(image_url: str) -> list[float] | None:
    """
    Download an image from image_url and encode it with CLIP's image encoder.
    Returns a normalised 512-dim vector, or None if the image can't be fetched.
    Uses browser-like headers to avoid 403 blocks from Kapruka's CDN.
    """
    if not image_url:
        return None
    try:
        resp = requests.get(image_url, headers=IMAGE_HEADERS, timeout=15)
        resp.raise_for_status()
        img  = Image.open(BytesIO(resp.content)).convert("RGB")

        model, processor = _get_clip()
        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            if hasattr(emb, "pooler_output"):
                emb = emb.pooler_output
            elif not isinstance(emb, torch.Tensor):
                emb = emb[0]
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].tolist()

    except Exception as e:
        print(f"  ⚠️  Image embedding failed ({image_url[:60]}...): {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Unified Embedding Interface
# ══════════════════════════════════════════════════════════════════════════════

def embed_text(text: str) -> list[float]:
    """
    Return a text embedding using whichever backend is configured.
    CLIP mode → clip_text_embedding()
    API mode  → generate_api_embedding()
    """
    if USE_CLIP:
        return clip_text_embedding(text)
    return generate_api_embedding(text)


def get_vector_dim() -> int:
    """Return the embedding dimension for the active backend."""
    return CLIP_DIM if USE_CLIP else EMBEDDING_DIM


# ══════════════════════════════════════════════════════════════════════════════
# LLM Auto-Tagger
# ══════════════════════════════════════════════════════════════════════════════

def auto_tag_product_via_llm(name: str, desc: str, category: str, retries: int = 3) -> list:
    """Extract 3–5 metadata tags for a product via LLM, with retry logic for rate limits."""
    prompt = (
        f"Analyze this Kapruka product.\n"
        f"Name: {name}\nCategory: {category}\nDescription: {desc}\n\n"
        f"Extract a comma-separated list of 3 to 5 highly relevant metadata tags. "
        f"Focus on dietary specifics (e.g., nuts, vegetarian, eggless), core ingredients "
        f"(e.g., chocolate, fruit), or item types (e.g., roses, electronics).\n"
        f"Output ONLY the comma-separated tags. Do not include any other text."
    )

    for attempt in range(retries):
        try:
            response = chat_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=30,
            )
            content = response.choices[0].message.content.strip()
            return [tag.strip().lower() for tag in content.split(",") if tag.strip()]
        except Exception as e:
            print(f"⚠️ LLM tagging error (attempt {attempt + 1}/{retries}) for '{name}': {e}")
            time.sleep(5)

    return [category.lower()]


# ══════════════════════════════════════════════════════════════════════════════
# Collection Schema Helper
# ══════════════════════════════════════════════════════════════════════════════

def _build_vectors_config() -> dict:
    """
    Return the Qdrant named-vector schema for the active embedding mode.

    API / CLIP text_only → {"text": VectorParams(size=dim)}
    CLIP text_image      → {"text": VectorParams(size=512),
                             "image": VectorParams(size=512)}
    """
    dim = get_vector_dim()
    cfg = {"text": VectorParams(size=dim, distance=Distance.COSINE)}
    if USE_CLIP and CLIP_MODE == "text_image":
        cfg["image"] = VectorParams(size=CLIP_DIM, distance=Distance.COSINE)
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# Ingest
# ══════════════════════════════════════════════════════════════════════════════

def ingest_catalog() -> None:
    """
    Load catalog.json, embed every product, tag via LLM, and upsert into Qdrant.

    Embedding behaviour is controlled entirely by params.yaml:
      embedding.provider = clip  + clip_mode = text_only   → CLIP text vector only
      embedding.provider = clip  + clip_mode = text_image  → CLIP text + image vectors
      embedding.provider = cohere / openai / …             → API text vector only

    Resume-safe: existing point IDs are scrolled and skipped.
    To force a full rebuild, delete the collection first:
        qdrant_client.delete_collection(COLLECTION_NAME)
    """
    print(f"📦 Loading {CATALOG_PATH} ...")
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            catalog = json.load(f)
    except FileNotFoundError:
        print(f"❌ {CATALOG_PATH} not found!")
        return

    # ── Create collection if it doesn't exist yet ─────────────────────────────
    if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        vectors_config = _build_vectors_config()
        mode_label = (
            f"CLIP [{CLIP_MODE}]" if USE_CLIP
            else f"API [{EMBEDDING_PROVIDER} / {EMBEDDING_MODEL}]"
        )
        print(f"🆕 Creating collection '{COLLECTION_NAME}' — {mode_label}")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=vectors_config,
        )
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="tags",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    # ── Scroll through all existing point IDs ─────────────────────────────────
    print("🔍 Scanning existing points in Qdrant...")
    existing_ids = set()
    next_offset  = None

    while True:
        result, next_offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=next_offset,
            with_payload=False,
            with_vectors=False,
        )
        for point in result:
            existing_ids.add(point.id)
        if next_offset is None:
            break

    print(f"✅ Already ingested: {len(existing_ids)}/{len(catalog)} points.")

    if len(existing_ids) == len(catalog):
        print("🎉 All points already ingested. Nothing to do.")
        return

    # Point IDs are 1-based positional: id = i + 1
    missing = [(i, item) for i, item in enumerate(catalog) if (i + 1) not in existing_ids]
    print(f"🔁 Resuming ingest: {len(missing)} missing product(s) to process...\n")

    has_image_count = 0

    for idx, (i, item) in enumerate(missing):
        point_id = i + 1

        # ── Text vector ───────────────────────────────────────────────────────
        text_str = (
            f"Product: {item['name']}. "
            f"Category: {item['category']}. "
            f"Description: {item['description']}."
        )
        text_vec = embed_text(text_str)
        vectors  = {"text": text_vec}

        # ── Image vector (CLIP text_image mode only) ──────────────────────────
        image_vec = None
        if USE_CLIP and CLIP_MODE == "text_image":
            image_vec = clip_image_embedding(item.get("image_url"))
            if image_vec:
                vectors["image"] = image_vec
                has_image_count += 1

        # ── LLM tags ──────────────────────────────────────────────────────────
        time.sleep(1)
        tags = auto_tag_product_via_llm(item["name"], item["description"], item["category"])

        # ── Payload ───────────────────────────────────────────────────────────
        payload = {
            "name":         item.get("name"),
            "price":        item.get("price"),
            "description":  item.get("description"),
            "availability": item.get("availability"),
            "category":     item.get("category"),
            "product_url":  item.get("product_url"),
            "image_url":    item.get("image_url"),
            "has_image":    image_vec is not None,
            "tags":         tags,
        }

        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(id=point_id, vector=vectors, payload=payload)],
        )

        img_icon = "🖼️" if image_vec else ("📝" if USE_CLIP and CLIP_MODE == "text_image" else "")
        print(f"  [{idx + 1}/{len(missing)}] id={point_id} | {item['name'][:50]} {img_icon}")

        # Gentle rate-limit pause every 10 items
        if (idx + 1) % 10 == 0:
            time.sleep(2)

    print(f"\n🎉 Ingest complete!")
    print(f"   📝 Text vectors   : {len(missing)}/{len(missing)}")
    if USE_CLIP and CLIP_MODE == "text_image":
        print(f"   🖼️ Image vectors  : {has_image_count}/{len(missing)}")
    print(f"   📦 Total in collection : {len(existing_ids) + len(missing)}/{len(catalog)}")


# ══════════════════════════════════════════════════════════════════════════════
# Retrieval
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_products(query: str, limit: int = 5, search_filter: rest.Filter = None) -> list:
    """
    Embed the query and search Qdrant. Retrieval strategy depends on the active mode:

      API / CLIP text_only  → single "text" vector search
      CLIP text_image       → dual "text" + "image" search, merged and deduplicated
                              (cross-modal: a text query can match products by visual similarity)

    Returns a list of product payloads sorted by score descending.
    """
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        print(f"⚠️ Collection '{COLLECTION_NAME}' does not exist. Run ingest first.")
        return []

    query_vec = embed_text(query)
    threshold = config.get("rag.score_threshold", 0.25)

    # ── Text vector search (always performed) ─────────────────────────────────
    try:
        text_hits = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            using="text",
            query_filter=search_filter,
            limit=limit,
            score_threshold=threshold,
        ).points
    except Exception as e:
        print(f"⚠️ Text vector search error: {e}")
        text_hits = []

    # ── Image vector search (CLIP text_image mode only) ───────────────────────
    # CLIP's shared embedding space makes text→image cross-modal search valid.
    image_hits = []
    if USE_CLIP and CLIP_MODE == "text_image":
        try:
            image_hits = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vec,
                using="image",
                query_filter=search_filter,
                limit=limit,
                score_threshold=threshold - 0.05,   # slightly more lenient for visual matches
            ).points
        except Exception as e:
            print(f"⚠️ Image vector search error: {e}")

    # ── Merge & deduplicate (text hits take priority) ─────────────────────────
    seen_ids = {h.id for h in text_hits}
    merged   = list(text_hits)

    for h in image_hits:
        if h.id not in seen_ids:
            merged.append(h)
            seen_ids.add(h.id)

    merged.sort(key=lambda x: x.score, reverse=True)
    return [h.payload for h in merged[:limit]]


if __name__ == "__main__":
    ingest_catalog()