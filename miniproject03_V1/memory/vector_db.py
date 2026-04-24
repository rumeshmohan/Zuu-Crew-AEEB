import os
import sys
import json
import time
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.models import VectorParams, Distance, PointStruct, PayloadSchemaType
from openai import OpenAI
from utils.config import get_config, get_api_key

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))
config = get_config()

CATALOG_PATH       = os.path.join(root_dir, "data", "catalog.json")
COLLECTION_NAME    = config.get("rag.collection_name", "kapruka_catalog")
BATCH_SIZE         = config.get("rag.batch_size", 50)
PROVIDER           = config.get("provider.default", "groq")
EMBEDDING_PROVIDER = config.get("embedding.provider", "cohere")

try:
    embedding_api_key = "ollama-local" if EMBEDDING_PROVIDER == "ollama" else get_api_key(EMBEDDING_PROVIDER)
    chat_api_key      = "ollama-local" if PROVIDER == "ollama" else get_api_key(PROVIDER)
except ValueError as e:
    raise ValueError(f"API setup error: {e}")


def get_base_url(provider_name: str) -> str | None:
    """Return the OpenAI-compatible base URL for the given provider, or None for default OpenAI."""
    urls = {
        "ollama":      "http://localhost:11434/v1",
        "openrouter":  "https://openrouter.ai/api/v1",
        "cohere":      "https://api.cohere.ai/compatibility/v1",
        "groq":        "https://api.groq.com/openai/v1",
        "gemini":      "https://generativelanguage.googleapis.com/v1beta/openai/",
    }
    return urls.get(provider_name)


embedding_client = OpenAI(api_key=embedding_api_key, base_url=get_base_url(EMBEDDING_PROVIDER))
EMBEDDING_MODEL  = config.get_model(EMBEDDING_PROVIDER, config.get("embedding.tier", "default"), is_embedding=True)

chat_client = OpenAI(api_key=chat_api_key, base_url=get_base_url(PROVIDER))
CHAT_MODEL  = config.get_model(PROVIDER, "general")

DB_PATH       = os.path.join(root_dir, "qdrant_db")
qdrant_client = QdrantClient(path=DB_PATH)


def get_vector_size(model_name: str) -> int:
    """Return the embedding dimension for the given model name."""
    name = model_name.lower()
    if "small" in name or "ada" in name:
        return 1536
    elif "large" in name:
        return 3072
    elif "cohere" in name or "embed-english" in name:
        return 1024
    elif "gemini" in name:
        return 768
    return 1536


def generate_embedding(text: str) -> list:
    """Return the embedding vector for a single text string, falling back to zeros on error."""
    try:
        response = embedding_client.embeddings.create(input=text, model=EMBEDDING_MODEL)
        return response.data[0].embedding
    except Exception as e:
        print(f"⚠️ Embedding generation error: {e}")
        return [0.0] * get_vector_size(EMBEDDING_MODEL)


def get_embeddings_batch(texts: list) -> list:
    """Return a list of embedding vectors for a batch of texts, ordered by input index."""
    response = embedding_client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def auto_tag_product_via_llm(name: str, desc: str, category: str, retries: int = 3) -> list:
    """Extract 3–5 metadata tags for a product via LLM, with retry logic for rate limits."""
    prompt = (
        f"Analyze this Kapruka product.\n"
        f"Name: {name}\nCategory: {category}\nDescription: {desc}\n\n"
        f"Extract a comma-separated list of 3 to 5 highly relevant metadata tags. "
        f"Focus on dietary specifics (e.g., nuts, vegetarian, eggless), core ingredients (e.g., chocolate, fruit), "
        f"or item types (e.g., roses, electronics).\n"
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


def ingest_catalog() -> None:
    """Load the catalog JSON, embed all products, tag them via LLM, and upsert into Qdrant."""
    print(f"📦 Loading {CATALOG_PATH} for vectorization and LLM tagging...")
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            catalog = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {CATALOG_PATH} not found!")
        return

    dim_size = get_vector_size(EMBEDDING_MODEL)

    if qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"🔄 Deleted old collection. Rebuilding for dimension size: {dim_size}...")

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim_size, distance=Distance.COSINE),
    )
    qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="tags",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    print(f"🧠 Generating embeddings ({EMBEDDING_MODEL}) and LLM tags ({CHAT_MODEL}) for {len(catalog)} items...")

    for i in range(0, len(catalog), BATCH_SIZE):
        batch = catalog[i:i + BATCH_SIZE]

        texts_to_embed = [
            f"Product: {item['name']}. Category: {item['category']}. Description: {item['description']}."
            for item in batch
        ]

        try:
            embeddings = get_embeddings_batch(texts_to_embed)
            points = []

            for j, item in enumerate(batch):
                payload = {
                    "name":         item.get("name"),
                    "price":        item.get("price"),
                    "description":  item.get("description"),
                    "availability": item.get("availability"),
                    "category":     item.get("category"),
                    "product_url":  item.get("product_url"),
                }

                time.sleep(1)
                payload["tags"] = auto_tag_product_via_llm(item["name"], item["description"], item["category"])
                points.append(PointStruct(id=i + j + 1, vector=embeddings[j], payload=payload))

            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"   ▶ Upserted items {i + 1} to {i + len(batch)}...")

            time.sleep(2)

        except Exception as e:
            print(f"⚠️ Error processing batch starting at index {i}: {e}")

    print("\n🎉 Long-term memory successfully populated with LLM-extracted metadata!")


def retrieve_products(query: str, limit: int = 5, search_filter: rest.Filter = None) -> list:
    """Embed the query and return the top matching product payloads from Qdrant."""
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        print(f"⚠️ Warning: Collection '{COLLECTION_NAME}' does not exist.")
        return []

    query_vector = generate_embedding(query)
    threshold = config.get("rag.score_threshold", 0.35)

    try:
        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=search_filter,
            limit=limit,
            score_threshold=threshold,
        ).points
        return [hit.payload for hit in results]
    except Exception as e:
        print(f"⚠️ Qdrant retrieval error: {e}")
        return []


if __name__ == "__main__":
    ingest_catalog()