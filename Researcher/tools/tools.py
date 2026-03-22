import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import re
import numpy as np
import requests
import tiktoken
import os
from dotenv import load_dotenv


def get_web_urls(query: str, page: int = 1) -> str:
    """
    Scrapes DuckDuckGo for search results using a headless browser.

    Args:
        query (str): The search term.
        page (int): The page number to retrieve (1 page = 10 results).
                    Defaults to 1.

    Returns:
        str: A JSON string containing a list of dictionaries with 'title' and 'url'.
    """

    if query == "__describe__":
        return (
            "Use this tool to get top 10 urls from duckduck go for a given query.",
            '{"tool": "get_web_urls", "args": {"query": "what is Large language models", "page": 1}}'
        )


    # 1. Setup Chrome Options
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920,1080")
    # User agent to prevent bot detection
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    chrome_options.add_argument(f'user-agent={user_agent}')

    # 2. Initialize Driver
    # Note: In a production server, you might want to initialize this once globally
    # rather than per-function-call to save startup time.
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    results_list = []

    try:
        # 3. Navigate
        base_url = "https://duckduckgo.com/?q="
        formatted_query = query.replace(" ", "+")
        driver.get(base_url + formatted_query)

        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "article[data-testid='result']")))

        # 4. Pagination Logic (Simulating pages on infinite scroll)
        # We assume 10 results per 'page'.
        # If user asks for page 2, we need indexes 10-19.
        # If user asks for page 3, we need indexes 20-29.
        results_per_page = 10
        needed_count = page * results_per_page

        current_results = driver.find_elements(By.CSS_SELECTOR, "article[data-testid='result']")

        # If we don't have enough results yet and page > 1, try to load more
        while len(current_results) < needed_count:
            try:
                # Look for the "More Results" button
                more_btn = driver.find_element(By.ID, "more-results")
                if more_btn.is_displayed():
                    driver.execute_script("arguments[0].click();", more_btn)
                    time.sleep(1.5) # Allow dynamic load

                    # Re-fetch elements
                    current_results = driver.find_elements(By.CSS_SELECTOR, "article[data-testid='result']")
                else:
                    # No more button found, break loop
                    break
            except Exception:
                # If button interaction fails or doesn't exist, we break
                break

        # 5. Extract specific slice for the requested page
        start_index = (page - 1) * results_per_page
        end_index = start_index + results_per_page

        # Slice the WebElement list
        target_elements = current_results[start_index:end_index]

        for result in target_elements:
            try:
                link_element = result.find_element(By.CSS_SELECTOR, "a[data-testid='result-title-a']")
                title = link_element.text
                url = link_element.get_attribute("href")

                if title and url:
                    results_list.append({
                        "title": title,
                        "url": url
                    })
            except Exception:
                continue

    except Exception as e:
        return json.dumps({"error": str(e)})

    finally:
        driver.quit()

    # 6. Return JSON string
    return json.dumps(results_list, indent=2)

# -------- CONFIG --------
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
API_KEY = os.getenv("APIKEY")
if not API_KEY:
    raise ValueError("APIKEY not found in environment")

MODEL = os.getenv("MODEL")
if not MODEL:
    raise ValueError("MODEL not found in environment")

CHUNK_SIZE = 800
OVERLAP = 150
TOP_K = 20


# -------- TOKENIZER --------
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))


# -------- EMBEDDING --------
def embed(texts):
    resp = requests.post(
        EMBED_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": MODEL,
            "input": texts
        },
        timeout=30
    )
    data = resp.json()
    return [d["embedding"] for d in data["data"]]


# -------- UTIL --------
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def chunk_text(text):
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = start + CHUNK_SIZE
        chunk = text[start:end]

        # avoid cutting sentence
        if end < n:
            last_period = chunk.rfind(".")
            if last_period > CHUNK_SIZE * 0.5:
                end = start + last_period + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - OVERLAP

    return chunks


# -------- MAIN FUNCTION --------
def perform_deep_research(query: str, url_list: list[str] = None) -> str:
    """
    Pure RAG-based web research (no LLM summarization)

    Returns:
        str: high-quality filtered context
    """
    max_tokens=30000
    if url_list is None:
        url_list = []

    if query == "__describe__":
        return (
            "RAG-based research tool. Scrapes URLs, chunks content, embeds, and returns most relevant context.",
            '{"tool": "perform_deep_research", "args": {"query": "example", "url_list": ["url1","url2"]}}'
        )

    if not isinstance(url_list, list):
        return "Error: url_list must be a list of strings."

    # -------- CLEAN --------
    def clean_text(text: str) -> str:
        text = re.sub(r'[\n\r\t]+', ' ', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    # -------- SELENIUM --------
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
    except Exception as e:
        return f"Error initializing Webdriver: {e}"

    all_chunks = []
    chunk_sources = []

    try:
        for url in url_list:
            if not isinstance(url, str) or not url.strip():
                continue

            try:
                driver.set_page_load_timeout(20)
                driver.get(url)

                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )

                # scroll
                last_height = driver.execute_script("return document.body.scrollHeight")
                for _ in range(3):
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(1)
                    new_height = driver.execute_script("return document.body.scrollHeight")
                    if new_height == last_height:
                        break
                    last_height = new_height

                body_text = driver.execute_script("return document.body.innerText;")
                clean = clean_text(body_text)

                # chunk immediately
                chunks = chunk_text(clean)

                for c in chunks:
                    if len(c) > 200:
                        all_chunks.append(c)
                        chunk_sources.append(url)

            except Exception as e:
                print(f"Failed to scrape {url}: {e}")

    finally:
        driver.quit()

    if not all_chunks:
        return "Error: No usable content extracted."

    # -------- EMBEDDING --------
    chunk_embs = embed(all_chunks)
    query_emb = embed([query])[0]

    # -------- RETRIEVAL --------
    scores = [
        (i, cosine(query_emb, emb))
        for i, emb in enumerate(chunk_embs)
    ]

    top_indices = sorted(scores, key=lambda x: x[1], reverse=True)[:TOP_K]

    top_chunks = [all_chunks[i] for i, _ in top_indices]
    top_embs = [chunk_embs[i] for i, _ in top_indices]
    top_sources = [chunk_sources[i] for i, _ in top_indices]

    token_counts = [count_tokens(c) for c in top_chunks]

    # -------- MMR + TOKEN PACKING --------
    selected_chunks = []
    selected_sources = []
    selected_indices = []
    total_tokens = 0

    while True:
        best_score = -1
        best_idx = -1

        for i, emb in enumerate(top_embs):
            if i in selected_indices:
                continue

            relevance = cosine(query_emb, emb)
            diversity = max(
                [cosine(emb, top_embs[j]) for j in selected_indices],
                default=0
            )

            score = 0.7 * relevance - 0.3 * diversity

            chunk_tokens = token_counts[i]

            if total_tokens + chunk_tokens > max_tokens:
                continue

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx == -1:
            break

        selected_chunks.append(top_chunks[best_idx])
        selected_sources.append(top_sources[best_idx])
        selected_indices.append(best_idx)
        total_tokens += token_counts[best_idx]

    # -------- FINAL FORMAT --------
    output = []
    for chunk, src in zip(selected_chunks, selected_sources):
        output.append(f"[SOURCE: {src}]\n{chunk}")

    return "\n\n".join(output)
