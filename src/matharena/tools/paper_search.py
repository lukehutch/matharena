import requests
import os
import time
import json
from ..api_client import APIClient
import yaml
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
import threading
from loguru import logger

STORE_FOLDER = "paper_search_cache/"
MODEL_CONFIG_PATH = "configs/models/deepseek/ocr.yaml"
os.makedirs(STORE_FOLDER, exist_ok=True)
_s2_rate_lock = threading.Lock()
_s2_last_call = 0.0  # monotonic seconds
S2_API_KEY = os.getenv("S2_API_KEY")
if not S2_API_KEY:
    logger.warning("Semantic Scholar API key (S2_API_KEY) not set. You may hit rate limits.")

def _rate_limit_one_call_per_10s():
    global _s2_last_call
    with _s2_rate_lock:
        now = time.monotonic()
        wait = 2.0 - (now - _s2_last_call)
        if wait > 0:
            time.sleep(wait)
        # Reserve the slot for this thread (IMPORTANT: do this while holding the lock)
        _s2_last_call = time.monotonic()

def query_semantic_scholar(query, result_limit=15):
    """Extract paper data from Semantic Scholar API given a query string."""
    n_retries = 10
    for attempt in range(1, n_retries + 1):
        try:
            # Enforce global rate limit across threads
            _rate_limit_one_call_per_10s()
            rsp = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                headers={"X-API-KEY": os.getenv("S2_API_KEY")} if os.getenv("S2_API_KEY") else {},
                params={
                    "query": query,
                    "limit": result_limit,
                    "fields": "title,url,abstract,publicationVenue,textAvailability,publicationDate,openAccessPdf",
                    "publicationDateOrYear": ":2025-08-31",
                },
            )
            rsp.raise_for_status()
            results = rsp.json()
            if "total" in results:
                break
        except Exception as e:
            # backoff a bit (don't hammer the API). keep it short; rate limiter already spaces calls.
            logger.warning(f"Attempt {attempt} failed to query Semantic Scholar: {str(e)}")
            time.sleep(min(2 ** attempt, 60))
    else:
        logger.error(f"Failed to query Semantic Scholar after {n_retries} attempts.")
        return "No results found."

    if results.get("total", 0) == 0:
        return "No results found."
    paper_string = ""
    for paper in results.get("data", []):
        url = None
        oap = paper.get("openAccessPdf") or {}
        url = oap.get("url")

        paper["url"] = url
        paper_id = (paper.get("paperId") or "unknown")[:10]
        open_paper = url is not None and url != ""

        paper_string += f"### {paper.get('title','(no title)')} (Paper ID: {paper_id}) ###\n"
        paper_string += f"PDF Available: {'YES' if open_paper else 'NO'}\n"
        if paper.get("abstract"):
            paper_string += f"Abstract:\n{paper['abstract']}\n\n"
        else:
            paper_string += "Abstract:\nNot available\n\n"

        json_path = os.path.join(STORE_FOLDER, f"{paper_id}.json")
        with open(json_path, "w") as f:
            json.dump(paper, f)

    return paper_string

def download_paper_pdf(paper_id):
    """Downloads the PDF of a paper given its Paper ID from the cache folder."""
    json_path = os.path.join(STORE_FOLDER, f"{paper_id}.json")
    if not os.path.exists(json_path):
        return None, "Paper ID not found in cache."

    with open(json_path, 'r') as f:
        paper = json.load(f)

    if paper["url"] is None:
        return None, "PDF not available for this paper."

    try:
        rsp = requests.get(paper['url'])
        rsp.raise_for_status()
        # check if content-type is pdf
        if 'application/pdf' not in rsp.headers.get('Content-Type', ''):
            return None, "The URL does not point to a valid PDF. PDF is therefore not available for this paper."
        pdf_path = os.path.join(STORE_FOLDER, f"{paper_id}.pdf")
        with open(pdf_path, 'wb') as f:
            f.write(rsp.content)
        return pdf_path, "PDF downloaded successfully."
    except Exception as e:
        return None, f"Failed to download PDF: {str(e)}. Do not attempt again, since the URL is likely invalid, making the PDF unavailable for this paper."


def pdf_page_to_pil(doc: fitz.Document, page_index: int, dpi: int = 300) -> Image.Image:
    page = doc.load_page(page_index)
    # Render at requested DPI
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return img


def pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    mime = "image/png"

    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def ocr_paper(paper_id):
    model_config = yaml.safe_load(open(MODEL_CONFIG_PATH, 'r'))
    del model_config["human_readable_id"]
    del model_config["date"]
    client = APIClient(**model_config)
    pdf_path = os.path.join(STORE_FOLDER, f"{paper_id}.pdf")

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    md_pages = []
    prompt = "Convert the following image to markdown."

    pages_queries = []
    
    for i in range(total_pages):
        img = pdf_page_to_pil(doc, i)
        data_uri = pil_to_data_uri(img)
        pages_queries.append([
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ])

    md_pages = [None] * total_pages
    for idx, result, _ in client.run_queries(pages_queries, no_tqdm=True):
        markdown = result[-1]["content"].strip()
        md_pages[idx] = f"##### Page {idx + 1} #####\n\n{markdown}"

    all_md = "\n\n".join(md_pages)
    with open(os.path.join(STORE_FOLDER, f"{paper_id}.md"), 'w') as f:
        f.write(all_md)
    
def check_and_prepare_paper(paper_id):
    json_path = os.path.join(STORE_FOLDER, f"{paper_id}.json")
    if not os.path.exists(json_path):
        return "Paper ID not found."
    md_path = os.path.join(STORE_FOLDER, f"{paper_id}.md")
    if not os.path.exists(md_path):
        done, message = download_paper_pdf(paper_id)
        if not done:
            return message
        ocr_paper(paper_id)
    return "Paper is ready."

def read_paper(paper_id):
    output_prepare = check_and_prepare_paper(paper_id)
    if output_prepare != "Paper is ready.":
        return output_prepare
    md_path = os.path.join(STORE_FOLDER, f"{paper_id}.md")
    
    with open(md_path, 'r') as f:
        content = f.read()
    return content

def read_pages(paper_id, min_page=None, max_page=None):
    output_prepare = check_and_prepare_paper(paper_id)
    if output_prepare != "Paper is ready.":
        return output_prepare
    md_path = os.path.join(STORE_FOLDER, f"{paper_id}.md")
    
    with open(md_path, 'r') as f:
        content = f.read()
    
    pages = content.split("##### Page ")
    selected_pages = []
    for page in pages[1:]:
        page_number_str, page_content = page.split(" #####\n\n", 1)
        page_number = int(page_number_str.strip())
        if (min_page is None or page_number >= min_page) and (max_page is None or page_number <= max_page):
            selected_pages.append(f"##### Page {page_number} #####\n\n{page_content}")
    
    return "\n\n".join(selected_pages)

def find_in_paper(paper_id, string):
    # returns the first 10 occurrences of string in the paper markdown
    # plus five sentences before and after it, as well as the page number
    output_prepare = check_and_prepare_paper(paper_id)
    if output_prepare != "Paper is ready.":
        return output_prepare
    md_path = os.path.join(STORE_FOLDER, f"{paper_id}.md")
    with open(md_path, 'r') as f:
        content = f.read()
    pages = content.split("##### Page ")
    occurrences = []
    for page in pages[1:]:
        page_number_str, page_content = page.split(" #####\n\n", 1)
        page_number = int(page_number_str.strip())
        sentences = page_content.split('. ')
        for i, sentence in enumerate(sentences):
            if str(string).lower() in sentence.lower():
                start = max(0, i - 3)
                end = min(len(sentences), i + 3)
                context = '. '.join(sentences[start:end])
                occurrences.append(f"### Occurrence on Page {page_number} ###\n...\n{context}\n...")
                if len(occurrences) >= 10:
                    break
        if len(occurrences) >= 10:
            break
    if not occurrences:
        return "No occurrences found."
    return "\n\n".join(occurrences)
