#!/usr/bin/env python3
import argparse
from datetime import datetime

from matharena.api_client import APIClient
from matharena.arxivbench_utils import (
    ensure_ocr,
    extract_json,
    load_annotation,
    load_model_config,
    load_prompt_template,
    list_paper_ids,
    resolve_model_config_path,
    save_annotation,
    get_latest_pair,
    load_metadata
)

def should_review(annotation, overwrite=False, key="full_text_review"):
    if annotation.get("keep") is not True:
        return False
    review = annotation.get("review") or {}
    if not overwrite and key in annotation:
        return False
    if not "review" in annotation or not review:
        return True
    if review.get("status") != "keep":
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Re-check kept arXiv questions against full paper OCR.")
    parser.add_argument("--model-config", required=True, help="Path under ../configs/models (e.g. openai/gpt-5-mini).")
    parser.add_argument("--paper-root", default="arxivbench/paper", help="Root directory containing paper folders.")
    parser.add_argument("--prompt", default="arxivbench/prompts/prompt_fulltext_review.md", help="Prompt template path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of papers to process.")
    parser.add_argument("--redo-ocr", action="store_true", help="Force OCR even if cached markdown exists.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing full-text review results.")
    parser.add_argument("--key", default="full_text_review", help="Annotation key to store the review under.")
    parser.add_argument("--enable-web-search", action="store_true", help="Enable web search for additional context.")
    args = parser.parse_args()

    prompt_template = load_prompt_template(args.prompt)
    model_config_path = resolve_model_config_path(args.model_config)
    model_config = load_model_config(model_config_path)
    model_name = model_config["model"]
    if args.enable_web_search:
        model_config["tools"] = [(None, {"type": "web_search"})]
    client = APIClient(**model_config)

    discarded = []
    updated = []
    kept = []
    total_cost = 0.0

    paper_ids = list_paper_ids(args.paper_root)
    queries = []
    query_paper_ids = []
    for paper_id in paper_ids:
        annotation = load_annotation(args.paper_root, paper_id)
        if not should_review(annotation, overwrite=args.overwrite, key=args.key):
            continue
        question, answer = get_latest_pair(annotation)
        if not question or not answer:
            continue
        full_text = ensure_ocr(paper_id, redo=args.redo_ocr)
        metadata = load_metadata(args.paper_root, paper_id)
        prompt = prompt_template.format(
            question=question or "",
            answer=answer or "",
            full_text=full_text or "",
            title=metadata.get("title") or "",
            authors=", ".join([f"{author['forenames']} {author['keyname']}" for author in metadata.get("authors", [])]),
            abstract=metadata.get("abstract") or "",
        )
        queries.append([{"role": "user", "content": prompt}])
        query_paper_ids.append(paper_id)

        if args.limit and len(queries) >= args.limit:
            break
    if not queries:
        print("No papers need review.")
        return

    for idx, conversation, cost in client.run_queries(queries):
        if idx >= len(query_paper_ids):
            continue
        paper_id = query_paper_ids[idx]
        annotation = load_annotation(args.paper_root, paper_id)
        response = ""
        if conversation and isinstance(conversation[-1], dict):
            response = conversation[-1].get("content", "") or ""
        parsed = extract_json(response)
        action = None
        keep_value = None
        if isinstance(parsed, dict):
            action = parsed.get("action")
            edited_question = parsed.get("question")
            keep_value = parsed.get("keep", True)

        review_record = {
            "model": model_name,
            "raw": response,
            "cost": cost.get("cost", 0.0),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        if isinstance(parsed, dict):
            review_record["parsed"] = parsed
            if "rationale" in parsed:
                review_record["rationale"] = parsed.get("rationale")
        if action:
            review_record["action"] = action

        review = annotation.get("review") or {}
        if action == "discard" or (action is None and '"action": "discard"' in response) or keep_value is False:
            review["status"] = "discard"
            review["updated_at"] = review_record["updated_at"]
            annotation["keep"] = False
            discarded.append(paper_id)
        elif action == "edit":
            if edited_question and str(edited_question).strip():
                review["question"] = str(edited_question).strip()
                annotation["question"] = review["question"]
                review["updated_at"] = review_record["updated_at"]
                review["status"] = "keep"
                annotation["keep"] = True
                updated.append(paper_id)
            else:
                kept.append(paper_id)
        else:
            annotation["keep"] = True
            kept.append(paper_id)

        annotation["review"] = review
        annotation[args.key] = review_record
        save_annotation(args.paper_root, paper_id, annotation)
        total_cost += review_record["cost"]

    print(f"Full-text review complete. Total cost: ${total_cost:.6f}")
    print(f"Discarded ({len(discarded)}): {', '.join(discarded)}")
    print(f"Updated ({len(updated)}): {', '.join(updated)}")
    print(f"Kept ({len(kept)}): {', '.join(kept)}")


if __name__ == "__main__":
    main()
