#!/usr/bin/env python3
import argparse
import os
from datetime import datetime

from matharena.api_client import APIClient
from matharena.arxivbench_utils import (
    extract_json,
    load_annotation,
    load_model_config,
    load_prompt_template,
    list_paper_ids,
    resolve_model_config_path,
    save_annotation,
)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))




def needs_verification(annotation, overwrite=False):
    review = annotation.get("review") or {}
    
    if annotation.get("keep") is not True:
        return False
    if not annotation.get("question") or not annotation.get("answer"):
        return False
    if "review" not in annotation:
        return True
    if review.get("status") != "keep":
        return False
    
    verification = annotation.get("verification")
    if overwrite:
        return True
    if isinstance(verification, dict) and "keep" in verification:
        return False
    return True


def render_prompt(template, question, answer):
    return template.format(
        question=question or "",
        answer=answer or "",
    )


def coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes"}:
            return True
        if lowered in {"false", "no"}:
            return False
    return None


def main():
    parser = argparse.ArgumentParser(description="Verify kept LLM annotations against the criteria.")
    parser.add_argument("--model-config", required=True, help="Path under ../configs/models (e.g. openai/gpt-5-mini).")
    parser.add_argument("--paper-root", default="arxivbench/paper", help="Root directory containing paper folders.")
    parser.add_argument("--prompt", default="arxivbench/prompts/prompt_verify.md", help="Prompt template path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of papers to verify.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing verification results.")
    args = parser.parse_args()

    prompt_template = load_prompt_template(args.prompt)
    model_config_path = resolve_model_config_path(args.model_config)
    model_config = load_model_config(model_config_path)
    model_name = model_config["model"]
    client = APIClient(**model_config)

    paper_ids = list_paper_ids(args.paper_root)
    queries = []
    query_paper_ids = []
    for paper_id in paper_ids:
        annotation = load_annotation(args.paper_root, paper_id)
        if not needs_verification(annotation, overwrite=args.overwrite):
            continue
        prompt = render_prompt(
            prompt_template,
            annotation.get("question"),
            annotation.get("answer"),
        )
        queries.append([{"role": "user", "content": prompt}])
        query_paper_ids.append(paper_id)
        if args.limit and len(queries) >= args.limit:
            break

    if not queries:
        print("No kept papers need verification.")
        return

    total_cost = 0.0
    kept_ids = []
    rejected_ids = []
    for idx, conversation, cost in client.run_queries(queries):
        paper_id = query_paper_ids[idx]
        response = ""
        if conversation and isinstance(conversation[-1], dict):
            response = conversation[-1].get("content", "") or ""
        parsed = extract_json(response)
        annotation = load_annotation(args.paper_root, paper_id)
        verification = {
            "model": model_name,
            "raw": response,
            "cost": cost.get("cost", 0.0),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        keep_value = None
        if isinstance(parsed, dict):
            keep_value = coerce_bool(parsed.get("keep"))
            verification["parsed"] = parsed
        if keep_value is not None:
            if "keep_original" not in annotation:
                annotation["keep_original"] = annotation.get("keep")
            annotation["keep"] = keep_value
            verification["keep"] = keep_value
            if keep_value:
                kept_ids.append(paper_id)
            else:
                rejected_ids.append(paper_id)
        annotation["verification"] = verification
        save_annotation(args.paper_root, paper_id, annotation)
        total_cost += verification["cost"]

    print(f"Completed {len(queries)} verification queries. Total cost: ${total_cost:.6f}")
    print(f"Verified keep: {len(kept_ids)} papers: {', '.join(kept_ids)}")
    print(f"Verified reject: {len(rejected_ids)} papers: {', '.join(rejected_ids)}")


if __name__ == "__main__":
    main()
