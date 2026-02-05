#!/usr/bin/env python3
import argparse
import json
import os
import re

from matharena.api_client import APIClient
from matharena.arxivbench_utils import (
    load_annotation,
    load_model_config,
    load_prompt_template,
    list_paper_ids,
    resolve_model_config_path,
    save_annotation,
    extract_json,
    load_metadata
)


def needs_annotation(annotation, overwrite=False):
    if overwrite:
        return True
    keep = annotation.get("keep")
    question = annotation.get("question")
    answer = annotation.get("answer")
    if keep is None:
        return True
    if keep is True and (not question or not answer):
        return True
    return False

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
    parser = argparse.ArgumentParser(description="Generate questions from paper abstracts using an LLM.")
    parser.add_argument("--model-config", required=True, help="Path under ../configs/models (e.g. openai/gpt-5-mini).")
    parser.add_argument("--paper-root", default="arxivbench/paper", help="Root directory containing paper folders.")
    parser.add_argument("--prompt", default="arxivbench/prompts/prompt.md", help="Prompt template path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of papers to query.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing annotations.")
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
        if not needs_annotation(annotation, overwrite=args.overwrite):
            continue
        metadata = load_metadata(args.paper_root, paper_id)
        prompt = prompt_template.format(
            title=metadata.get("title") or "",
            abstract=metadata.get("abstract") or "",
        )
        queries.append([{"role": "user", "content": prompt}])
        query_paper_ids.append(paper_id)
        if args.limit and len(queries) >= args.limit:
            break

    if not queries:
        print("No papers need annotation.")
        return

    total_cost = 0.0
    kept_ids = []
    for idx, conversation, cost in client.run_queries(queries):
        paper_id = query_paper_ids[idx]
        response = ""
        if conversation and isinstance(conversation[-1], dict):
            response = conversation[-1].get("content", "") or ""
        parsed = extract_json(response)
        annotation = {
            "model": model_name,
            "raw": response,
            "cost": cost.get("cost", 0.0),
        }
        keep_value = None
        question_value = None
        answer_value = None
        if isinstance(parsed, dict):
            keep_value = coerce_bool(parsed.get("keep"))
            if "question" in parsed:
                question_value = parsed.get("question")
            if "answer" in parsed:
                answer_value = parsed.get("answer")
            annotation["parsed"] = parsed
        if keep_value is not None:
            annotation["keep"] = keep_value
            if keep_value is True:
                kept_ids.append(paper_id)
        if question_value not in (None, ""):
            annotation["question"] = str(question_value).strip()
        if answer_value not in (None, ""):
            annotation["answer"] = str(answer_value).strip()
        save_annotation(args.paper_root, paper_id, annotation)
        total_cost += annotation["cost"]

    print(f"Completed {len(queries)} queries. Total cost: ${total_cost:.6f}")
    print(f"Kept {len(kept_ids)} papers: {', '.join(kept_ids)}")


if __name__ == "__main__":
    main()
