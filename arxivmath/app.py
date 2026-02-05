#!/usr/bin/env python3
import argparse
import json
import os
import uuid
from datetime import datetime

from flask import Flask, abort, redirect, render_template, request, session, url_for


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PAPER_ROOT = os.path.join(APP_ROOT, "paper")
ANNOTATION_FILENAME = "annotation.json"
LLM_ANNOTATION_FILENAME = "llm_annotation.json"
CHECK_ONLY = False
CHECK_ONLY_KEPT = False

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")
SKIPPED_BY_SESSION = {}


def is_annotated(paper_id):
    annotation = load_annotation(paper_id)
    return annotation.get("status") in {"keep", "discard"}


def list_paper_ids(
    include_annotated=False, check_only=False, check_only_kept=False, skip_ids=None
):
    if not os.path.isdir(PAPER_ROOT):
        return []
    skip_set = set(skip_ids or [])
    paper_ids = []
    for name in os.listdir(PAPER_ROOT):
        meta_path = os.path.join(PAPER_ROOT, name, "metadata.json")
        if os.path.isfile(meta_path):
            if name in skip_set:
                continue
            annotation = load_annotation(name, check_only=check_only)
            if check_only:
                if annotation.get("keep") is True:
                    review = annotation.get("review")
                    review_status = review.get("status") if isinstance(review, dict) else None
                    if check_only_kept:
                        if review_status == "keep" or not review_status:
                            paper_ids.append(name)
                    else:
                        if review_status not in {"keep", "discard"}:
                            paper_ids.append(name)
            else:
                if include_annotated or annotation.get("status") not in {"keep", "discard"}:
                    paper_ids.append(name)
    return sorted(paper_ids)


def load_metadata(paper_id):
    meta_path = os.path.join(PAPER_ROOT, paper_id, "metadata.json")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_annotation(paper_id, check_only=False):
    filename = LLM_ANNOTATION_FILENAME if check_only else ANNOTATION_FILENAME
    path = os.path.join(PAPER_ROOT, paper_id, filename)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_annotation(paper_id, data, check_only=False):
    filename = LLM_ANNOTATION_FILENAME if check_only else ANNOTATION_FILENAME
    path = os.path.join(PAPER_ROOT, paper_id, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def get_session_id():
    session_id = session.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        session_id = uuid.uuid4().hex
        session["session_id"] = session_id
    return session_id


def add_session_skip(paper_id):
    session_id = get_session_id()
    skipped = SKIPPED_BY_SESSION.setdefault(session_id, set())
    skipped.add(paper_id)


def get_session_skips():
    session_id = get_session_id()
    skipped = SKIPPED_BY_SESSION.get(session_id, set())
    if isinstance(skipped, set):
        return list(skipped)
    return []


@app.route("/")
def index():
    skip_ids = get_session_skips() if not CHECK_ONLY else []
    paper_ids = list_paper_ids(
        check_only=CHECK_ONLY,
        check_only_kept=CHECK_ONLY_KEPT,
        skip_ids=skip_ids,
    )
    if not paper_ids:
        return "No papers found in ./paper.", 200
    return redirect(url_for("paper_view", paper_id=paper_ids[0]))


@app.route("/paper/<paper_id>")
def paper_view(paper_id):
    if not CHECK_ONLY and is_annotated(paper_id):
        skip_ids = get_session_skips()
        paper_ids = list_paper_ids(check_only=CHECK_ONLY, skip_ids=skip_ids)
        if not paper_ids:
            return redirect(url_for("done"))
        return redirect(url_for("paper_view", paper_id=paper_ids[0]))
    skip_ids = get_session_skips() if not CHECK_ONLY else []
    paper_ids = list_paper_ids(
        check_only=CHECK_ONLY,
        check_only_kept=CHECK_ONLY_KEPT,
        skip_ids=skip_ids,
    )
    if paper_id not in paper_ids:
        abort(404)
    metadata = load_metadata(paper_id)
    if metadata is None:
        abort(404)
    annotation = load_annotation(paper_id, check_only=CHECK_ONLY)
    if CHECK_ONLY:
        review = annotation.get("review")
        if isinstance(review, dict):
            annotation = annotation.copy()
            if "question" in review:
                annotation["question"] = review.get("question")
            if "answer" in review:
                annotation["answer"] = review.get("answer")
    index = paper_ids.index(paper_id)
    next_id = paper_ids[index + 1] if index + 1 < len(paper_ids) else None
    prev_id = paper_ids[index - 1] if index > 0 else None
    return render_template(
        "paper.html",
        paper_id=paper_id,
        metadata=metadata,
        annotation=annotation,
        next_id=next_id,
        prev_id=prev_id,
        position=index + 1,
        total=len(paper_ids),
        check_only=CHECK_ONLY,
    )


@app.route("/paper/<paper_id>/annotate", methods=["POST"])
def annotate(paper_id):
    if load_metadata(paper_id) is None:
        abort(404)
    status = request.form.get("status", "").strip().lower()
    question = (request.form.get("question") or "").strip()
    answer = (request.form.get("answer") or "").strip()
    if status not in {"keep", "discard"}:
        status = ""
    if CHECK_ONLY:
        paper_ids_before = list_paper_ids(
            check_only=True,
            check_only_kept=CHECK_ONLY_KEPT,
        )
        index = paper_ids_before.index(paper_id) if paper_id in paper_ids_before else -1
        annotation = load_annotation(paper_id, check_only=True)
        review_status = status if status in {"keep", "discard"} else "keep"
        annotation["review"] = {
            "status": review_status,
            "question": question,
            "answer": answer,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        save_annotation(paper_id, annotation, check_only=True)
        paper_ids_after = list_paper_ids(
            check_only=True,
            check_only_kept=CHECK_ONLY_KEPT,
        )
        if paper_ids_after:
            if index >= 0:
                for candidate in paper_ids_before[index + 1 :]:
                    if candidate in paper_ids_after:
                        return redirect(url_for("paper_view", paper_id=candidate))
            return redirect(url_for("paper_view", paper_id=paper_ids_after[0]))
        return redirect(url_for("done"))
    annotation = {
        "status": status,
        "question": question,
        "answer": answer,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    save_annotation(paper_id, annotation)
    if CHECK_ONLY:
        return redirect(url_for("paper_view", paper_id=paper_id))
    if status == "keep":
        return redirect(url_for("paper_view", paper_id=paper_id))
    skip_ids = get_session_skips() if not CHECK_ONLY else []
    paper_ids = list_paper_ids(check_only=CHECK_ONLY, skip_ids=skip_ids)
    if paper_ids:
        return redirect(url_for("paper_view", paper_id=paper_ids[0]))
    return redirect(url_for("done"))


@app.route("/paper/<paper_id>/skip")
def skip_paper(paper_id):
    if load_metadata(paper_id) is None:
        abort(404)
    paper_ids = list_paper_ids(check_only=CHECK_ONLY)
    if paper_id not in paper_ids:
        abort(404)
    index = paper_ids.index(paper_id)
    add_session_skip(paper_id)
    skip_ids = get_session_skips()
    remaining_ids = [pid for pid in paper_ids if pid not in skip_ids]
    if not remaining_ids:
        return redirect(url_for("done"))
    for candidate in paper_ids[index + 1 :]:
        if candidate in remaining_ids:
            return redirect(url_for("paper_view", paper_id=candidate))
    return redirect(url_for("paper_view", paper_id=remaining_ids[0]))


@app.route("/done")
def done():
    return "All papers annotated.", 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate arXiv metadata.")
    parser.add_argument("--check", action="store_true", help="Review kept papers with Q/A for manual edits.")
    parser.add_argument(
        "--check-kept",
        action="store_true",
        help="In check mode, only show papers previously marked keep in review.",
    )
    parser.add_argument("--port", type=int, default=5000, help="Port to run the web server on.")
    args = parser.parse_args()
    CHECK_ONLY = args.check or args.check_kept
    CHECK_ONLY_KEPT = args.check_kept
    app.run(debug=True, port=args.port)
