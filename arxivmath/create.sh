uv run python arxivbench/create_queries.py --model-config openai/gpt-52-medium
uv run python arxivbench/verify_queries.py --model-config openai/gpt-52-medium
uv run python arxivbench/fulltext_review.py --model-config openai/gpt-52-medium
uv run python arxivbench/fulltext_review.py --model-config openai/gpt-52-medium --key prior_work_filter --prompt arxivbench/prompts/prompt_prior_work_filter.md
uv run python arxivbench/fulltext_review.py --model-config openai/gpt-52-medium --key solid_authors --prompt arxivbench/prompts/prompt_solid_authors.md --enable-web-search