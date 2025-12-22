import argparse

from loguru import logger

from matharena.runner import Runner

# Main args: which competition to run with which models; how many runs per problem
parser = argparse.ArgumentParser()
parser.add_argument("--comp", type=str, required=True, help="Competition config to run")
parser.add_argument(
    "--models",
    type=str,
    nargs="+",
    required=True,
    help="List of model configs to run, might have scaffolding, example: xai/grok-4",
)
parser.add_argument("--n", type=int, default=4, help="Number of runs per problem")
parser.add_argument(
    "--problems",
    type=int,
    nargs="+",
    required=False,
    help="List of 1-based problem indices to run, example: 1 2 3 (default: all problems in competition)",
)

# skip-existing is default
parser.add_argument(
    "--redo-all", action="store_true", help="Redo all (model, problem) pairs regardless of existing runs"
)

# Generally ok to keep defaults here
parser.add_argument("--comp-configs-dir", type=str, default="configs/competitions")
parser.add_argument("--model-configs-dir", type=str, default="configs/models")
parser.add_argument("--output-dir", type=str, default="outputs")
args = parser.parse_args()

logger.info(f"Initializing runner for competition {args.comp}")
runner = Runner(
    args.comp, args.n, args.problems, args.comp_configs_dir, args.model_configs_dir, args.output_dir, args.redo_all
)

# Run each model
for model in args.models:
    logger.info(f"Calling runner for model: {model}")
    runner.run(model)

exit(0)
