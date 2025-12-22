"""
Cleanly nukes only a single run from a given output json.
"""

import argparse
import json
import os

import yaml
from loguru import logger

from matharena.grader import extract_and_grade
from matharena.runner import Runner
from matharena.runs import Runs
from matharena.utils import is_conversation_broken, normalize_conversation

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
parser.add_argument("--n", type=int, default=4)
parser.add_argument("--idx", type=int)  # ZERO-BASED

parser.add_argument("--comp-configs-dir", type=str, default="configs/competitions")
parser.add_argument("--model-configs-dir", type=str, default="configs/models")
parser.add_argument("--output-dir", type=str, default="outputs")
args = parser.parse_args()

toks = args.path.split("/")
comp = f"{toks[-5]}/{toks[-4]}"
api = toks[-3]
solver = toks[-2]
print(f"Processing {comp} {api} {solver}")

runner = Runner(comp, args.n, None, args.comp_configs_dir, None, args.output_dir, False)

solver_config_path = f"{args.model_configs_dir}/{api}/{solver}.yaml"
with open(solver_config_path, "r") as f:
    solver_config = yaml.safe_load(f)
solver_type = solver_config.get("type", "pure_model")
output_dir = f"{args.output_dir}/{comp}/{api}/{solver}"

problem_idx = toks[-1].split(".")[0]
for problem in runner.problems:
    if str(problem["problem_idx"]) == problem_idx:
        runs = Runs(comp, runner.is_fa_comp, solver, solver_type, problem, output_dir)
        runs.load_from_file()
        print(f"Found problem {problem_idx} with {runs.N} runs; dropping run {args.idx} (0-based)")
        runs.drop_runs([args.idx])
        logger.info("Saving back to file")
        runs.save_to_file()
        exit(0)
