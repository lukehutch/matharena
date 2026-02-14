"""CLI solver that invokes an external command (e.g. Derive) via subprocess."""

import json
import subprocess
import time
from typing import Any, override

from matharena.solvers import BaseSolver, SolverResponse


class CLISolver(BaseSolver):
    """
    A solver that invokes an external CLI tool to solve math problems.
    The CLI receives the problem on stdin and outputs a JSON response on stdout.
    """

    def __init__(self, solver_config, default_prompt_template, default_api_client_args, last_chance_prompt):
        super().__init__(solver_config, default_prompt_template, default_api_client_args, last_chance_prompt)
        model_config = solver_config["model_config"]
        self.cli_command = model_config["cli_command"]
        self.cli_args = model_config.get("cli_args", [])
        self.timeout = model_config.get("timeout", 600)

    @override
    def solve_batch(
        self,
        stmt_batch: list[tuple[str, Any]],
        batch_idx_to_problem_idx: dict[int, int],
        batch_idx_to_run_idx: dict[int, int],
    ):
        for batch_idx, (text, image_b64) in enumerate(stmt_batch):
            if text is None:
                text = "See image."
            prompt = self.default_prompt_template.format(problem=text)

            problem_idx = batch_idx_to_problem_idx[batch_idx]
            run_idx = batch_idx_to_run_idx[batch_idx]
            print(f"[cli-solver] Problem {problem_idx + 1}, run {run_idx + 1}: invoking {self.cli_command}")

            start_time = time.time()
            try:
                result = subprocess.run(
                    [self.cli_command] + self.cli_args,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                elapsed = time.time() - start_time
                stdout = result.stdout.strip()
                stderr = result.stderr.strip()

                if stderr:
                    # Print stderr lines (agent progress) to console
                    for line in stderr.split('\n'):
                        if line.strip():
                            print(f"  {line.strip()}")

                # Parse JSON output from the CLI
                try:
                    output = json.loads(stdout)
                    response_text = output.get("response", stdout)
                    cost = output.get("cost", 0)
                except (json.JSONDecodeError, TypeError):
                    response_text = stdout if stdout else stderr
                    cost = 0

                print(f"[cli-solver] Problem {problem_idx + 1}: done in {elapsed:.1f}s ({len(response_text)} chars)")

            except subprocess.TimeoutExpired:
                elapsed = time.time() - start_time
                response_text = f"CLI solver timed out after {self.timeout}s"
                cost = 0
                print(f"[cli-solver] Problem {problem_idx + 1}: TIMEOUT after {elapsed:.1f}s")
            except Exception as e:
                elapsed = time.time() - start_time
                response_text = f"CLI solver error: {e}"
                cost = 0
                print(f"[cli-solver] Problem {problem_idx + 1}: ERROR: {e}")

            conversation = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response_text},
            ]
            detailed_cost = {
                "cost": cost,
                "input_tokens": 0,
                "output_tokens": 0,
                "time": elapsed,
            }
            yield SolverResponse(batch_idx, conversation, detailed_cost, history=[])

    @override
    def last_chance(self, previous_response: SolverResponse) -> SolverResponse:
        """No reprompting for CLI solver — return as-is."""
        return previous_response
