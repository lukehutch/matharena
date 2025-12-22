This readme explains how to run Project Euler problems with MathArena.

## Adding a new problem

When there is a new problem, you can add its problem statement automatically by running a command like this (replace id with the actual id of the new problem):

```bash
uv run python scripts/euler/add_euler.py --problem_id 954
```

This will:
- Add the problem statement to the `data/euler/euler/problems` directory.
- Update the number of problems in the config file `configs/competitions/euler/euler.yaml` and set the date as today, which you should edit or confirm.
- Update the `source.csv` file which maps the ids shown in the table to the actual problem id in the format `eulerXXX` (e.g. `euler955` for problem `955`). 
- Set the answer to "none" in the `answers.csv` file.
- Update `website/flaskr/static/data/competitions.json` with info such as difficulty (set as todo, update when known).

Note: These problem statements have HTML in them such as `<p>...</p>`. You can manually remove these from the problem statements.

## Running the models

For most of the models, code execution is done remotely using Modal. You can create an account there and follow the quickstart setup (https://modal.com/docs/guide).
We have a backup option that uses Docker locally. To enables this, first run:

```bash
docker build -t matharena-docker docker/
```

After the setup, you can test whether everything works by running the script with a few basic test cases:

```bash
uv run pytest tests/test_code_execution.py
```
Running the models is done with the same command as for the non-coding competitions, e.g.

```bash
uv run python scripts/run.py --models gemini/gemini-pro-2.5 --comp euler/euler
```

## Submitting the results

Once all the models have run, the last part is submitting the results to the Project Euler website to check correctness. You can do this manually or by running the following command:

```bash
uv run playwright install # if not installed yet, otherwise this step can be skipped
uv run python scripts/euler/project_euler_submit.py --problem_id 12
```

The script will prefill the answers to submit, but you will likely be asked to solve a captcha on the login page and when submitting each result (NOTE: you have to enter the captcha in the **terminal**, not the **browser**).
The script has constants `MIN_DELAY` and `MAX_DELAY` that control the delay between submissions. You can change them to your liking.

The script will stop once it has submitted all the answers or once a correct answer has been submitted.

Also, the script will store the submission logs in the `logs/euler` directory (HTML of the submission page and screenshot of the submission result), for inspection and so that it can skip submitting already submitted answers if you re-run the script.

## Re-running with correct answers

If all answers are incorrect, we are finished. 

Otherwise, we have to manually modify the `answers.csv` with the correct answer.

Then use `scripts/regrade.py` to repeat the grading for all runs.