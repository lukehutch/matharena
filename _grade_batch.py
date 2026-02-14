import json
import sys

from matharena.grader import extract_and_grade


def main() -> None:
    raw = sys.stdin.read()
    if not raw.strip():
        print("[]")
        return

    items = json.loads(raw)
    results = []

    for item in items:
        try:
            answer, correct, warning = extract_and_grade(
                messages=[
                    {"role": "user", "content": ""},
                    {"role": "assistant", "type": "response", "content": item["response"]},
                ],
                output_tokens=0,
                gold_answer=item["gold"],
                competition_config={"answer_type": "integer"},
            )
            results.append(
                {
                    "extracted": str(answer) if answer is not None else None,
                    "correct": bool(correct),
                    "warning": str(warning),
                }
            )
        except Exception as err:  # noqa: BLE001
            results.append({"extracted": None, "correct": False, "warning": str(err)})

    print(json.dumps(results))


if __name__ == "__main__":
    main()
