# Verification Task

You are verifying a proposed question-answer pair.

Main question: Is this question answerable or are there missing elements? 
In other words, can the question be understood and answered without additional context or definitions?

---

Answer "keep": true only if the question is self-contained and answerable without missing definitions or context. Otherwise "keep": false.

## Output Format

Respond **only** with a JSON object:

```json
{{
  "keep": boolean
}}
```

If any criterion fails, output `"keep": false`.
If all criteria pass, output `"keep": true`.

---

# Proposed Question
{question}

# Proposed Answer
{answer}
