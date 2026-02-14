
import json
from datasets import load_dataset
ds = load_dataset('MathArena/aime_2025', split='train')
problems = []
for item in ds:
    problems.append({
        'idx': item['problem_idx'],
        'problem': item['problem'],
        'gold_answer': str(item['answer']),
        'type': item.get('problem_type', 'Unknown'),
    })
print(json.dumps(problems))
