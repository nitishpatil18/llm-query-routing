import json
import random
from datasets import load_dataset

random.seed(42)

mmlu = load_dataset("cais/mmlu", "all", split="test")
gsm8k = load_dataset("openai/gsm8k", "main", split="test")

mmlu_idx = random.sample(range(len(mmlu)), 80)
gsm8k_idx = random.sample(range(len(gsm8k)), 80)

rows = []
letters = ["A", "B", "C", "D"]

for i in mmlu_idx:
    ex = mmlu[i]
    rows.append({
        "id": f"mmlu_{i}",
        "source": "mmlu",
        "subject": ex["subject"],
        "question": ex["question"],
        "choices": ex["choices"],
        "answer": letters[ex["answer"]],
    })

for i in gsm8k_idx:
    ex = gsm8k[i]
    final_answer = ex["answer"].split("####")[-1].strip().replace(",", "")
    rows.append({
        "id": f"gsm8k_{i}",
        "source": "gsm8k",
        "subject": "math",
        "question": ex["question"],
        "choices": None,
        "answer": final_answer,
    })

with open("benchmark_slice.json", "w") as f:
    json.dump(rows, f, indent=2)

print(f"total rows: {len(rows)}")
print("sample mmlu:", rows[0])
print("sample gsm8k:", rows[80])
