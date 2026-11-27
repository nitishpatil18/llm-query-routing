import json, random, re
from collections import Counter
from datasets import load_dataset

SEED = 42
N_MMLUPRO = 200
N_MATH = 150
rng = random.Random(SEED)

with open("review1/benchmark_slice.json") as f:
    old = json.load(f)

mp = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
new_mp = []
for i in rng.sample(range(len(mp)), N_MMLUPRO):
    r = mp[i]
    new_mp.append({
        "id": f"mmlupro_{r['question_id']}", "source": "mmlupro", "subject": r["category"],
        "question": r["question"], "choices": list(r["options"]), "answer": r["answer"],
    })

math = load_dataset("HuggingFaceH4/MATH-500", split="test")
numeric = [i for i in range(len(math)) if re.fullmatch(r"-?\d+(\.\d+)?", math[i]["answer"].strip())]
print("MATH-500 rows with plain numeric answers:", len(numeric), "of", len(math))
new_math = []
for i in rng.sample(numeric, N_MATH):
    r = math[i]
    new_math.append({
        "id": "math_" + r["unique_id"].replace("/", "_"), "source": "math", "subject": r["subject"],
        "level": r["level"], "question": r["problem"], "choices": None, "answer": r["answer"].strip(),
    })

rows = old + new_mp + new_math
assert len(set(r["id"] for r in rows)) == len(rows), "duplicate ids"
with open("benchmark_v2.json", "w") as f:
    json.dump(rows, f)

print("total rows:", len(rows))
print("by source:", dict(Counter(r["source"] for r in rows)))
print("math levels:", dict(sorted(Counter(r["level"] for r in new_math).items())))
print("mmlupro options per question:", dict(sorted(Counter(len(r["choices"]) for r in new_mp).items())))
print("sample math:", new_math[0]["question"][:100], "->", new_math[0]["answer"])
