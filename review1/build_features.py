import json
import numpy as np
from sentence_transformers import SentenceTransformer

with open("review1/benchmark_slice.json") as f:
    bench = {r["id"]: r for r in json.load(f)}

small_correct = {}
large_correct = {}
with open("review1/run_results.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if "error" in r:
            continue
        if r["tier"] == "small":
            small_correct[r["id"]] = r["correct"]
        else:
            large_correct[r["id"]] = r["correct"]

ids = [i for i in bench if i in small_correct and i in large_correct]
questions = [bench[i]["question"] for i in ids]
sources = [bench[i]["source"] for i in ids]
y_small = np.array([int(small_correct[i]) for i in ids])
y_large = np.array([int(large_correct[i]) for i in ids])

print(f"usable rows: {len(ids)}")

model = SentenceTransformer("all-MiniLM-L6-v2")
X = model.encode(questions, show_progress_bar=True)

np.savez("review1/router_features.npz", X=X, y_small=y_small, y_large=y_large,
         ids=np.array(ids), sources=np.array(sources))
print("saved review1/router_features.npz, X shape:", X.shape)
