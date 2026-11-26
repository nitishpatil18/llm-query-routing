import json, os, re, time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

with open("benchmark_slice.json") as f:
    rows = json.load(f)

MODELS = {"small": "openai/gpt-oss-20b", "large": "openai/gpt-oss-120b", "premium": "qwen/qwen3.8-27b"}
OUT = "run_results.jsonl"

def build_prompt(row):
    if row["source"] == "mmlu":
        c = row["choices"]
        return (f"Question: {row['question']}\nA. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\n\n"
                "Answer with just the letter (A, B, C, or D).")
    return (f"{row['question']}\n\nSolve step by step, then on the final line write exactly:\n"
            "Answer: <number>")

def parse_mmlu(text):
    t = text.strip().upper()
    if t[:1] in "ABCD" and (len(t) == 1 or not t[1].isalnum()):
        return t[0]
    m = re.search(r"\b([ABCD])\b", t)
    return m.group(1) if m else None

def parse_gsm8k(text):
    m = re.search(r"Answer:\s*\$?(-?[\d,]+\.?\d*)", text)
    val = m.group(1) if m else (re.findall(r"-?[\d,]+\.?\d*", text) or [None])[-1]
    return val.replace(",", "").rstrip(".") if val else None

def is_correct(row, parsed):
    if parsed is None:
        return False
    if row["source"] == "mmlu":
        return parsed == row["answer"]
    try:
        return abs(float(parsed) - float(row["answer"])) < 1e-4
    except ValueError:
        return False

done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        for line in f:
            r = json.loads(line)
            done.add((r["id"], r["tier"]))

total = len(rows) * len(MODELS)
count = len(done)

with open(OUT, "a") as out_f:
    for row in rows:
        prompt = build_prompt(row)
        for tier, model in MODELS.items():
            if (row["id"], tier) in done:
                continue
            record = None
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=model, messages=[{"role": "user", "content": prompt}], temperature=0,
                    )
                    text = resp.choices[0].message.content
                    usage = resp.usage
                    parsed = parse_mmlu(text) if row["source"] == "mmlu" else parse_gsm8k(text)
                    record = {
                        "id": row["id"], "source": row["source"], "tier": tier, "model": model,
                        "parsed_answer": parsed, "true_answer": row["answer"],
                        "correct": is_correct(row, parsed),
                        "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                    }
                    break
                except Exception as e:
                    if attempt == 2:
                        record = {"id": row["id"], "source": row["source"], "tier": tier, "model": model, "error": str(e)}
                    else:
                        time.sleep(5 * (attempt + 1))
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            count += 1
            if count % 20 == 0:
                print(f"progress: {count}/{total}")
            time.sleep(2.2)

# summary
from collections import defaultdict
stats = defaultdict(lambda: {"correct": 0, "total": 0, "ptok": 0, "ctok": 0, "errors": 0})
with open(OUT) as f:
    for line in f:
        r = json.loads(line)
        k = (r["tier"], r["source"])
        if "error" in r:
            stats[k]["errors"] += 1
            continue
        stats[k]["total"] += 1
        stats[k]["correct"] += int(r["correct"])
        stats[k]["ptok"] += r["prompt_tokens"]
        stats[k]["ctok"] += r["completion_tokens"]

print("\n--- summary ---")
for (tier, source), s in sorted(stats.items()):
    acc = s["correct"] / s["total"] if s["total"] else 0
    print(f"{tier:6s} {source:6s}: acc={acc:.3f} ({s['correct']}/{s['total']}) ptok={s['ptok']} ctok={s['ctok']} errors={s['errors']}")
