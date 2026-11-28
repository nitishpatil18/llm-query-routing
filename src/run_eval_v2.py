import json, os, re, sys, time, random
from collections import defaultdict
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODELS = {"small": "openai/gpt-oss-20b", "large": "openai/gpt-oss-120b", "premium": "qwen/qwen3.8-27b"}
OUT = "results/run_results_v2.jsonl"
TIERS = sys.argv[1:] or list(MODELS)
MAX_TOKENS = 4096
SLEEP = 2.2
LETTERS = "ABCDEFGHIJ"

with open("data/benchmark_v2.json") as f:
    rows = json.load(f)
random.Random(7).shuffle(rows)
by_id = {r["id"]: r for r in rows}


def retry_after(msg):
    m = re.search(r"try again in ([0-9hms.]+)", msg)
    if not m:
        return 60.0
    total = 0.0
    for val, unit in re.findall(r"([\d.]+)(ms|h|m|s)", m.group(1)):
        total += float(val) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total or 60.0


def build_prompt(row):
    if row["source"] in ("mmlu", "mmlupro"):
        c = row["choices"]
        opts = "\n".join(f"{LETTERS[i]}. {c[i]}" for i in range(len(c)))
        ls = list(LETTERS[:len(c)])
        listing = ", ".join(ls[:-1]) + ", or " + ls[-1]
        return f"Question: {row['question']}\n{opts}\n\nAnswer with just the letter ({listing})."
    return (f"{row['question']}\n\nSolve step by step, then on the final line write exactly:\n"
            "Answer: <number>")


def parse_letter(text, n):
    valid = LETTERS[:n]
    t = text.strip().upper()
    if t and t[0] in valid and (len(t) == 1 or not t[1].isalnum()):
        return t[0]
    m = re.search(r"ANSWER\s*(?:IS|:)?\s*\(?([" + valid + r"])\b", t)
    if m:
        return m.group(1)
    found = re.findall(r"\b([" + valid + r"])\b", t)
    return found[-1] if found else None


def parse_number(text):
    m = re.search(r"Answer:\s*\$?(-?[\d,]+\.?\d*)", text)
    val = m.group(1) if m else (re.findall(r"-?[\d,]+\.?\d*", text) or [None])[-1]
    return val.replace(",", "").rstrip(".") if val else None


def is_correct(row, parsed):
    if parsed is None:
        return False
    if row["source"] in ("mmlu", "mmlupro"):
        return parsed == row["answer"]
    try:
        return abs(float(parsed) - float(row["answer"])) < 1e-4
    except ValueError:
        return False


def call(model, prompt):
    errors = 0
    waits = 0
    while True:
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0, max_completion_tokens=MAX_TOKENS,
            )
            return resp, time.time() - t0
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate limit" in msg:
                waits += 1
                if waits > 50:
                    raise
                wait = retry_after(str(e))
                print(f"  rate limit on {model}: waiting {wait:.0f}s", flush=True)
                time.sleep(wait + 15)
                continue
            errors += 1
            if errors >= 3:
                raise
            time.sleep(5 * errors)


def load_latest():
    latest = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                r = json.loads(line)
                latest[(r["id"], r["tier"])] = r
    return latest


def summary():
    latest = load_latest()
    st = defaultdict(lambda: {"c": 0, "n": 0, "pt": 0, "ct": 0, "err": 0, "trunc": 0})
    for (rid, tier), r in latest.items():
        if rid not in by_id:
            continue
        s = st[(r["source"], tier)]
        if "error" in r:
            s["err"] += 1
            continue
        s["n"] += 1
        s["c"] += int(r["correct"])
        s["pt"] += r["prompt_tokens"]
        s["ct"] += r["completion_tokens"]
        s["trunc"] += int(r.get("finish_reason") == "length")
    print("\n--- summary ---")
    for (src, tier), s in sorted(st.items()):
        acc = s["c"] / s["n"] if s["n"] else 0
        print(f"{src:8s} {tier:8s} acc={acc:.3f} ({s['c']}/{s['n']}) ptok={s['pt']} ctok={s['ct']} "
              f"truncated={s['trunc']} errors={s['err']}")
    complete = sum(
        1 for rid in by_id
        if all((rid, t) in latest and "error" not in latest[(rid, t)] for t in MODELS)
    )
    print(f"\nquestions finished on all 3 models: {complete} of {len(rows)}")


latest = load_latest()
with open(OUT, "a") as out_f:
    for tier, model in [(t, MODELS[t]) for t in TIERS]:
        todo = [r for r in rows if (r["id"], tier) not in latest or "error" in latest[(r["id"], tier)]]
        print(f"[{tier}] {len(todo)} to run")
        for n, row in enumerate(todo, 1):
            base = {"id": row["id"], "source": row["source"], "tier": tier, "model": model}
            try:
                resp, dt = call(model, build_prompt(row))
                text = resp.choices[0].message.content or ""
                if row["source"] in ("mmlu", "mmlupro"):
                    parsed = parse_letter(text, len(row["choices"]))
                else:
                    parsed = parse_number(text)
                record = dict(base, parsed_answer=parsed, true_answer=row["answer"],
                              correct=is_correct(row, parsed),
                              prompt_tokens=resp.usage.prompt_tokens,
                              completion_tokens=resp.usage.completion_tokens,
                              finish_reason=resp.choices[0].finish_reason,
                              latency_s=round(dt, 2))
            except Exception as e:
                record = dict(base, error=str(e))
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            print(f"[{tier}] {n}/{len(todo)}", flush=True)
            time.sleep(SLEEP)

summary()
