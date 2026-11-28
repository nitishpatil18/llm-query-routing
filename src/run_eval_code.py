import json
import os
import random
import re
import resource
import subprocess
import sys
import tempfile
import time
from collections import defaultdict

from datasets import load_dataset
from dotenv import load_dotenv
from groq import Groq

load_dotenv(".env")
MODELS = {"small": "openai/gpt-oss-20b", "large": "openai/gpt-oss-120b", "premium": "qwen/qwen3.8-27b"}
MAX_TOKENS = 4096
SLEEP = 2.2
OUT = "results/run_results_code.jsonl"
FENCE = "`" * 3

limit = None
tiers = []
for a in sys.argv[1:]:
    if a.startswith("--limit="):
        limit = int(a.split("=", 1)[1])
    else:
        tiers.append(a)
tiers = tiers or list(MODELS)
for t in tiers:
    assert t in MODELS, f"unknown tier {t}"

client = Groq(api_key=os.environ["GROQ_API_KEY"])


def retry_after(msg):
    m = re.search(r"try again in ([0-9hms.]+)", msg)
    if not m:
        return 30.0
    total = 0.0
    for val, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", m.group(1)):
        total += float(val) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total or 30.0


def call(tier, prompt):
    waits = fails = 0
    while True:
        t0 = time.time()
        try:
            r = client.chat.completions.create(
                model=MODELS[tier],
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=MAX_TOKENS,
            )
            ch = r.choices[0]
            return {"text": ch.message.content or "", "finish": ch.finish_reason,
                    "ptok": r.usage.prompt_tokens, "ctok": r.usage.completion_tokens,
                    "lat": round(time.time() - t0, 2)}
        except Exception as e:
            code = getattr(e, "status_code", None)
            msg = str(e)
            if code == 429 or "rate_limit" in msg:
                waits += 1
                if waits > 50:
                    return {"error": f"{type(e).__name__} {code} (gave up after 50 waits)"}
                w = retry_after(msg) + 15
                print(f"  rate limit on {MODELS[tier]}: waiting {w:.0f}s", flush=True)
                time.sleep(w)
            else:
                fails += 1
                if fails >= 3:
                    return {"error": f"{type(e).__name__} {code}"}
                time.sleep(5)


def extract_code(text, entry):
    blocks = re.findall(FENCE + r"(?:python|py)?[ \t]*\n(.*?)" + FENCE, text, re.S)
    for b in reversed(blocks):
        if f"def {entry}" in b:
            return b
    if blocks:
        return blocks[-1]
    if f"def {entry}" in text:
        return text
    return None


def _limits():
    for res, val in ((resource.RLIMIT_CPU, 10), (resource.RLIMIT_FSIZE, 10 * 1024 * 1024)):
        try:
            resource.setrlimit(res, (val, val))
        except (ValueError, OSError):
            pass


HARNESS = "import sys; exec(compile(open(sys.argv[1]).read(), 'prog', 'exec'), {'__name__': 'prog'})"


def run_tests(program):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            r = subprocess.run([sys.executable, "-I", "-c", HARNESS, path], cwd=td,
                               env={"PATH": "/usr/bin:/bin"}, stdin=subprocess.DEVNULL,
                               capture_output=True, timeout=10, preexec_fn=_limits)
        except subprocess.TimeoutExpired:
            return "timeout"
    return "pass" if r.returncode == 0 else "fail"


rows = [dict(r) for r in load_dataset("openai/openai_humaneval", split="test")]
random.Random(7).shuffle(rows)

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "error" not in r:
            done.add((r["id"], r["tier"]))

for tier in tiers:
    todo = [r for r in rows if (r["task_id"], tier) not in done]
    if limit:
        todo = todo[:limit]
    for i, r in enumerate(todo, 1):
        prompt = ("Complete the following Python function. Reply with the complete function "
                  "(including its signature and any imports it needs) in a single "
                  f"{FENCE}python code block and nothing else.\n\n{r['prompt']}")
        res = call(tier, prompt)
        if "error" in res:
            rec = {"id": r["task_id"], "source": "humaneval", "tier": tier, "model": MODELS[tier], "error": res["error"]}
            status = "error"
        else:
            code = None
            if res["finish"] == "length":
                status = "truncated"
            else:
                code = extract_code(res["text"], r["entry_point"])
                if code is None:
                    status = "no_code"
                else:
                    status = run_tests(r["prompt"] + "\n\n" + code + "\n\n" + r["test"] + f"\n\ncheck({r['entry_point']})\n")
            rec = {"id": r["task_id"], "source": "humaneval", "tier": tier, "model": MODELS[tier],
                   "parsed_answer": status, "true_answer": "tests pass", "correct": status == "pass",
                   "prompt_tokens": res["ptok"], "completion_tokens": res["ctok"],
                   "finish_reason": res["finish"], "latency_s": res["lat"], "code": code}
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{tier}] {i}/{len(todo)} {r['task_id']} {status}", flush=True)
        time.sleep(SLEEP)

last = {}
for line in open(OUT):
    try:
        r = json.loads(line)
    except Exception:
        continue
    last[(r["id"], r["tier"])] = r
for tier in tiers:
    rs = [r for (i, t), r in last.items() if t == tier and "error" not in r]
    if not rs:
        continue
    st = defaultdict(int)
    for r in rs:
        st[r["parsed_answer"]] += 1
    print(f"{tier}: n={len(rs)} pass={sum(r['correct'] for r in rs) / len(rs):.3f} "
          f"mean_completion_tokens={sum(r['completion_tokens'] for r in rs) / len(rs):.0f} status={dict(st)}")
