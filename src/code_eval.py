import json

import numpy as np
from scipy.stats import binomtest

PRICE = {"small": (0.000000075, 0.0000003), "large": (0.00000015, 0.0000006), "premium": (0.0000008, 0.000004)}
SEED = 42

last = {}
for line in open("results/run_results_code.jsonl"):
    try:
        r = json.loads(line)
    except Exception:
        continue
    last[(r["id"], r["tier"])] = r
ok = {t: {i: r for (i, tt), r in last.items() if tt == t and "error" not in r} for t in PRICE}
ids = sorted(set.intersection(*[set(ok[t]) for t in PRICE]))
n = len(ids)
print("per-tier records:", {t: len(ok[t]) for t in PRICE})
print(f"HumanEval problems finished on all 3 tiers: {n}")
if n == 0:
    raise SystemExit


def arrs(t):
    c = np.array([ok[t][i]["correct"] for i in ids], dtype=float)
    ct = np.array([ok[t][i]["completion_tokens"] for i in ids])
    cost = np.array([ok[t][i]["prompt_tokens"] * PRICE[t][0] + ok[t][i]["completion_tokens"] * PRICE[t][1] for i in ids])
    tr = np.array([ok[t][i]["parsed_answer"] == "truncated" for i in ids])
    return c, ct, cost, tr


cs, cts, ps0, trs = arrs("small")
cl, ctl, pl0, trl = arrs("large")
cp, ctp, pp0, trp = arrs("premium")
for name, c, ct, cost, tr in (("small", cs, cts, ps0, trs), ("large", cl, ctl, pl0, trl), ("premium", cp, ctp, pp0, trp)):
    print(f"{name:8} pass={c.mean():.3f} ({int(c.sum())}/{n}) truncated={int(tr.sum())} mean_completion_tokens={ct.mean():.0f} cost=${cost.sum():.4f}")
print(f"oracle (any of the 3 passes): {np.maximum.reduce([cs, cl, cp]).mean():.3f}")


def mcnemar(a, b):
    x = int(((a == 1) & (b == 0)).sum())
    y = int(((a == 0) & (b == 1)).sum())
    return x, y, (binomtest(x, x + y, 0.5).pvalue if x + y else 1.0)


def mix_cost(a, aS, aL, cS, cL):
    f = np.clip((a - aS) / max(aL - aS, 1e-9), 0, 1)
    return cS + f * (cL - cS)


trunc = trs
acc_i = np.where(trunc, cl, cs)
print(f"\ncascade (escalate if small hits the cap): acc={acc_i.mean():.3f}, escalated {int(trunc.sum())} of {n}")
x, y, p = mcnemar(acc_i, cl)
print(f"McNemar cascade vs large: cascade-only right={x} large-only right={y} p={p:.4f}")
gap = cl.mean() - cs.mean()
for m in [1, 2, 4, 16]:
    pl = pl0 * m
    cost_i = ps0 + np.where(trunc, pl, 0.0)
    line = f"  large price x{m:<2} (ratio {pl.sum() / ps0.sum():4.1f}x): saved_vs_large={(1 - cost_i.sum() / pl.sum()) * 100:5.1f}%"
    if gap >= 0.01:
        rng = np.random.default_rng(SEED)
        diffs = []
        for _ in range(2000):
            b = rng.integers(0, n, n)
            aS, aL = cs[b].mean(), cl[b].mean()
            if aL - aS < 1e-9:
                continue
            mc = mix_cost(acc_i[b].mean(), aS, aL, ps0[b].sum(), pl[b].sum())
            diffs.append((mc - cost_i[b].sum()) / pl[b].sum() * 100)
        lo, mid, hi = np.percentile(diffs, [2.5, 50, 97.5])
        line += f" | vs random mix {mid:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]"
    print(line)
if gap < 0.01:
    print("note: small is within 1 point of large (or better), so there is no accuracy gap for routing to exploit")
