import json

import numpy as np
from scipy.stats import binomtest

SEED = 42
CAP = 4096
d = np.load("data/router_features_v2.npz", allow_pickle=True)
dev = set(json.load(open("data/dev_ids.json")))
ids = [str(i) for i in d["ids"]]
tiers = [str(t) for t in d["tiers"]]
S, L = tiers.index("small"), tiers.index("large")
cs_all = d["correct"][:, S].astype(float)
cl_all = d["correct"][:, L].astype(float)
ps_all, pl_all = d["cost"][:, S], d["cost"][:, L]
cts_all = d["ctok"][:, S]
src_all = np.array([str(s) for s in d["sources"]])
held = np.array([i not in dev for i in ids])


def mix_cost(a, aS, aL, cS, cL):
    f = np.clip((a - aS) / max(aL - aS, 1e-9), 0, 1)
    return cS + f * (cL - cS)


def mcnemar(a, b):
    x = int(((a == 1) & (b == 0)).sum())
    y = int(((a == 0) & (b == 1)).sum())
    p = binomtest(x, x + y, 0.5).pvalue if x + y else 1.0
    return x, y, p


def report(mask, label):
    cs, cl, ps, pl, cts, src = (a[mask] for a in (cs_all, cl_all, ps_all, pl_all, cts_all, src_all))
    n = len(cs)
    print(f"\n===== {label}: n={n} =====")
    if n < 10:
        print("too few rows")
        return
    trunc = cts >= CAP
    print("rows by source:", {s: int((src == s).sum()) for s in np.unique(src)})
    print("truncation rate by source:", {s: f"{trunc[src == s].sum()}/{(src == s).sum()}" for s in np.unique(src)})
    if trunc.any():
        print(f"truncated {trunc.sum()} ({trunc.mean():.1%}): small acc there={cs[trunc].mean():.2f} large acc there={cl[trunc].mean():.2f}")
    print(f"finished answers: small acc={cs[~trunc].mean():.3f} large acc={cl[~trunc].mean():.3f}")
    acc_i = np.where(trunc, cl, cs)
    print(f"accuracy: always-small={cs.mean():.3f} always-large={cl.mean():.3f} cascade={acc_i.mean():.3f}")
    x, y, p = mcnemar(acc_i, cs)
    print(f"McNemar cascade vs small: cascade-only right={x} small-only right={y} p={p:.4f}")
    x, y, p = mcnemar(acc_i, cl)
    print(f"McNemar cascade vs large: cascade-only right={x} large-only right={y} p={p:.4f}")
    for m in [1, 1.25, 1.5, 1.75, 2, 3, 4, 8, 16]:
        pm = pl * m
        cost_i = ps + np.where(trunc, pm, 0.0)
        rng = np.random.default_rng(SEED)
        diffs = []
        for _ in range(2000):
            b = rng.integers(0, n, n)
            aS, aL = cs[b].mean(), cl[b].mean()
            if aL - aS < 1e-9:
                continue
            mc = mix_cost(acc_i[b].mean(), aS, aL, ps[b].sum(), pm[b].sum())
            diffs.append((mc - cost_i[b].sum()) / pm[b].sum() * 100)
        lo, mid, hi = np.percentile(diffs, [2.5, 50, 97.5])
        print(f"  large price x{m:<2} (ratio {pm.sum() / ps.sum():4.1f}x): saved_vs_large={(1 - cost_i.sum() / pm.sum()) * 100:5.1f}% | vs random mix {mid:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]")


report(held, "HELD-OUT (not seen when the rule was frozen)")
report(np.ones(len(ids), dtype=bool), "ALL rows (dev + held-out)")
