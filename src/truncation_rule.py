import numpy as np

SEED = 42
d = np.load("data/router_features_v2.npz", allow_pickle=True)
correct, cost, ctok = d["correct"], d["cost"], d["ctok"]
tiers = [str(t) for t in d["tiers"]]
S, L = tiers.index("small"), tiers.index("large")
n = len(correct)
cs, cl = correct[:, S].astype(float), correct[:, L].astype(float)
ps0, pl0 = cost[:, S], cost[:, L]
cts = ctok[:, S]
trunc = cts >= 4096
print(f"small completion tokens: max={cts.max()} n>=4096: {(cts >= 4096).sum()} n>=4000: {(cts >= 4000).sum()}")
print(f"truncated n={trunc.sum()} ({trunc.mean():.1%}): small acc there={cs[trunc].mean():.2f} large acc there={cl[trunc].mean():.2f}")
print(f"not truncated: small acc={cs[~trunc].mean():.3f} large acc={cl[~trunc].mean():.3f}")


def mix_cost(a, aS, aL, cS, cL):
    f = np.clip((a - aS) / max(aL - aS, 1e-9), 0, 1)
    return cS + f * (cL - cS)


acc_i = np.where(trunc, cl, cs)
for m in [1, 4, 16]:
    ps, pl = ps0, pl0 * m
    cost_i = ps + np.where(trunc, pl, 0.0)
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(2000):
        b = rng.integers(0, n, n)
        aS, aL = cs[b].mean(), cl[b].mean()
        if aL - aS < 1e-9:
            continue
        mc = mix_cost(acc_i[b].mean(), aS, aL, ps[b].sum(), pl[b].sum())
        diffs.append((mc - cost_i[b].sum()) / pl[b].sum() * 100)
    lo, mid, hi = np.percentile(diffs, [2.5, 50, 97.5])
    print(f"large price x{m:<2} (ratio {pl.sum() / ps.sum():4.1f}x): acc={acc_i.mean():.3f} saved_vs_large={(1 - cost_i.sum() / pl.sum()) * 100:5.1f}% | vs mix {mid:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]")
