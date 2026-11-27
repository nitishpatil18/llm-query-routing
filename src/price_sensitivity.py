import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

SEED = 42
d = np.load("data/router_features_v2.npz", allow_pickle=True)
X, correct, cost, ctok = d["X"], d["correct"], d["cost"], d["ctok"]
tiers = [str(t) for t in d["tiers"]]
S, L = tiers.index("small"), tiers.index("large")
n = len(X)
cs, cl = correct[:, S].astype(float), correct[:, L].astype(float)
ps0, pl0 = cost[:, S], cost[:, L]
log_len = np.log1p(ctok[:, S]).reshape(-1, 1)
variants = [
    ("pre-call router: embedding", X, "route"),
    ("cascade: token count", log_len, "cascade"),
    ("cascade: embedding + token count", np.hstack([X, log_len]), "cascade"),
]
taus = np.linspace(0, 1.0001, 400)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

P = {}
for name, feats, mode in variants:
    clf = XGBClassifier(n_estimators=100, max_depth=3, eval_metric="logloss", random_state=SEED)
    P[name] = cross_val_predict(clf, feats, cs, cv=skf, method="predict_proba")[:, 1]


def evaluate(p, idx, tau, mode, ps, pl):
    esc = p[idx] < tau
    a = np.where(esc, cl[idx], cs[idx])
    c = np.where(esc, pl[idx], ps[idx]) if mode == "route" else ps[idx] + np.where(esc, pl[idx], 0.0)
    return a, c


def pick_tau(p, tr, mode, frac, ps, pl):
    target = frac * cl[tr].mean()
    best = None
    for tau in taus:
        a, c = evaluate(p, tr, tau, mode, ps, pl)
        if a.mean() >= target - 1e-12 and (best is None or c.sum() < best[0]):
            best = (c.sum(), tau)
    return best[1] if best else taus[-1]


def mix_cost(a, aS, aL, cS, cL):
    f = np.clip((a - aS) / max(aL - aS, 1e-9), 0, 1)
    return cS + f * (cL - cS)


for m in [1, 4, 16]:
    ps, pl = ps0, pl0 * m
    print(f"\n=== large price x{m} (hypothetical): large/small cost ratio = {pl.sum() / ps.sum():.1f}x ===")
    for name, feats, mode in variants:
        p = P[name]
        for frac in [0.98, 0.95]:
            outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
            pa, pc = np.zeros(n), np.zeros(n)
            for tr, te in outer.split(X, cs):
                tau = pick_tau(p, tr, mode, frac, ps, pl)
                a, c = evaluate(p, te, tau, mode, ps, pl)
                pa[te], pc[te] = a, c
            rng = np.random.default_rng(SEED)
            diffs = []
            for _ in range(1000):
                b = rng.integers(0, n, n)
                aS, aL = cs[b].mean(), cl[b].mean()
                if aL - aS < 1e-9:
                    continue
                mc = mix_cost(pa[b].mean(), aS, aL, ps[b].sum(), pl[b].sum())
                diffs.append((mc - pc[b].sum()) / pl[b].sum() * 100)
            lo, mid, hi = np.percentile(diffs, [2.5, 50, 97.5])
            print(f"  {name:<34} {frac:.0%}: acc={pa.mean():.3f} saved_vs_large={(1 - pc.sum() / pl.sum()) * 100:5.1f}% | vs mix {mid:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]")
