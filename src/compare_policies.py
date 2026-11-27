import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

SEED = 42
d = np.load("data/router_features_v2.npz", allow_pickle=True)
X, correct, cost, ctok = d["X"], d["correct"], d["cost"], d["ctok"]
tiers = [str(t) for t in d["tiers"]]
S, L = tiers.index("small"), tiers.index("large")
n = len(X)
cs, cl = correct[:, S].astype(float), correct[:, L].astype(float)
ps, pl = cost[:, S], cost[:, L]
log_len = np.log1p(ctok[:, S]).reshape(-1, 1)
variants = [
    ("pre-call router: embedding", X, "route"),
    ("cascade: token count", log_len, "cascade"),
    ("cascade: embedding + token count", np.hstack([X, log_len]), "cascade"),
]
taus = np.linspace(0, 1.0001, 400)


def evaluate(p, idx, tau, mode):
    esc = p[idx] < tau
    a = np.where(esc, cl[idx], cs[idx])
    c = np.where(esc, pl[idx], ps[idx]) if mode == "route" else ps[idx] + np.where(esc, pl[idx], 0.0)
    return a, c


def pick_tau(p, tr, mode, frac):
    target = frac * cl[tr].mean()
    best = None
    for tau in taus:
        a, c = evaluate(p, tr, tau, mode)
        if a.mean() >= target - 1e-12 and (best is None or c.sum() < best[0]):
            best = (c.sum(), tau)
    return best[1] if best else taus[-1]


def mix_cost(a, aS, aL, cS, cL):
    f = np.clip((a - aS) / max(aL - aS, 1e-9), 0, 1)
    return cS + f * (cL - cS)


print(f"n={n} always-small acc={cs.mean():.3f} ${ps.sum():.4f} | always-large acc={cl.mean():.3f} ${pl.sum():.4f}")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for name, feats, mode in variants:
    clf = XGBClassifier(n_estimators=100, max_depth=3, eval_metric="logloss", random_state=SEED)
    p = cross_val_predict(clf, feats, cs, cv=skf, method="predict_proba")[:, 1]
    print(f"\n--- {name} --- AUC={roc_auc_score(cs, p):.3f}")
    for frac in [0.98, 0.95]:
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
        pa, pc = np.zeros(n), np.zeros(n)
        for tr, te in outer.split(X, cs):
            tau = pick_tau(p, tr, mode, frac)
            a, c = evaluate(p, te, tau, mode)
            pa[te], pc[te] = a, c
        rng = np.random.default_rng(SEED)
        diffs = []
        for _ in range(2000):
            b = rng.integers(0, n, n)
            aS, aL = cs[b].mean(), cl[b].mean()
            if aL - aS < 1e-9:
                continue
            m = mix_cost(pa[b].mean(), aS, aL, ps[b].sum(), pl[b].sum())
            diffs.append((m - pc[b].sum()) / pl[b].sum() * 100)
        lo, mid, hi = np.percentile(diffs, [2.5, 50, 97.5])
        print(f"  target {frac:.0%}: acc={pa.mean():.3f} cost=${pc.sum():.4f} saved_vs_large={(1 - pc.sum() / pl.sum()) * 100:.1f}% | vs random mix: {mid:+.1f} pts of large cost, 95% CI [{lo:+.1f}, {hi:+.1f}]")
