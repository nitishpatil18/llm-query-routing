import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

SEED = 42
d = np.load("router_features_v2.npz", allow_pickle=True)
X, correct, cost, ctok = d["X"], d["correct"], d["cost"], d["ctok"]
tiers = [str(t) for t in d["tiers"]]
S, L = tiers.index("small"), tiers.index("large")
n = len(X)
cs, cl = correct[:, S], correct[:, L]
cost_s, cost_l = cost[:, S], cost[:, L]
acc_large, cost_large = cl.mean(), cost_l.sum()

print(f"n={n}")
print(f"always-small acc={cs.mean():.3f} cost=${cost_s.sum():.4f}")
print(f"always-large acc={acc_large:.3f} cost=${cost_large:.4f}")

log_len = np.log1p(ctok[:, S]).reshape(-1, 1)
variants = [
    ("pre-call router: embedding only", X, "route"),
    ("cascade: small's token count only", log_len, "cascade"),
    ("cascade: embedding + small's token count", np.hstack([X, log_len]), "cascade"),
]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
taus = np.linspace(0, 1.0001, 400)

for name, feats, mode in variants:
    clf = XGBClassifier(n_estimators=100, max_depth=3, eval_metric="logloss", random_state=SEED)
    p = cross_val_predict(clf, feats, cs, cv=skf, method="predict_proba")[:, 1]  # P(small is correct)
    print(f"\n--- {name} --- AUC={roc_auc_score(cs, p):.3f}")
    sweep = []
    for tau in taus:
        esc = p < tau  # True means the question goes to the large model
        acc = np.where(esc, cl, cs).mean()
        if mode == "route":
            c = np.where(esc, cost_l, cost_s).sum()
        else:
            c = cost_s.sum() + cost_l[esc].sum()  # escalated questions pay for both calls
        sweep.append((acc, c, esc.mean()))
    for frac in [1.0, 0.98, 0.95]:
        ok = [s for s in sweep if s[0] >= frac * acc_large - 1e-12]
        if not ok:
            print(f"  retain {frac:.0%} of always-large acc: not reached")
            continue
        acc, c, e = min(ok, key=lambda s: s[1])
        print(f"  retain {frac:.0%}: acc={acc:.3f} cost=${c:.4f} saved_vs_large={(1 - c / cost_large) * 100:.1f}% escalated={e:.2f}")
