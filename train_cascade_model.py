import numpy as np, joblib
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

SEED = 42
d = np.load("router_features_v2.npz", allow_pickle=True)
correct, cost, ctok = d["correct"], d["cost"], d["ctok"]
tiers = [str(t) for t in d["tiers"]]
S, L = tiers.index("small"), tiers.index("large")
cs, cl = correct[:, S], correct[:, L]
x = np.log1p(ctok[:, S]).reshape(-1, 1)


def make():
    return XGBClassifier(n_estimators=100, max_depth=3, eval_metric="logloss", random_state=SEED)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
p = cross_val_predict(make(), x, cs, cv=skf, method="predict_proba")[:, 1]
model = make().fit(x, cs)
joblib.dump(model, "cascade_model.joblib")
np.savez("cascade_oof.npz", p=p, cs=cs, cl=cl, ps=cost[:, S], pl=cost[:, L])
print(f"rows={len(x)} small acc={cs.mean():.3f} large acc={cl.mean():.3f}")
for t in [100, 300, 1000, 2000, 4096]:
    print(f"small used {t:>4} completion tokens -> P(small correct) = {model.predict_proba(np.log1p([[t]]))[0, 1]:.2f}")
