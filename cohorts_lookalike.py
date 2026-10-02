"""Cohort retention and a look-alike propensity model on real CDNOW customer transactions.

23,570 customers who first bought in Jan-Mar 1997, tracked through June 1998.
Look-alike: from a customer's first 90 days only, score how closely they resemble the
customers who become the top 20% by spend over the following year.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

tx = pd.read_csv("data/CDNOW_master.txt", sep=r"\s+", dtype={"customer_id": str})
tx["date"] = pd.to_datetime(tx["date"].astype(str), format="%Y%m%d")
first = tx.groupby("customer_id").date.min().rename("first_date")
tx = tx.join(first, on="customer_id")
tx["day"] = (tx.date - tx.first_date).dt.days

# --- cohort retention: share of each monthly cohort buying again in month k ----------
tx["cohort"] = tx.first_date.dt.to_period("M")
tx["k"] = (tx.date.dt.to_period("M") - tx.cohort).apply(lambda d: d.n)
sizes = tx.groupby("cohort").customer_id.nunique()
retention = tx.groupby(["cohort", "k"]).customer_id.nunique().unstack().div(sizes, axis=0)
rev_per_cust = tx.groupby(["cohort", "k"]).dollar_value.sum().unstack().div(sizes, axis=0)

# --- look-alike: features from days 0-89, target from days 90-454 ---------------------
early, later = tx[tx.day < 90], tx[(tx.day >= 90) & (tx.day < 455)]
orders = early.groupby(["customer_id", "date"]).agg(cds=("number_of_cds", "sum"), spend=("dollar_value", "sum")).reset_index()
g = orders.groupby("customer_id")
X = pd.DataFrame({
    "orders_90d": g.size(),
    "spend_90d": g.spend.sum(),
    "cds_90d": g.cds.sum(),
    "first_order_spend": g.spend.first(),
    "first_order_cds": g.cds.first(),
    "days_to_2nd_order": g.date.apply(lambda d: (d.iloc[1] - d.iloc[0]).days if len(d) > 1 else 90),
    "cohort_month": first.dt.month,
}).fillna(0)
future = later.groupby("customer_id").dollar_value.sum().reindex(X.index, fill_value=0)
y = (future >= future.quantile(0.8)).astype(int)     # "best customers": top 20% of next-year spend
assert future.quantile(0.8) > 0, "top-20% label needs more than 20% of customers with future spend"

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
models = {
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Gradient boosting": HistGradientBoostingClassifier(random_state=42),
}
scores = {name: m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1] for name, m in models.items()}


def decile_capture(y_true, s):
    """Share of all positives found in the top 10% of scores."""
    top = np.argsort(-s)[: int(len(s) * 0.1)]
    return y_true.values[top].sum() / y_true.sum()


if __name__ == "__main__":
    print(f"customers {len(X):,}; cohorts {dict(sizes.astype(int))}")
    print("repeat-purchase rate by months since first purchase (avg of cohorts):")
    print(retention.mean().loc[1:12].round(3).to_string())
    print(f"best-customer base rate {y.mean():.1%}; best customers' share of next-year revenue "
          f"{future[y == 1].sum() / future.sum():.1%}")
    for name, s in scores.items():
        cap = decile_capture(y_te, s)
        print(f"{name}: AUC {roc_auc_score(y_te, s):.3f}, top-decile captures {cap:.1%} of best customers ({cap / 0.1:.1f}x lift)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    show = retention.loc[:, 1:15]
    im = ax[0].imshow(show.values, cmap="Blues", aspect="auto")
    ax[0].set(title="Repeat-purchase rate by acquisition cohort", xlabel="Months since first purchase",
              yticks=range(len(show)), yticklabels=[str(c) for c in show.index],
              xticks=range(0, 15, 2), xticklabels=range(1, 16, 2))
    fig.colorbar(im, ax=ax[0], format="{x:.0%}")
    for name, s in scores.items():
        order = np.argsort(-s)
        captured = np.cumsum(y_te.values[order]) / y_te.sum()
        ax[1].plot(np.arange(1, len(s) + 1) / len(s), captured, label=name)
    ax[1].plot([0, 1], [0, 1], ls="--", c="gray", label="Random")
    ax[1].set(title="Look-alike model: best customers captured", xlabel="Share of customers contacted (by score)",
              ylabel="Share of best customers found")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig("figures/cohorts_lookalike.png", dpi=130)
