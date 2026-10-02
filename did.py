"""Difference-in-differences read on an offline event program, using a simulated 40-market weekly panel.

The simulation plants a known +6% lift in 20 markets that got field events from week 70, on top of
shared seasonality and a market-wide demand upswing that starts the same week. That upswing is the
"coincidence" a naive pre/post read mistakes for campaign impact.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRUE_LIFT, LAUNCH, N_MKT, N_WK = 0.06, 70, 40, 104
rng = np.random.default_rng(7)

# --- simulate -----------------------------------------------------------------
size = rng.lognormal(mean=5, sigma=0.5, size=N_MKT)               # market baseline sales
treated = np.zeros(N_MKT, bool)
treated[np.argsort(size)[-20:]] = True                             # events go to the biggest markets (not random)
wk = np.arange(N_WK)
season = 0.15 * np.sin(2 * np.pi * (wk - 10) / 52)                 # shared yearly seasonality
upswing = np.where(wk >= LAUNCH, 0.08, 0.0)                        # market-wide demand shift, same week as launch
rows = []
for m in range(N_MKT):
    noise = np.zeros(N_WK)
    for t in range(1, N_WK):                                       # AR(1) market noise
        noise[t] = 0.5 * noise[t - 1] + rng.normal(0, 0.03)
    log_sales = np.log(size[m]) + 0.001 * wk + season + upswing + noise
    log_sales += np.where(treated[m] & (wk >= LAUNCH), np.log1p(TRUE_LIFT), 0.0)
    rows += [dict(market=m, week=t, treated=int(treated[m]), post=int(t >= LAUNCH),
                  sales=np.exp(log_sales[t])) for t in wk]
df = pd.DataFrame(rows)
df["log_sales"] = np.log(df.sales)

# --- naive read: treated markets, 12 weeks after vs 12 weeks before ----------------
tr = df[df.treated == 1]
naive = (tr[tr.week.between(LAUNCH, LAUNCH + 11)].sales.mean()
         / tr[tr.week.between(LAUNCH - 12, LAUNCH - 1)].sales.mean() - 1)

# --- DiD: two-way fixed effects, SEs clustered by market --------------------------
fit = smf.ols("log_sales ~ treated:post + C(market) + C(week)", df).fit(
    cov_type="cluster", cov_kwds={"groups": df.market})
b, se = fit.params["treated:post"], fit.bse["treated:post"]
did, lo, hi = np.expm1(b), np.expm1(b - 1.96 * se), np.expm1(b + 1.96 * se)

# --- robustness ---------------------------------------------------------------------
# Balanced panel + one launch date: the TWFE estimate equals the simple 2x2 DiD of mean log sales,
# which is fast enough to re-run thousands of times.
def did2x2(d, launch, is_treated):
    diff = (d[d.week >= launch].groupby("market").log_sales.mean()
            - d[d.week < launch].groupby("market").log_sales.mean())
    return diff[is_treated].mean() - diff[~is_treated].mean()

assert abs(did2x2(df, LAUNCH, treated) - b) < 1e-9, "2x2 shortcut must match TWFE"

# placebo launches: every fake date in the pre-period (weeks 20-60), pre-period data only
pre = df[df.week < LAUNCH]
placebos = np.expm1([did2x2(pre, L, treated) for L in range(20, 61)])

# permutation inference: reassign the event program to 20 random markets 2,000 times
perm = np.array([did2x2(df, LAUNCH, rng.permutation(treated)) for _ in range(2000)])
perm_p = np.mean(perm >= b)

# --- event study: lift by weeks relative to launch (week -1 is the reference) -----------
rel = (df.week - LAUNCH).clip(-20, 20)
leads_lags = [k for k in range(-20, 21) if k != -1]
for k in leads_lags:
    df[f"d{k + 20}"] = ((rel == k) & (df.treated == 1)).astype(int)
es = smf.ols("log_sales ~ " + " + ".join(f"d{k + 20}" for k in leads_lags) + " + C(market) + C(week)", df).fit(
    cov_type="cluster", cov_kwds={"groups": df.market})
coef = {k: (es.params[f"d{k + 20}"], es.bse[f"d{k + 20}"]) for k in leads_lags}
coef[-1] = (0.0, 0.0)
ks = sorted(coef)
pretrend_p = float(es.f_test(" = 0, ".join(f"d{k + 20}" for k in leads_lags if k < -1) + " = 0").pvalue)
# 19 leads on 40 clusters makes that joint Wald test over-reject; a one-parameter
# differential-trend test on the pre-period is the better-behaved check.
tfit = smf.ols("log_sales ~ treated:week + C(market) + C(week)", pre).fit(
    cov_type="cluster", cov_kwds={"groups": pre.market})
trend_slope, trend_p = tfit.params["treated:week"], tfit.pvalues["treated:week"]

if __name__ == "__main__":
    print(f"Naive pre/post lift:  {naive:+.1%}")
    print(f"DiD lift:             {did:+.1%}  (95% CI {lo:+.1%} to {hi:+.1%})   true {TRUE_LIFT:+.0%}")
    print(f"Naive / true ratio:   {naive / TRUE_LIFT:.1f}x")
    print(f"Pre-trend joint F-test p = {pretrend_p:.2f} (19 leads, 40 clusters)")
    print(f"Differential pre-trend: {trend_slope * 100:+.3f}% per week, p = {trend_p:.2f}")
    print(f"Placebo launches (41 dates): max |lift| {np.max(np.abs(placebos)):.1%}, mean {placebos.mean():+.1%}")
    print(f"Permutation p (2,000 reassignments): {perm_p:.4f}; max permuted lift {np.expm1(perm.max()):+.1%}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    avg = df.groupby(["week", "treated"]).sales.mean().unstack()
    ax[0].plot(avg.index, avg[1], label="Event markets")
    ax[0].plot(avg.index, avg[0], label="Non-event markets")
    ax[0].axvline(LAUNCH, ls="--", c="gray")
    ax[0].set(title="Average weekly sales", xlabel="Week")
    ax[0].legend()
    est = np.array([np.expm1(coef[k][0]) for k in ks])
    err = np.array([1.96 * coef[k][1] for k in ks])
    ax[1].errorbar(ks, est, yerr=err, fmt="o", ms=3)
    ax[1].axhline(0, c="gray")
    ax[1].axhline(TRUE_LIFT, ls=":", c="green", label="True lift")
    ax[1].axvline(-0.5, ls="--", c="gray")
    ax[1].set(title="Event study: lift vs. weeks from launch", xlabel="Weeks from launch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig("figures/did_event_study.png", dpi=130)
