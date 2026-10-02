"""Read the fitted MMM: convergence, holdout accuracy, channel ROAS, and a budget reallocation scenario."""
import sys
import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymc_marketing.mmm import MMM
from fit_mmm import X, y, HOLDOUT, CHANNELS

m = MMM.load(sys.argv[1] if len(sys.argv) > 1 else "mmm_fit.nc")
train_X, test_X, train_y, test_y = X.iloc[:-HOLDOUT], X.iloc[-HOLDOUT:], y.iloc[:-HOLDOUT], y.iloc[-HOLDOUT:]
NAMES = {"tv_S": "TV", "ooh_S": "OOH", "print_S": "Print", "facebook_S": "Facebook", "search_S": "Search"}


def mape(actual, pred):
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(pred)) / np.asarray(actual)))


# convergence
rhat = float(az.summary(m.idata, var_names=["adstock_alpha", "saturation_lam", "saturation_beta", "gamma_control", "gamma_fourier", "intercept_contribution", "y_sigma"])["r_hat"].max())
divergences = int(m.idata.sample_stats.diverging.sum())

# fit: in-sample and 26-week holdout
fitted = m.sample_posterior_predictive(train_X, extend_idata=False, progressbar=False)
fitted = fitted[m.target_column if m.target_column in fitted else "y"].mean(("sample")).values * float(m.idata.constant_data["target_scale"])
pred = m.sample_posterior_predictive(test_X, extend_idata=False, include_last_observations=True, progressbar=False)
pred = pred[m.target_column if m.target_column in pred else "y"].mean(("sample")).values * float(m.idata.constant_data["target_scale"])
r2 = 1 - np.sum((train_y - fitted) ** 2) / np.sum((train_y - train_y.mean()) ** 2)

# channel contribution and ROAS over the training window
post = m.idata.posterior
contrib = (post["channel_contribution"] * float(m.idata.constant_data["target_scale"])).sum("date")  # (chain, draw, channel)
spend = xr.DataArray(train_X[CHANNELS].sum().values, dims="channel", coords={"channel": CHANNELS})
roas = contrib / spend
table = pd.DataFrame({
    "spend": spend.values,
    "contribution": contrib.mean(("chain", "draw")).values,
    "roas": roas.mean(("chain", "draw")).values,
    "roas_lo": roas.quantile(0.05, ("chain", "draw")).values,
    "roas_hi": roas.quantile(0.95, ("chain", "draw")).values,
}, index=[NAMES[c] for c in CHANNELS])
table["spend_share"] = table.spend / table.spend.sum()
table["effect_share"] = table.contribution / table.contribution.sum()
media_share = float(contrib.sum("channel").mean() / train_y.sum())

# budget scenario: same total weekly budget over the next 12 weeks, each channel within 50-150% of its average
avg = train_X[CHANNELS].mean()
weekly_budget = float(avg.sum())
bounds = xr.DataArray(np.stack([avg.values * 0.5, avg.values * 1.5], axis=1),
                      dims=("channel", "bound"), coords={"channel": CHANNELS, "bound": ["lower", "upper"]})
start = X.DATE.iloc[-HOLDOUT]
opt = m.budget_optimizer(start, start + pd.Timedelta(weeks=11))
result = opt.allocate_budget(total_budget=weekly_budget, budget_bounds=bounds)
current = xr.DataArray(avg.values, dims="channel", coords={"channel": CHANNELS})
resp_cur = opt.evaluate_response_distribution(current)
resp_opt = opt.evaluate_response_distribution(result.budgets)
gain = (resp_opt - resp_cur) / resp_cur
alloc = pd.DataFrame({"current": avg.values, "optimized": result.budgets.sel(channel=CHANNELS).values},
                     index=[NAMES[c] for c in CHANNELS])
alloc["change"] = alloc.optimized / alloc.current - 1

if __name__ == "__main__":
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
    print(f"max r_hat {rhat:.3f}, divergences {divergences}")
    print(f"in-sample R^2 {r2:.3f}, in-sample MAPE {mape(train_y, fitted):.1%}, 26-week holdout MAPE {mape(test_y, pred):.1%}")
    print(f"media share of revenue: {media_share:.1%}")
    print(table.round(3))
    print(f"weekly budget {weekly_budget:,.0f}; optimized sums to {float(result.budgets.sum()):,.0f}")
    print(alloc.round(3))
    print(f"response gain at same budget: mean {float(gain.mean()):+.1%}, 90% interval "
          f"{float(gain.quantile(0.05)):+.1%} to {float(gain.quantile(0.95)):+.1%}, P(gain>0) {float((gain > 0).mean()):.0%}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    table[["spend_share", "effect_share"]].plot.bar(ax=ax[0], rot=0)
    ax[0].set(title="Share of spend vs. share of media-driven revenue")
    ax[0].legend(["Spend share", "Revenue contribution share"])
    ax[1].bar(table.index, table.roas)
    ax[1].errorbar(table.index, table.roas, yerr=[table.roas - table.roas_lo, table.roas_hi - table.roas],
                   fmt="none", c="black", capsize=4)
    ax[1].axhline(1, ls="--", c="gray")
    ax[1].set(title="ROAS by channel (90% credible interval)")
    fig.tight_layout()
    fig.savefig("figures/mmm_roas.png", dpi=130)
