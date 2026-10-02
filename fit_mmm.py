"""Fit a Bayesian marketing mix model (PyMC-Marketing) on Robyn's public weekly dataset.

Channels: TV, OOH, print (offline) and Facebook, search (digital) spend.
Controls: competitor sales, newsletter sends, event weeks, yearly seasonality.
Last 26 weeks held out to check out-of-sample fit.
"""
import numpy as np
import pandas as pd
import pyreadr
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

df = list(pyreadr.read_r("data/dt_simulated_weekly.RData").values())[0]
df["DATE"] = pd.to_datetime(df["DATE"])
df["event"] = (df["events"] != "na").astype(float)
for c in ["competitor_sales_B", "newsletter"]:   # put controls on a 0-1 scale like the media inputs
    df[c] = df[c] / df[c].max()

CHANNELS = ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"]
CONTROLS = ["competitor_sales_B", "newsletter", "event"]
HOLDOUT = 26

X = df[["DATE", *CHANNELS, *CONTROLS]]
y = df["revenue"]

mmm = MMM(
    date_column="DATE",
    channel_columns=CHANNELS,
    control_columns=CONTROLS,
    target_column="revenue",
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    yearly_seasonality=2,
)

if __name__ == "__main__":
    mmm.fit(X.iloc[:-HOLDOUT], y.iloc[:-HOLDOUT], chains=4, cores=2, draws=1000, tune=1500,
            target_accept=0.95, random_seed=42)
    mmm.save("mmm_fit.nc")
    print("saved mmm_fit.nc")
