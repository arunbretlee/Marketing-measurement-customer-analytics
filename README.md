# Marketing Measurement & Customer Analytics

Three questions every go-to-market team asks: **which channels actually drive revenue and how should budget move**, **did a campaign really work or ride a coincidence**, and **which new customers look like our best ones**. Part 1 answers the first with a Bayesian marketing mix model, Part 2 the second with a difference-in-differences read on an offline event program, and Part 3 the third with cohort retention reporting and a look-alike model on real customer transactions.

## Part 1: Bayesian marketing mix model (PyMC-Marketing)

**Data.** Meta's public Robyn demo dataset (`dt_simulated_weekly`): 208 weeks of revenue (Nov 2015 – Nov 2019) with spend on three offline channels (TV, out-of-home, print) and two digital channels (Facebook, search). Newsletter sends, event weeks and competitor sales are controls.

**Model.** Geometric adstock (up to 8 weeks of carryover) and logistic saturation per channel, yearly seasonality (Fourier terms) and the controls. 4 chains × 1,000 draws; max r-hat 1.003, no divergences.

**Validation.** The last 26 weeks were held out. Holdout MAPE was about 4% (in-sample R² 0.91, MAPE 6.6%).

**Findings**
- Media drove about 13% of revenue over the training window.
- Out-of-home took 64% of spend but produced 18% of media-driven revenue: ROAS 0.90 (90% interval 0.18–1.74), the only channel near break-even.
- TV was the most reliable performer: ROAS 6.1 (4.3–8.2). Facebook, print and search show higher point estimates with wide intervals, so they are candidates to test before scaling.
- **Budget scenario.** Holding weekly spend flat and limiting each channel to 50–150% of its historical average, moving budget out of out-of-home raises modeled media-driven revenue by 25% (90% interval +18% to +32%). Every channel except out-of-home hit its +50% cap, so read this as a direction plus a test plan rather than a final allocation.

![MMM results](figures/mmm_roas.png)

## Part 2: Difference-in-differences read on an offline event program

**Setup.** A simulated 40-market × 104-week sales panel. Field events launch in the 20 largest markets in week 70 with a planted **+6%** lift. A market-wide demand upswing starts the same week, on top of yearly seasonality. Because the true effect is known, the simulation can show whether each method gets the answer right.

| Method | Estimated lift |
|---|---|
| Naive pre/post (event markets, 12 weeks after vs. before) | **+28.9%**, about 4.8× the truth |
| Difference-in-differences (two-way fixed effects, market-clustered SEs) | **+5.6%** (95% CI +4.9% to +6.3%) |

**Checks**
- Parallel trends: the differential pre-launch trend is +0.004% per week (p = 0.68).
- Placebo launches: 41 fake launch dates in the pre-period produced lifts of at most 0.9%.
- Permutation inference: across 2,000 random reassignments of which markets got events, none produced a lift as large as the real one; the largest was 2.9%.
- A joint Wald test of all 19 event-study leads rejected (p = 0.01) even though the simulation has no pre-trend. That test is known to over-reject with few clusters, which is why the single-parameter trend test is the one reported above.

![Event study](figures/did_event_study.png)

## Part 3: Cohort retention and a look-alike model (real customer data)

**Data.** CDNOW transactions: 69,659 purchases by 23,570 real customers who first bought between January and March 1997, tracked through June 1998 (bundled with the `lifetimes` package).

**Cohort report.** Customers were grouped by first-purchase month. About 15% of each cohort bought again in month 1, falling to roughly 7–9% by month 12. The heatmap below tracks every cohort month by month.

**Look-alike model.** "Best customers" are the top 20% by spend in the year after their first 90 days; they produced 90% of that year's revenue. Using only first-90-day behavior (order count, spend, CDs, first-order size, days to second order, cohort), the model scores how closely a new customer resembles that group.
- Logistic regression: AUC 0.77 on a 30% holdout. The top-scoring 10% of customers contain 31% of eventual best customers, a 3.1x lift over random.
- Gradient boosting matched it (AUC 0.77) without improving on it, so the simpler, more explainable logistic model is the one to ship.

![Cohorts and look-alike](figures/cohorts_lookalike.png)

## Limitations
- The Robyn data is simulated, so channel ROAS here demonstrates the method, not real-world channel performance.
- ROAS is measured in-sample. In practice I would calibrate the MMM with lift-test results and refresh it on a regular cadence.
- The DiD panel is simulated so the true answer is known. Real event programs also need checks for spillover between nearby markets.

## Reproduce
```bash
pip install -r requirements.txt
python fit_mmm.py      # fits the MMM (~1 minute on 2 cores), saves mmm_fit.nc
python analyze_mmm.py  # convergence, holdout error, ROAS, budget scenario, figures/mmm_roas.png
python did.py          # difference-in-differences, placebo and permutation tests, figures/did_event_study.png
python cohorts_lookalike.py  # cohort retention + look-alike model, figures/cohorts_lookalike.png
```

Data: `data/dt_simulated_weekly.RData` from [facebookexperimental/Robyn](https://github.com/facebookexperimental/Robyn) (MIT license); `data/CDNOW_master.txt` from the [lifetimes](https://github.com/CamDavidsonPilon/lifetimes) package.
