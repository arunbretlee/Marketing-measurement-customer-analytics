# Marketing Measurement & Customer Analytics

An end-to-end marketing analytics project focused on measuring marketing effectiveness, evaluating campaign impact, and identifying high-value customers. The project combines **Bayesian Marketing Mix Modeling (MMM)**, **Difference-in-Differences (DiD)**, **cohort retention analysis**, and **look-alike customer modeling** to translate marketing and customer data into actionable insights.

### What This Project Covers

* **Marketing Mix Modeling:** Built a Bayesian MMM using PyMC-Marketing to measure the contribution and ROAS of TV, OOH, Print, Facebook, and Search while accounting for adstock, saturation, seasonality, competitor sales, newsletters, and events.
* **Budget Optimization:** Evaluated channel-level ROAS and simulated budget reallocations under fixed spending constraints to estimate potential revenue improvements.
* **Campaign Measurement:** Applied Difference-in-Differences with market and week fixed effects to estimate the incremental impact of an offline event program while controlling for seasonality and market-wide demand changes.
* **Customer Retention:** Built cohort-based retention analysis to measure repeat-purchase behavior across customer acquisition cohorts.
* **Look-Alike Modeling:** Developed Logistic Regression and Gradient Boosting models using customers' first 90 days of behavior to identify customers likely to become high-value buyers.
* **Model Validation:** Used holdout testing, placebo tests, permutation inference, pre-trend analysis, ROC-AUC, and lift analysis to evaluate model reliability.

### Key Results

* MMM achieved approximately **4% holdout MAPE** with an in-sample **R² of 0.91**.
* Estimated that marketing media contributed approximately **13% of revenue** during the training period.
* Identified substantial differences in channel-level ROAS and evaluated a budget scenario that increased modeled media-driven revenue by approximately **25%** while maintaining the same total budget.
* DiD estimated a **5.6% campaign lift** compared with a **28.9% naive pre/post estimate**, demonstrating the importance of controlling for market-wide trends.
* The look-alike model achieved **0.77 AUC** and captured **31% of eventual best customers within the top 10% of scored customers**, representing a **3.1× lift over random targeting**.

### Tech Stack

**Python • PyMC-Marketing • PyMC • Pandas • NumPy • Scikit-learn • Statsmodels • Matplotlib • Bayesian Modeling • Marketing Mix Modeling • Causal Inference • Cohort Analysis • Customer Segmentation • Predictive Modeling**

### Data

The project uses Meta's public **Robyn simulated weekly marketing dataset** for MMM and the **CDNOW customer transaction dataset** for cohort and look-alike analysis. The project clearly distinguishes simulated data from real customer transaction data when interpreting results.
