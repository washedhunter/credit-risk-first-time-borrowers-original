# Design decisions

Plain-English notes on why the project is built the way it is, and what else was considered.
Numbers quoted here come from `reports/metrics.md` (the committed training run).

## Data and evaluation

**Hold-out from Kaggle's training file, not Kaggle's test set.** Kaggle's `application_test` has no
labels, so it can't be scored locally. 20% of `application_train` (61,503 applicants) is held out,
stratified on default × thin-file so the small thin-file segment appears in the same proportion in
both parts. Everything that involves a choice (hyper-parameters, imbalance strategy, threshold)
is decided on the other 80% with 5-fold out-of-fold predictions. The hold-out is scored once,
at the end. That way the reported numbers aren't inflated by choices made while looking at them.

**Bootstrap confidence intervals.** Only ~500 hold-out applicants are thin-file, with ~36 defaults.
An AUC from 36 positives moves a lot by chance, so every hold-out metric comes with a 95%
bootstrap interval, and thin-file metrics are also reported out-of-fold on the ~2,000 thin-file
applicants in the training split as a larger-sample check.

**No raw data in git.** The Kaggle competition rules (as summarised by secondary sources; the
rules page only renders in a browser) don't allow redistributing the data. The download script
needs your own Kaggle token. The demo uses a **synthetic** sample instead (below).

## Features

**A reusable pipeline module, not a notebook.** `src/credit_risk/features/` has one module per
source table plus `build.py` to join them. The same `add_ratio_features` function runs in training
and in the API, so a ratio can't be computed one way in training and another way in production;
a test checks the API path and the training path produce identical frames.

**Hand-picked aggregates (~160 features), not thousands of automatic ones.** Kaggle kernels often
generate 700+ features (every min/max/mean/var of every column). That gains maybe 0.01 AUC but
makes the API schema, the reason codes and the interview explanation unmanageable. Each feature
here has a plain-English label and a reason group (`scoring/feature_info.py`), and a test fails if
a feature is added without one.

What the aggregates capture, per the resume bullet:
- *Credit-bureau history (other lenders):* number of loans, active/closed, recency of the newest
  and oldest, outstanding debt / original amount, card utilisation, overdue amounts, months paid
  late (from `bureau_balance`), bad debt.
- *Previous Home Credit applications:* approval/refusal rates, recency of the last refusal, amounts
  applied for vs granted, an interest-rate proxy, loans still running.
- *Repayment records:* share of instalments paid late (and 30+ days late), share underpaid, share
  of the total amount left unpaid, missed instalments, last-year late rate; POS/cash arrears;
  credit-card utilisation and paying below the minimum.

**Sentinels and missing values.** `365243` is Home Credit's placeholder for "no date" (e.g.
`DAYS_EMPLOYED` for pensioners) and becomes missing. Missing is left as missing: LightGBM learns
which way to send it, and for a first-time borrower "no history" is genuine information. Counts
(`BUR_COUNT`, `PREV_COUNT`) are 0, not missing, when there are no rows.

**Leakage.** There is no target encoding anywhere, so nothing target-dependent needs to be fitted
inside the folds. LightGBM handles categorical columns natively (its split statistics are computed
on the training fold only). The logistic baseline's imputer, quantile transform and one-hot
encoder sit inside an sklearn `Pipeline`, which `cross_val_predict` refits on each training fold.
Aggregates only use the applicant's own records.

## Fairness

**Excluded from the model:** `CODE_GENDER`, `DAYS_BIRTH` (age), `NAME_FAMILY_STATUS`,
`CNT_CHILDREN`, `CNT_FAM_MEMBERS`. Gender and marital status are protected characteristics under
fair-lending laws in many countries (e.g. the US Equal Credit Opportunity Act, the EU's equal
treatment rules), and RBI's Fair Practices Code bars discrimination on grounds such as sex.
Age is a protected characteristic in many places too, and family size is a proxy for marital
status. Dropping them costs a little accuracy (age is predictive in this data) and that cost is
accepted.

**Excluding is not the same as being fair.** Other features can act as proxies (occupation,
employer type, years in the job all correlate with age and gender). So gender and age are kept
*outside* the feature set purely to audit outcomes on the hold-out: approval rate and default rate
among approved loans by group. The audit table is in the README. It shows differences in approval
rates that track the differences in actual default rates; whether those are acceptable is a
policy decision a real lender would have to make (and document), not something a model can settle.
In this run the mean predicted score is slightly above the actual default rate for women
(7.3% vs 7.0%) and slightly below it for men (9.3% vs 10.1%), so women aren't penalised
relative to their observed risk.

## Model

**Logistic regression first.** It's the industry-standard scorecard model and a sanity check: if
the boosted model can't beat it clearly, the extra complexity isn't worth it.

**LightGBM with stratified 5-fold CV and early stopping.** Gradient-boosted trees handle missing
values, skewed amounts and non-linear effects without manual binning, and LightGBM is fast on
250K × 160. Early stopping (200 rounds) picks the number of trees per fold; the final model is
refitted on the whole training split with the mean of the folds' best iteration counts.

**Optuna, bounded.** A fixed budget of 40 trials, 3-fold CV, at learning rate 0.1 (faster trials),
on the training split only. The best parameters are committed in `config/lgbm_params.json`, so
`credit-risk train` reproduces the reported model without re-running the search; `credit-risk
tune` re-runs it. The first search put `num_leaves` on the lower edge of its range (15), so the
range was widened to 7–127 and the search was re-run; only the second run's parameters are used
(best 3-fold CV AUC 0.7883 vs 0.7874). The winner is small trees (9 leaves) with strong
regularisation; `min_child_samples` ended near its upper edge (489 of 500), so a wider search might
gain a little more, but the second search only moved CV AUC by 0.001.

**Class imbalance (~8% defaults).** Three options were compared out-of-fold: no reweighting,
`scale_pos_weight = sqrt(neg/pos)` and `scale_pos_weight = neg/pos` ("balanced"). Reweighting
inflates the predicted probabilities (the model is no longer calibrated), which matters because
the approval rule compares a probability with a cost-derived threshold. The strategy with the
lowest out-of-fold expected cost is used. In the committed run, **no reweighting** won: best ROC-AUC
(0.7907 vs 0.7897 sqrt and 0.7886 balanced), calibrated probabilities (mean prediction 8.0% vs an 8.1%
default rate; 37% under "balanced"), and the lowest expected cost. Reweighting shifts the scores
upwards but doesn't rank borrowers better, and the cost-tuned threshold already handles the imbalance.

## Threshold

**Expected cost, in money, proportional to the loan.** Approving a borrower who defaults loses
`LGD × loan amount` (LGD = 45%, the Basel foundation-IRB figure for senior unsecured exposures);
declining a borrower who would have repaid loses `margin × loan amount` (10%, an assumption for the
lifetime profit of a consumer loan). Both are illustrative, not Home Credit's real economics, and
both are environment variables. For a calibrated model the break-even probability is
`margin / (margin + LGD)` ≈ 0.18; the threshold actually used is the one that minimises total
expected cost on the out-of-fold predictions, which also absorbs any miscalibration.

Weighting by loan amount (rather than counting every loan the same) matters: a default on a large
loan costs more than a declined small one. A sensitivity table in the README shows how the
threshold and approval rate move when LGD and margin change.

**One threshold for everyone.** A segment-specific threshold for thin-file applicants is computed
and reported, but not deployed: with ~2,000 thin-file training applicants it is noisy, and a
separate cut-off for one group of customers is a policy choice that needs its own justification.

## Explainability

**SHAP (TreeSHAP).** Exact per-applicant attributions for tree models, in log-odds, that add up to
the model's output. Training produces the global summary (beeswarm), the mean-|SHAP| bar chart
and an example waterfall with the `shap` package. The API calls LightGBM's built-in
`pred_contrib=True` directly. For LightGBM models `shap.TreeExplainer` calls exactly that function
(see `shap/explainers/_tree.py`), so the values are identical (max difference 0.0 on the hold-out
sample; a test guards it), and `shap`/`numba` (hundreds of MB) stay out of the API image.

**Reason codes are grouped.** Summing SHAP values within ~12 reason groups (external scores,
affordability, bureau arrears, repayment record at Home Credit, ...) means three correlated
features (say, the three external scores) give one reason instead of three. Each reason names its
most influential feature and its value, e.g. "Late or short payments on earlier Home Credit loans
(Share of instalments paid late: 31%)". When the driving feature is missing, the reason is the
absence itself ("No credit-bureau record"), which is what "insufficient credit history" means for
a first-time borrower.

## Serving

**FastAPI with a generated request model.** The `Applicant` Pydantic model is built from the
trained model's feature list at startup: every field typed, documented and bounded where it makes
sense (scores and flags in [0, 1], counts ≥ 0, income and loan amount > 0), unknown fields
rejected (`extra="forbid"`, which also rejects protected attributes and the derived ratios),
categorical fields restricted to the levels seen in training, and NaN/Infinity rejected. Only
income and loan amount are required; history fields are optional because a first-time borrower
has none. If fields that 99% of real applications have are missing, the response carries a
warning, because the model hasn't seen inputs like that.

**The dashboard only talks to the API.** It never loads the model, so there's exactly one place
where scoring happens. What-if sliders send modified applications to `/predict`; the server
recomputes the ratio features.

**Two images.** The API image has only the scoring dependencies (pandas, numpy, LightGBM,
FastAPI); a test checks that importing the API doesn't pull in sklearn, shap, optuna, matplotlib
or kaggle. The dashboard image has Streamlit and Plotly.

## Synthetic sample

The demo and CI must work without Kaggle, but the raw data can't be committed. So
`data/sample/synthetic_applicants.csv` holds 300 **synthetic** applicants (60 thin-file,
60 with Home Credit history but no bureau record, 180 with a bureau record), generated by a
Gaussian copula fitted on the training split's engineered features, per segment:
each column's values become normal scores by rank; columns that are sometimes missing also get a
"present/missing" latent; correlations are estimated pairwise; samples are mapped back through
each column's interpolated percentiles. Values are smoothed population percentiles, so no row is a
real applicant (a test checks no synthetic row equals a real one), while realistic patterns such
as "all bureau fields missing together" are preserved. A few logical constraints are then
enforced (e.g. active bureau loans ≤ all bureau loans). The sample is good for a demo, not for
measuring accuracy; all reported metrics come from the real hold-out.

## Alternatives considered

- **CatBoost / XGBoost:** both are common on this dataset (not benchmarked here); LightGBM was the
  stack named in the resume and is the fastest of the three here.
- **SMOTE / undersampling:** resampling changes the class balance the model sees, just like
  `scale_pos_weight`, but adds synthetic or discards real rows; reweighting was the simpler
  comparison to make, and threshold tuning addresses the business question directly.
- **Probability calibration (isotonic/Platt) on top of a reweighted model:** a sensible next step if
  reweighting had won on ranking metrics; the cost-optimal threshold already absorbs a monotone
  miscalibration for the approve/decline decision.
- **Serving the `shap` package:** rejected for image size; LightGBM's own TreeSHAP gives the same
  numbers.
