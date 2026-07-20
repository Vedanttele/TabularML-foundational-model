# Tabular Foundation Models vs Tuned GBDTs on HR Attrition Data

A small, controlled benchmark that asks two questions:

1. On a typical enterprise HR table (1470 rows), does a tabular foundation model beat tuned gradient boosting?
2. Does the foundation model still work when you strip away every semantic clue (column names and category labels)?

Models compared: **SAP RPT 1 (OSS)**, **XGBoost**, **LightGBM**, **CatBoost**.
Dataset: IBM HR Analytics Employee Attrition (Kaggle).

---

## 1. Why this question matters

Most enterprise tables are small. A few thousand rows, thirty columns, a business owner who wants an answer this week. The default answer for the last ten years has been "tune a GBDT". Tabular foundation models such as SAP RPT 1, Google TabFM and TabPFN change the shape of that answer: they treat the training rows as a prompt and predict in a single forward pass, with no per dataset training loop and no hyperparameter search.

The second question is the one enterprises actually care about. Real customer tables have messy, inconsistent, sometimes anonymised column names. If a model only works because it read the string `OverTime` and knew what overtime means, it will fall apart on a client system where that column is called `Z_FLD_0042`. So the benchmark runs every experiment twice: once with real names, once fully masked.

---

## 2. Data

| Item | Value |
|---|---|
| Raw file | 1470 rows, 35 columns |
| Dropped | EmployeeCount, Over18, StandardHours (constant), EmployeeNumber (identifier) |
| Working table | 1470 rows, 31 columns |
| Categorical columns | 7 (BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus, OverTime) |

**Task A, classification.** Target `Attrition`. 30 features. Class balance 237 Yes vs 1233 No, about 16 percent positive, roughly 47 positives per test fold.

**Task B, regression.** Target `MonthlyIncome`. 28 features. `JobLevel` is dropped as well because it correlates about 0.95 with income and turns the task into a lookup.

---

## 3. The semantic masking procedure

This is the core experimental device, so it is worth stating exactly.

```python
def mask_semantics(X, seed=42):
    # 1. rename every column to col_00, col_01, ...
    # 2. replace each categorical level with a deterministic hash token, cat_a87f
    # 3. leave every numeric value byte identical
```

Two properties matter:

* **Information preserving.** Nothing is deleted. The same rows, same values, same cardinalities. Only the meaning of the strings is destroyed.
* **Deterministic.** Seeded, so the masked table is reproducible across folds and models.

A tree model should be completely indifferent to this, because it only sees integer encoded splits. A foundation model that leans on language priors should degrade. That difference is the measurement.

---

## 4. Algorithm

```
for task in [TaskA classification, TaskB regression]:
    for variant in [semantic, masked]:
        X = raw or mask_semantics(raw)
        folds = StratifiedKFold(5) if classification else KFold(5)
        for fold in folds:
            for model in [xgboost, lightgbm, catboost, sap_rpt]:

                if model is a GBDT:
                    label encode categoricals using train fold levels only
                    Optuna search, N_TRIALS, 3 fold inner CV on the train fold
                    refit best params on full train fold
                    time the fit
                else:
                    SAP_RPT_OSS(max_context_size=2048, bagging=2)
                    fit on train fold (in context, no gradient updates)
                    time the fit

                predict on test fold
                record AUC and macro F1, or R2 and RMSE
```

Fixed seed 42 throughout. Same folds for every model, so all comparisons are paired.

---

## 5. Results

All numbers are means over 5 folds, computed from [`Final_result.csv`](./Final_result.csv).

### Task A, attrition classification

| Variant | Model | AUC | Macro F1 |
|---|---|---|---|
| semantic | **sap_rpt** | **0.837** | **0.687** |
| semantic | catboost | 0.813 | 0.570 |
| semantic | lightgbm | 0.806 | 0.665 |
| semantic | xgboost | 0.805 | 0.650 |
| masked | **sap_rpt** | **0.834** | 0.671 |
| masked | catboost | 0.819 | 0.560 |
| masked | xgboost | 0.806 | **0.689** |
| masked | lightgbm | 0.805 | 0.666 |

### Task B, monthly income regression

| Variant | Model | R2 | RMSE |
|---|---|---|---|
| semantic | **sap_rpt** | **0.894** | **1525** |
| semantic | xgboost | 0.877 | 1640 |
| semantic | catboost | 0.864 | 1726 |
| semantic | lightgbm | 0.854 | 1783 |
| masked | **sap_rpt** | **0.894** | **1523** |
| masked | catboost | 0.873 | 1666 |
| masked | xgboost | 0.868 | 1705 |
| masked | lightgbm | 0.860 | 1751 |

### Paired t tests, SAP RPT against each GBDT across the 5 shared folds

| Comparison | Task A AUC | Task B R2 |
|---|---|---|
| vs XGBoost (semantic) | +0.032, p = 0.097 | +0.017, p = 0.013 |
| vs LightGBM (semantic) | +0.031, p = 0.016 | +0.040, p = 0.012 |
| vs CatBoost (semantic) | +0.024, p = 0.007 | +0.030, p = 0.006 |
| vs XGBoost (masked) | +0.028, p = 0.052 | +0.027, p = 0.003 |
| vs LightGBM (masked) | +0.029, p = 0.029 | +0.034, p = 0.024 |
| vs CatBoost (masked) | +0.015, p = 0.098 | +0.021, p = 0.005 |

### Robustness to masking, semantic minus masked

| Model | Task A AUC drop | Task B R2 drop |
|---|---|---|
| sap_rpt | +0.003 (p = 0.44) | 0.000 (p = 0.74) |
| xgboost | 0.001 (p = 0.91) | +0.010 (p = 0.17) |
| lightgbm | +0.001 (p = 0.92) | 0.006 (p = 0.47) |
| catboost | 0.006 (p = 0.37) | 0.009 (p = 0.11) |

---

## 6. What the numbers actually say

**Finding 1. SAP RPT 1 wins on both tasks, and the regression win is the more solid one.**
Every regression comparison is significant at p below 0.03. On classification the AUC edge is consistent in sign across all six comparisons but only three of six clear p = 0.05, which is what you expect from 5 folds and 47 positives per fold. Treat the classification result as a directional signal, not a proven gap.

**Finding 2. Masking costs SAP RPT 1 essentially nothing.**
Task A AUC moves 0.003 and Task B R2 moves 0.000, both far inside fold noise. The model is not riding on the English in the column headers. For enterprise deployment across clients with different schemas, this is the result that matters more than the raw accuracy number.

**Finding 3. The GBDT masked and semantic rows are a validity check, and they pass.**
Tree models see label encoded integers, so masking should be a no operation for them. All four deltas are inside noise, confirming the harness itself is not leaking anything between variants.

**Finding 4. Macro F1 is measuring the threshold, not the model.**
CatBoost sits at 0.56 to 0.57 macro F1 while holding the second best AUC. That gap is a fixed 0.5 decision threshold applied to a 16 percent positive rate, not weak ranking ability. AUC is the trustworthy column here. Any F1 based ranking in this repo should be read with that caveat.

**Finding 5. The timing column in the CSV is not a fair speed comparison and should not be quoted.**
The GBDT timer covers the final refit only and excludes the Optuna search that produced the parameters. The SAP RPT timer covers a fit that mainly stores the context, while the real compute happens inside `predict`, which is never timed. The 0.0008 second figure is an artefact. A proper comparison needs wall clock from raw table to scored test set, tuning included, for both sides.

---

## 7. Alternatives considered, and why this design was chosen

| Alternative | Why it was not used |
|---|---|
| Single train and test split | 1470 rows with 237 positives makes one split extremely noisy. Paired 5 fold CV lets every model see the identical folds, which is what makes the t tests valid. |
| Drop columns to test robustness | Deleting columns removes information, so a drop in accuracy is ambiguous. Masking holds information constant and removes only meaning, which isolates the variable of interest. |
| Compare against untuned GBDT defaults | That is the comparison most foundation model blog posts make, and it flatters the foundation model. Optuna is used here so the baseline is at least trying. |
| Add a logistic regression or MLP baseline | Would broaden the picture, especially for calibration. Left as future work. |
| Use TabPFN v2 as the foundation model baseline | Reasonable and arguably the stronger classification baseline. SAP RPT 1 was chosen because the schema invariance claim is central to its enterprise pitch and is exactly what the masking experiment tests. |
| Deep tabular nets such as FT Transformer or SAINT | Need their own tuning budget to be fair, which puts them outside the scope of a lightweight baseline study. |

**Where this design is still weak, stated plainly:**

* `N_TRIALS = 3`. Three Optuna trials is not a tuned GBDT, it is a lightly perturbed one. The honest claim is that SAP RPT 1 beats a cheaply tuned GBDT at a fraction of the search cost, not that it beats a well tuned GBDT.
* One dataset, 1470 rows. Nothing here generalises to a million row table, where in context methods hit context limits and GBDTs get stronger.
* The Optuna study has no sampler seed, so the GBDT numbers carry extra run to run variance on top of fold variance.
* TabFM appears in the notebook imports but is not present in the results file. It is planned, not done.

---

## 8. Conclusion

For small enterprise tables, a tabular foundation model is now a sensible **first move**, not an exotic one. You get a strong number in one pass, with no search budget and no feature engineering, and on this dataset that number is better than what three cheaply tuned boosting libraries produce. The practical workflow this suggests:

1. Run the foundation model first to establish a baseline in minutes.
2. Compare that baseline against the business requirement.
3. Only invest in a custom model, feature engineering and a full tuning budget if the gap between the two justifies the cost.

The schema masking result is what makes step 1 realistic across many clients. A model that does not depend on column naming can be pointed at a new customer's table without a mapping exercise first.

---

## 9. Future work

The most interesting open direction is **regression quality in tabular foundation models**. SAP RPT 1 learns schema independent representations and holds up under column masking and anonymisation, which is exactly what enterprise deployment needs. Published benchmarks nevertheless report that some tabular foundation models still trail on regression relative to classification. Notably, that gap does **not** reproduce on this dataset: SAP RPT 1 posts its cleanest, most statistically significant wins on Task B. So the honest framing is that the reported weakness is benchmark dependent and does not show up on a small HR income table.

The research question that follows: can regression aware pretraining objectives, or richer numerical encodings, be added without giving up the schema invariance? Semantic context and fine grained numerical structure are learned by different parts of the model, and a training objective that pushes on continuous targets specifically is the obvious lever to test.

Concrete next steps for this repo:

* Run the TabFM pass and fill in the missing rows.
* Raise `N_TRIALS` to 50 or more and report a compute matched comparison.
* Log end to end wall clock including tuning, and time `predict` separately for the foundation model.
* Tune the classification threshold per fold, or report precision at fixed recall instead of macro F1.
* Add calibration curves and Brier score, since attrition scores are usually used for ranking and intervention budgets.
* Extend to two or three more tabular datasets before making any general claim.

---

## 10. Repository contents

```
.
├── README.md
├── Untitled8__1_.ipynb        # full experimental notebook
└── Final_result.csv           # 80 rows: 2 tasks x 2 variants x 4 models x 5 folds
```

Reproduce:

```bash
pip install git+https://github.com/SAP-samples/sap-rpt-1-oss.git
pip install xgboost lightgbm catboost optuna huggingface_hub
```

Then set `RUN_MODELS = ["xgboost", "lightgbm", "catboost", "sap_rpt"]` and run the notebook top to bottom. The TabFM pass needs a separate environment because of dependency conflicts; restart the runtime, install from the TabFM repo, and set `RUN_MODELS = ["tabfm"]`.

## 11. References

* SAP RPT 1 OSS: https://github.com/SAP-samples/sap-rpt-1-oss
* Google TabFM: https://github.com/google-research/tabfm
* TabFM announcement: https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/
* Dataset: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
* Optuna: https://optuna.org

---

Author: Vedant Tele
Repo: `https://github.com/<your-username>/tabular-fm-vs-gbdt-hr-attrition`
