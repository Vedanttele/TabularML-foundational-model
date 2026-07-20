# I Masked Every Column Name in an HR Dataset. The Foundation Model Did Not Care.

*A small controlled study on SAP RPT 1, XGBoost, LightGBM and CatBoost, run on 1470 rows of IBM HR attrition data.*

---

## The question I actually wanted to answer

Most enterprise tables are small. A few thousand rows, thirty columns, a business owner who needs an answer this week. For the last decade the standard reply has been "tune a gradient boosting model". Tabular foundation models now offer a different reply: hand the training rows to the model as context, get predictions in one forward pass, skip the training loop and the hyperparameter search entirely.

Fine. But the number I care about in an enterprise setting is not accuracy on a Kaggle file. It is this:

> Is the model good because it understands the data, or is it good because it read the string `OverTime` and knew what overtime means?

That distinction decides whether you can point the model at a new client system where the same column is called `Z_FLD_0042`. So I built the experiment around it.

---

## The setup

**Dataset.** IBM HR Analytics Employee Attrition, 1470 rows and 35 columns. Four columns come out immediately.

```python
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
df = df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'])
```

The first three are constant across every row. The fourth is an identifier. All four carry zero signal. That leaves 1470 rows and 31 columns, of which 7 are categorical.

**Two tasks.**

```python
# Task A: binary classification
y_a = df['Attrition'].map({'Yes': 1, 'No': 0})
X_a = df.drop(columns=['Attrition'])          # 30 features

# Task B: regression
y_b = df['MonthlyIncome']
X_b = df.drop(columns=['Attrition', 'MonthlyIncome', 'JobLevel'])   # 28 features
```

`JobLevel` is dropped from Task B on purpose. It correlates about 0.95 with monthly income, so leaving it in turns regression into a lookup table and tells you nothing about any model.

Task A is imbalanced: 237 Yes against 1233 No, roughly 16 percent positive. With 5 stratified folds that is about 47 positives in each test fold. Small. I will come back to why that matters.

---

## The core trick: semantic masking

This is the part that makes the study more than a leaderboard.

I rename every column to `col_00`, `col_01` and so on, and replace every categorical string with a deterministic hash token. Numeric values are left byte identical.

```python
def mask_semantics(X: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    X_masked = X.copy()
    np.random.seed(seed)

    # 1. mask column names
    new_cols = [f"col_{i:02d}" for i in range(len(X.columns))]
    X_masked.rename(columns=dict(zip(X.columns, new_cols)), inplace=True)

    # 2. mask categorical values, keep numerics untouched
    for col in X_masked.columns:
        if not pd.api.types.is_numeric_dtype(X_masked[col]):
            unique_levels = sorted(X_masked[col].unique().astype(str))
            permuted = np.random.permutation(len(unique_levels))
            level_map = {
                lvl: f"cat_{hashlib.md5(str(permuted[i]).encode()).hexdigest()[:4]}"
                for i, lvl in enumerate(unique_levels)
            }
            X_masked[col] = X_masked[col].astype(str).map(level_map)

    return X_masked
```

![Original table beside masked table. Column names and category strings are replaced, numeric values are unchanged.](figures/fig1_masking.svg)

Two properties I needed:

* **Information preserving.** Nothing is deleted. Same rows, same values, same cardinalities. Only meaning is destroyed. If I had dropped columns instead, a drop in accuracy would be ambiguous: did the model lose meaning, or did it lose information? Masking removes exactly one variable.
* **Deterministic.** Seeded, so the masked table is identical across folds and models.

The output looks like this:

```
Masked columns: ['col_00', 'col_01', 'col_02', 'col_03', 'col_04']
Masked values:  cat_c81e, cat_eccb, cat_a87f
```

Now the prediction: a tree model should not notice this at all, because it only ever sees label encoded integers. A model leaning on language priors should degrade. That difference is the measurement.

---

## The algorithm

```
for task in [TaskA classification, TaskB regression]:
    for variant in [semantic, masked]:
        X = raw table, or mask_semantics(raw table)
        folds = StratifiedKFold(5) if classification else KFold(5)

        for fold in folds:
            for model in [xgboost, lightgbm, catboost, sap_rpt]:

                if GBDT:
                    label encode categoricals using train fold levels only
                    Optuna search with 3 fold inner CV on the train fold
                    refit best params on the full train fold
                else:
                    SAP_RPT_OSS(max_context_size=2048, bagging=2)
                    fit in context, no gradient updates

                predict on test fold
                record AUC and macro F1, or R2 and RMSE
```

Seed 42 everywhere, and **every model sees the identical folds**.

![One of five folds, with four training blocks and one test block, feeding the same split into XGBoost, LightGBM, CatBoost and SAP RPT 1.](figures/fig2_folds.svg)

That picture is doing real work. Because all four models are scored on exactly the same rows, fold to fold difficulty cancels out. A fold that happens to contain three unusual leavers makes every model look worse together, so the *difference* between two models on that fold is a cleaner signal than either score alone. That is what makes a paired t test legitimate later. If each model had drawn its own random split, I would only be allowed to compare noisy averages.

The two model wrappers, side by side, show how different the two paradigms are:

```python
def fit_sap(task, X_tr, y_tr, X_te, y_te):
    cls = SAP_RPT_OSS_Classifier if task == "clf" else SAP_RPT_OSS_Regressor
    model = cls(max_context_size=2048, bagging=2)
    model.fit(X_tr, y_tr)              # no gradient updates, context is stored
    preds = model.predict(X_te)
    ...
```

```python
def fit_gbdt(name, task, X_tr, y_tr, X_te, y_te):
    X_tr_enc, X_te_enc = encode_for_gbdt(X_tr, X_te)

    def objective(trial):
        m = get_gbdt_model(name, task, trial)
        return cross_val_score(m, X_tr_enc, y_tr, cv=3, scoring=metric).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)
    best_model = get_gbdt_model(name, task, study.best_params)
    best_model.fit(X_tr_enc, y_tr)
    ...
```

Note the label encoding is fitted on the train fold only, with unseen test levels mapped to minus one. Small detail, but fitting the encoder on the full table is one of the most common quiet leaks in tabular work.

```python
for c in X_tr.columns:
    if not pd.api.types.is_numeric_dtype(X_tr[c]):
        le = {v: i for i, v in enumerate(sorted(X_tr[c].astype(str).unique()))}
        X_tr[c] = X_tr[c].astype(str).map(le).fillna(-1).astype(np.int64)
        X_te[c] = X_te[c].astype(str).map(le).fillna(-1).astype(np.int64)
```

---

## Results

80 rows in total: 2 tasks times 2 variants times 4 models times 5 folds. All figures below are fold means.

![Grouped bar charts of attrition AUC and income R squared for four models, real versus masked column names.](figures/fig3_results.svg)

Two things to read here, and the second one matters more. First, the teal blue bar for SAP RPT 1 is the tallest in both panels. Second, and this is the actual finding, **the blue and orange bars for SAP RPT 1 are the same height**. Masking every column name and category label moves it almost not at all.

One caveat before anyone screenshots this: the y axis does not start at zero. Task A is drawn from 0.78 to 0.85 and Task B from 0.82 to 0.91, because the whole interesting range sits inside a few hundredths. That zoom makes the gaps look larger than they are. The p values below are the honest version of the same comparison.

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

### Is any of this real, or is it fold noise?

Because every model saw the same folds, I can test this properly instead of eyeballing the table.

```python
from scipy import stats

d = df[(df.task == 'TaskA') & (df.variant == 'semantic')] \
      .pivot(index='fold', columns='model', values='auc')

t, p = stats.ttest_rel(d['sap_rpt'], d['lightgbm'])
```

| Comparison | Task A AUC | Task B R2 |
|---|---|---|
| vs XGBoost (semantic) | +0.032, p = 0.097 | +0.017, p = 0.013 |
| vs LightGBM (semantic) | +0.031, p = 0.016 | +0.040, p = 0.012 |
| vs CatBoost (semantic) | +0.024, p = 0.007 | +0.030, p = 0.006 |
| vs XGBoost (masked) | +0.028, p = 0.052 | +0.027, p = 0.003 |
| vs LightGBM (masked) | +0.029, p = 0.029 | +0.034, p = 0.024 |
| vs CatBoost (masked) | +0.015, p = 0.098 | +0.021, p = 0.005 |

Every regression comparison clears p = 0.03. Classification is positive in all six comparisons but only three clear p = 0.05, which is exactly what you should expect with 5 folds and 47 positives per fold.

### The masking result

| Model | Task A AUC change | Task B R2 change |
|---|---|---|
| **sap_rpt** | **0.003 (p = 0.44)** | **0.000 (p = 0.74)** |
| xgboost | 0.001 (p = 0.91) | 0.010 (p = 0.17) |
| lightgbm | 0.001 (p = 0.92) | 0.006 (p = 0.47) |
| catboost | 0.006 (p = 0.37) | 0.009 (p = 0.11) |

---

## What I take from this

**1. The masking result is the headline, not the accuracy result.**
Strip every column name and every category label, and SAP RPT 1 moves 0.003 AUC and 0.000 R2. Both sit well inside fold noise. The model is not reading English off the headers. For anyone deploying across many client schemas, that matters more than a two point accuracy gain.

**2. The GBDT rows are a built in control, and they pass.**
Trees see label encoded integers, so masking should be a no operation for them. All four deltas land inside noise. That confirms the harness is not leaking anything between the two variants. If CatBoost had suddenly jumped under masking, the experiment would be broken and I would know.

**3. Regression is where the win is cleanest, which surprised me.**
Published benchmarks report tabular foundation models trailing on regression more than on classification. On this dataset the opposite happens: Task B gives the tightest p values and the largest consistent margin. My honest reading is that the published gap is benchmark dependent, and it does not reproduce on a small HR income table. That is a more interesting result than agreeing with the paper.

**4. Macro F1 here measures a threshold, not a model.**
CatBoost holds the second best AUC and the worst macro F1 by eleven points. That is a fixed 0.5 decision cut applied to a 16 percent positive rate, not weak ranking ability. AUC is the column to trust. I am reporting F1 for completeness, not for ranking.

---

## Where this study is weak

I would rather state this myself than have someone find it.

* **Three Optuna trials is not a tuned GBDT.** `N_TRIALS = 3` means the baseline is lightly perturbed, not optimised. The defensible claim is that SAP RPT 1 beats a cheaply tuned GBDT at a fraction of the search cost. Not that it beats a well tuned one. Raising this to 50 trials and reporting a compute matched comparison is the next job.
* **The timing column in my results file is not usable.** The GBDT timer covers the final refit and excludes the whole Optuna search. The SAP RPT timer covers a fit that mostly stores context, while the real compute sits inside `predict`, which I never timed. So the sub millisecond figure is an artefact. A real comparison needs wall clock from raw table to scored test set, tuning included, on both sides.
* **One dataset, 1470 rows.** Nothing here generalises to a million row table, where in context methods run into context limits and boosting gets stronger.
* **The Optuna study has no sampler seed**, so GBDT numbers carry run to run variance on top of fold variance.
* **Google TabFM is in the notebook imports but not in the results.** It needs a separate environment because of dependency conflicts. Planned, not done.

---

## What I would use this for

Not "replace your GBDTs". Something narrower and more useful:

1. Run the foundation model first. You get a strong baseline in minutes, with no feature engineering and no search budget.
2. Hold that baseline against the actual business requirement.
3. Only spend on a custom model, feature work and a full tuning budget if the gap between the two justifies the cost.

The masking result is what makes step 1 realistic at scale. A model that does not depend on column naming can be pointed at a new customer table without a schema mapping exercise first.

---

## The open question I find most interesting

SAP RPT 1 learns schema independent representations and holds up under full anonymisation. That is exactly the property enterprise deployment needs. The open question is whether regression quality can be pushed further without giving that property up.

Semantic context and fine grained numerical structure are plausibly learned by different parts of the model. A pretraining or fine tuning objective aimed specifically at continuous targets, or a richer numerical encoding scheme, is the obvious lever to test. The experiment writes itself: add a regression aware objective, then rerun the masking ablation and check whether schema invariance survives.

---

## Reproduce it

```bash
pip install git+https://github.com/SAP-samples/sap-rpt-1-oss.git
pip install xgboost lightgbm catboost optuna huggingface_hub
```

Set `RUN_MODELS = ["xgboost", "lightgbm", "catboost", "sap_rpt"]` and run the notebook top to bottom. The TabFM pass needs its own environment: restart the runtime, install from the TabFM repo, then set `RUN_MODELS = ["tabfm"]`.

**Code, results and figures:** `https://github.com/<your-username>/tabular-fm-vs-gbdt-hr-attrition`

The three figures in this post live in `figures/` as SVG, so they stay sharp and stay in version control alongside the numbers that produced them.

* SAP RPT 1 OSS: https://github.com/SAP-samples/sap-rpt-1-oss
* Google TabFM: https://github.com/google-research/tabfm
* Dataset: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

---

*Vedant Tele. People Analytics and Reporting, Allianz Global, Munich.*
