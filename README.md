# Knowledge-augmented vs data-only credit decision models

Code for the paper *"Toward Trustworthy AI Decision-Making: An Empirical Test of Knowledge-Augmented Classification"*.

Tests whether encoding credit-risk domain knowledge (rule-derived features and monotone constraints) in a
gradient-boosting model reduces domain-rule violations without hurting accuracy, on two public datasets
(German Credit, Taiwan credit-card default), against logistic-regression and random-forest baselines.

## Run

```bash
pip install -r requirements.txt
python run_experiment.py            # real run: downloads data on first use, ~15-40 min on CPU
python run_experiment.py --smoke    # code test on SYNTHETIC data (numbers are NOT results)
```

Options: `--folds 5 --repeats 3 --seed 42 --data-dir data --out results`. No GPU is needed.

## Files

| Path | Content |
|---|---|
| `run_experiment.py` | the whole pipeline (data, models, metrics, statistics, tables, figures) |
| `data/` | raw datasets cached on first run (created by the script) |
| `results/` | Table 1, Table 2, fold-level metrics, Figures 1-2, graphical abstract, `run_info.json` |

## Design (fixed before looking at results)

* Models: LR, RF, GBM (data-only), GBM+K (rule-derived features), GBM+M (monotone constraints), GBM+K+M.
* Evaluation: repeated stratified 5-fold CV (same folds for every model).
* Metrics: ROC-AUC, approved-default rate at 70% approval, rule-violation rate (perturbation test),
  permutation-importance rank stability and concentration.
* Statistics: Nadeau-Bengio corrected resampled paired t-test, Holm correction.
* Sex and marital status are excluded from all models.
