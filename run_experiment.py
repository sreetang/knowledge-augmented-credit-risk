#!/usr/bin/env python3
"""
Knowledge-augmented vs data-only credit-decision models.

One script, end to end: download data -> prepare -> cross-validated experiment ->
2 tables, 2 figures, graphical abstract, supporting CSVs, run_info.json.

    python run_experiment.py                 # real run (needs internet on first use)
    python run_experiment.py --smoke         # code test on SYNTHETIC data (not results)

What is compared (identical folds for every model):
    LR   logistic regression, data-only
    RF   random forest, data-only
    GBM  gradient boosting, data-only                      (ablation cell A)
    GBM+K   ... + knowledge-derived features               (cell B)
    GBM+M   ... + monotone domain constraints              (cell C)
    GBM+K+M ... + both = knowledge-augmented               (cell D)

Metrics: ROC-AUC, Brier score, default rate among approved applicants at a 70% approval
rate, domain-rule violation rate (perturbation test), and permutation-importance
stability / concentration (ablation cells only).
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------------------
# Colours (validated categorical slots 1-2 of the reference palette) and plot style
# ----------------------------------------------------------------------------------
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdbd7"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "figure.dpi": 100,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
})

# ----------------------------------------------------------------------------------
# Dataset specification (domain knowledge lives here, fixed a priori, never tuned)
# ----------------------------------------------------------------------------------
# rules: column -> (risk_direction, step_kind, cap)
#   risk_direction: +1 = default risk should NOT fall when the feature rises; -1 = should NOT rise
#   step_kind: "sd"  -> perturbation = +1 training-fold standard deviation
#              "lvl" -> perturbation = +1 level (skipped when value >= cap, if cap given)
@dataclass
class Spec:
    name: str
    cat_cols: list
    rules: dict
    derived_rules: dict
    engineer: Callable = field(repr=False)


def eng_german(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["duration_x_amount"] = X["duration"] * X["credit_amount"]
    X["installment_load"] = X["installment_commitment"] * X["credit_amount"]
    X["financial_buffer"] = X["checking_lvl"].fillna(0) + X["savings_lvl"].fillna(0)
    return X


def eng_taiwan(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    delay = X[[f"pay_delay_{k}" for k in range(1, 7)]]
    bills = X[[f"bill_amt_{k}" for k in range(1, 7)]]
    pays = X[[f"pay_amt_{k}" for k in range(1, 7)]]
    X["max_delay"] = delay.max(axis=1)
    X["n_delayed"] = (delay > 0).sum(axis=1).astype(float)
    X["utilization"] = bills.mean(axis=1) / X["limit_bal"].clip(lower=1)
    X["payment_ratio"] = pays.sum(axis=1) / (bills.clip(lower=0).sum(axis=1) + 1)
    return X


GERMAN = Spec(
    name="German Credit",
    cat_cols=["credit_history", "purpose", "employment", "other_parties", "property_magnitude",
              "other_payment_plans", "housing", "job", "own_telephone"],
    rules={"duration": (+1, "sd", None), "credit_amount": (+1, "sd", None),
           "installment_commitment": (+1, "lvl", 4), "checking_lvl": (-1, "lvl", 2),
           "savings_lvl": (-1, "lvl", 3)},
    derived_rules={"duration_x_amount": +1, "installment_load": +1, "financial_buffer": -1},
    engineer=eng_german,
)
TAIWAN = Spec(
    name="Taiwan Default",
    cat_cols=["education"],
    rules={"limit_bal": (-1, "sd", None),
           **{f"pay_delay_{k}": (+1, "lvl", None) for k in range(1, 7)},
           **{f"pay_amt_{k}": (-1, "sd", None) for k in range(1, 7)}},
    derived_rules={"max_delay": +1, "n_delayed": +1, "utilization": +1, "payment_ratio": -1},
    engineer=eng_taiwan,
)

# ----------------------------------------------------------------------------------
# Data loading + preparation (with sanity checks: fail loudly on anything unexpected)
# ----------------------------------------------------------------------------------
GERMAN_COLS = ["checking_status", "duration", "credit_history", "purpose", "credit_amount",
               "savings_status", "employment", "installment_commitment", "personal_status",
               "other_parties", "residence_since", "property_magnitude", "age",
               "other_payment_plans", "housing", "existing_credits", "job", "num_dependents",
               "own_telephone", "foreign_worker", "class"]
TAIWAN_COLS = (["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
               + ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
               + [f"BILL_AMT{k}" for k in range(1, 7)] + [f"PAY_AMT{k}" for k in range(1, 7)])
CHK = {"<0": 0, "0<=X<200": 1, ">=200": 2}
SAV = {"<100": 0, "100<=X<500": 1, "500<=X<1000": 2, ">=1000": 3}


def load_german_raw(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "german_credit_raw.csv"
    if path.exists():
        return pd.read_csv(path)
    from sklearn.datasets import fetch_openml
    print("  downloading German Credit (OpenML 'credit-g') ...")
    df = fetch_openml("credit-g", version=1, as_frame=True).frame.copy()
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_taiwan_raw(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "taiwan_default_raw.csv"
    if path.exists():
        return pd.read_csv(path)
    print("  downloading Taiwan credit-card default (UCI id 350) ...")
    try:
        from ucimlrepo import fetch_ucirepo
        d = fetch_ucirepo(id=350)
        X = d.data.features.copy()
        y = d.data.targets.iloc[:, 0].astype(int)
    except Exception as e:  # noqa: BLE001
        print(f"  ucimlrepo failed ({e!r}); trying OpenML id 42477 ...")
        from sklearn.datasets import fetch_openml
        b = fetch_openml(data_id=42477, as_frame=True)
        X, y = b.data.copy(), pd.to_numeric(b.target).astype(int)
    if X.shape[1] != 23:
        raise RuntimeError(
            f"Taiwan data has {X.shape[1]} feature columns, expected 23. Download "
            "default_of_credit_card_clients.xls from UCI (dataset 350), save it as "
            f"{path} with the 23 canonical columns {TAIWAN_COLS} plus a 'default' column.")
    X.columns = TAIWAN_COLS  # the dataset's column order is fixed and documented
    X["default"] = y.values
    data_dir.mkdir(parents=True, exist_ok=True)
    X.to_csv(path, index=False)
    return X


def prepare_german(raw: pd.DataFrame):
    df = raw.copy()
    missing = [c for c in GERMAN_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"German data is missing columns {missing}")
    chk, sav = df["checking_status"].astype(str), df["savings_status"].astype(str)
    bad_chk = set(chk.unique()) - set(CHK) - {"no checking"}
    bad_sav = set(sav.unique()) - set(SAV) - {"no known savings"}
    if bad_chk or bad_sav:
        raise ValueError(f"Unexpected category labels: checking={bad_chk}, savings={bad_sav}")
    out = pd.DataFrame(index=df.index)
    for c in ["duration", "credit_amount", "installment_commitment", "residence_since", "age",
              "existing_credits", "num_dependents"]:
        out[c] = pd.to_numeric(df[c]).astype(float)
    out["checking_lvl"] = chk.map(CHK).astype(float)
    out["savings_lvl"] = sav.map(SAV).astype(float)
    out["checking_none"] = (chk == "no checking").astype(float)
    out["savings_none"] = (sav == "no known savings").astype(float)
    for c in GERMAN.cat_cols:  # personal_status (sex) and foreign_worker are deliberately dropped
        out[c] = df[c].astype(str)
    y = (df["class"].astype(str) == "bad").astype(int).values
    return out.reset_index(drop=True), y


def prepare_taiwan(raw: pd.DataFrame):
    df = raw.copy()
    out = pd.DataFrame(index=df.index)
    out["limit_bal"] = df["LIMIT_BAL"].astype(float)
    out["age"] = df["AGE"].astype(float)
    edu = df["EDUCATION"].astype(int)
    out["education"] = edu.where(edu.isin([1, 2, 3, 4]), 4).astype(str)  # 0,5,6 -> 'other'
    for k, c in enumerate(["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"], 1):
        out[f"pay_delay_{k}"] = df[c].astype(float).clip(lower=0)  # months overdue (0 = none)
    for k in range(1, 7):
        out[f"bill_amt_{k}"] = df[f"BILL_AMT{k}"].astype(float)
    for k in range(1, 7):
        out[f"pay_amt_{k}"] = df[f"PAY_AMT{k}"].astype(float)
    y = df["default"].astype(int).values  # SEX and MARRIAGE are deliberately dropped
    return out.reset_index(drop=True), y


def check_dataset(name: str, X: pd.DataFrame, y: np.ndarray, smoke: bool) -> dict:
    info = {"dataset": name, "n": int(len(X)), "n_features": int(X.shape[1]),
            "default_rate": float(np.mean(y)), "missing_cells": int(X.isna().sum().sum())}
    if not smoke:  # known published shapes / prevalences
        if name == "German Credit":
            assert info["n"] == 1000 and abs(info["default_rate"] - 0.30) < 1e-9, info
        else:
            assert info["n"] == 30000 and abs(info["default_rate"] - 0.2212) < 0.002, info
    print(f"  {name}: n={info['n']}, features={info['n_features']}, "
          f"default rate={info['default_rate']:.4f}")
    return info


# ----------------------------------------------------------------------------------
# Synthetic stand-ins used ONLY by --smoke (to test the code path; never reported)
# ----------------------------------------------------------------------------------
def synth_german(n: int, rng: np.random.Generator) -> pd.DataFrame:
    ch = lambda opts, p=None: rng.choice(opts, n, p=p)  # noqa: E731
    dur = rng.integers(4, 73, n)
    amt = np.exp(rng.normal(7.8, 0.7, n)).round()
    inst = rng.integers(1, 5, n)
    chk = ch(["<0", "0<=X<200", ">=200", "no checking"], [.27, .27, .06, .40])
    sav = ch(["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"],
             [.6, .1, .06, .05, .19])
    z = (-2.2 + 0.03 * dur + 0.00015 * amt + 0.2 * inst + 0.7 * (chk == "<0")
         - 0.6 * (chk == "no checking") + 0.4 * (sav == "<100") + rng.normal(0, .6, n))
    bad = rng.random(n) < 1 / (1 + np.exp(-z))
    return pd.DataFrame({
        "checking_status": chk, "duration": dur, "credit_history": ch(["a", "b", "c", "d"]),
        "purpose": ch(["car", "tv", "edu", "biz"]), "credit_amount": amt, "savings_status": sav,
        "employment": ch(["<1", "1<=X<4", ">=7"]), "installment_commitment": inst,
        "personal_status": ch(["m", "f"]), "other_parties": ch(["none", "guarantor"]),
        "residence_since": rng.integers(1, 5, n), "property_magnitude": ch(["re", "car", "none"]),
        "age": rng.integers(19, 70, n), "other_payment_plans": ch(["none", "bank"]),
        "housing": ch(["own", "rent"]), "existing_credits": rng.integers(1, 4, n),
        "job": ch(["skilled", "unskilled"]), "num_dependents": rng.integers(1, 3, n),
        "own_telephone": ch(["yes", "none"]), "foreign_worker": ch(["yes", "no"]),
        "class": np.where(bad, "bad", "good")})


def synth_taiwan(n: int, rng: np.random.Generator) -> pd.DataFrame:
    limit = np.exp(rng.normal(11.5, 0.8, n)).round(-3)
    state = rng.random(n) < 0.25
    pay = np.where(state[:, None], rng.integers(-1, 5, (n, 6)), rng.integers(-2, 1, (n, 6)))
    bills = [np.exp(rng.normal(10, 1, n)).round() for _ in range(6)]
    pays = [(b * rng.uniform(0, .5, n)).round() for b in bills]
    z = (-2.0 + 0.7 * np.clip(pay, 0, None).max(axis=1) - 0.4 * np.log(limit / 1e5)
         + rng.normal(0, .5, n))
    d = rng.random(n) < 1 / (1 + np.exp(-z))
    df = pd.DataFrame({"LIMIT_BAL": limit, "SEX": rng.integers(1, 3, n),
                       "EDUCATION": rng.integers(0, 7, n), "MARRIAGE": rng.integers(0, 4, n),
                       "AGE": rng.integers(21, 70, n)})
    for k, c in enumerate(["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]):
        df[c] = pay[:, k]
    for k in range(6):
        df[f"BILL_AMT{k + 1}"] = bills[k]
        df[f"PAY_AMT{k + 1}"] = pays[k]
    df["default"] = d.astype(int)
    return df


# ----------------------------------------------------------------------------------
# Model wrapper: optional knowledge features, optional monotone constraints
# ----------------------------------------------------------------------------------
class RiskModel(BaseEstimator, ClassifierMixin):
    def __init__(self, kind="hgb", spec=None, use_features=False, use_mono=False, seed=0):
        self.kind, self.spec = kind, spec
        self.use_features, self.use_mono, self.seed = use_features, use_mono, seed

    def _frame(self, X):
        return self.spec.engineer(X) if self.use_features else X.copy()

    def fit(self, X, y):
        Xf = self._frame(X)
        self.cat_ = [c for c in self.spec.cat_cols if c in Xf.columns]
        self.num_ = [c for c in Xf.columns if c not in self.cat_]
        if self.kind == "lr":
            num_tf = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
        elif self.kind == "rf":
            num_tf = SimpleImputer(strategy="median")
        else:
            num_tf = "passthrough"
        parts = [("num", num_tf, self.num_)]
        if self.cat_:
            parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_))
        self.pre_ = ColumnTransformer(parts, remainder="drop")
        Xt = self.pre_.fit_transform(Xf)
        if self.kind == "lr":
            self.est_ = LogisticRegression(max_iter=3000)
        elif self.kind == "rf":
            self.est_ = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, n_jobs=-1,
                                               random_state=self.seed)
        else:
            cst = None
            if self.use_mono:
                dirs = {c: d for c, (d, _, _) in self.spec.rules.items()}
                if self.use_features:
                    dirs.update(self.spec.derived_rules)
                cst = [dirs.get(c, 0) for c in self.num_] + [0] * (Xt.shape[1] - len(self.num_))
            self.est_ = HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.1, max_leaf_nodes=15, l2_regularization=1.0,
                early_stopping=False, monotonic_cst=cst, random_state=self.seed)
        self.est_.fit(Xt, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.est_.predict_proba(self.pre_.transform(self._frame(X)))


MODELS = [  # (label, short label, kwargs, uses_domain_knowledge)
    ("Logistic regression", "LR", dict(kind="lr"), False),
    ("Random forest", "RF", dict(kind="rf"), False),
    ("GBM (data-only)", "GBM", dict(kind="hgb"), False),
    ("GBM + knowledge features", "GBM+K", dict(kind="hgb", use_features=True), True),
    ("GBM + monotone constraints", "GBM+M", dict(kind="hgb", use_mono=True), True),
    ("GBM + both (knowledge-augmented)", "GBM+K+M", dict(kind="hgb", use_features=True, use_mono=True), True),
]
ABLATION = [m[0] for m in MODELS[2:]]
BASE_CELL = "GBM (data-only)"


# ----------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------
def approved_default_rate(y, p, approval=0.70):
    thr = np.quantile(p, approval)
    approved = p <= thr
    return float(np.mean(y[approved])) if approved.any() else float("nan")


def violation_rate(model, X, spec, steps):
    """Share of (applicant, rule) perturbations where predicted risk moves against domain knowledge."""
    base = model.predict_proba(X)[:, 1]
    total = viol = 0
    per_rule = {}
    for col, (direction, kind, cap) in spec.rules.items():
        v = X[col].astype(float)
        mask = v.notna()
        if kind == "sd":
            newv = v + steps[col]
        else:
            if cap is not None:
                mask &= v < cap
            newv = v + 1.0
        if not mask.any():
            continue
        Xp = X.copy()
        Xp[col] = newv.where(mask, v)
        delta = direction * (model.predict_proba(Xp)[:, 1] - base)
        n_viol = int(np.sum(delta[mask.values] < -1e-9))
        per_rule[col] = (n_viol, int(mask.sum()))
        viol += n_viol
        total += int(mask.sum())
    return (viol / total if total else float("nan")), per_rule


def permutation_importance(model, X, y, rng, n_repeats=3):
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    out = {}
    for c in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].values)
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        out[c] = float(np.mean(drops))
    return pd.Series(out)


def stability_metrics(imp: pd.DataFrame, top_k=5, top_share=3):
    """imp: folds x features. Mean pairwise Spearman, mean pairwise top-k Jaccard, top-3 share."""
    ranks = imp.rank(axis=1).values
    rho = [stats.spearmanr(ranks[i], ranks[j])[0] for i, j in combinations(range(len(ranks)), 2)]
    tops = [set(imp.iloc[i].nlargest(top_k).index) for i in range(len(imp))]
    jac = [len(a & b) / len(a | b) for a, b in combinations(tops, 2)]
    pos = imp.clip(lower=0)
    share = (pos.apply(lambda r: r.nlargest(top_share).sum() / r.sum() if r.sum() > 0 else np.nan,
                       axis=1))
    return {"spearman": float(np.nanmean(rho)), "jaccard_top5": float(np.mean(jac)),
            "top3_share": float(np.nanmean(share))}


def nb_test(d, n_folds):
    """Nadeau-Bengio corrected resampled paired t-test on per-fold differences d."""
    d = np.asarray(d, float)
    k = len(d)
    var = np.var(d, ddof=1)
    se = np.sqrt((1 / k + 1 / (n_folds - 1)) * var)
    if se == 0:
        return float(d.mean()), (float(d.mean()),) * 2, 1.0
    t = d.mean() / se
    p = 2 * stats.t.sf(abs(t), k - 1)
    half = stats.t.ppf(0.975, k - 1) * se
    return float(d.mean()), (float(d.mean() - half), float(d.mean() + half)), float(p)


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


# ----------------------------------------------------------------------------------
# The experiment
# ----------------------------------------------------------------------------------
def run_dataset(spec, X, y, args):
    rskf = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed)
    records, rule_rows, imp_store = [], [], {m: [] for m in ABLATION}
    t0 = time.time()
    for f, (tr, te) in enumerate(rskf.split(X, y)):
        Xtr, Xte, ytr, yte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True), y[tr], y[te]
        rng = np.random.default_rng(args.seed + f)
        sub = rng.choice(len(Xte), size=min(2000, len(Xte)), replace=False)  # shared by all models
        steps = {c: float(Xtr[c].std()) for c, (_, k, _) in spec.rules.items() if k == "sd"}
        for label, short, kw, _ in MODELS:
            m = RiskModel(spec=spec, seed=args.seed + f, **kw).fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
            vr, per_rule = violation_rate(m, Xte.iloc[sub].reset_index(drop=True), spec, steps)
            records.append(dict(dataset=spec.name, model=label, short=short, fold=f,
                                auc=roc_auc_score(yte, p), brier=brier_score_loss(yte, p),
                                approved_default=approved_default_rate(yte, p), violation=vr))
            for col, (nv, nt) in per_rule.items():
                rule_rows.append(dict(dataset=spec.name, model=label, fold=f, rule=col,
                                      violations=nv, tests=nt))
            if label in ABLATION:
                imp_store[label].append(permutation_importance(
                    m, Xte.iloc[sub].reset_index(drop=True), yte[sub], np.random.default_rng(args.seed + f)))
        if (f + 1) % 5 == 0 or f == 0:
            print(f"    [{spec.name}] fold {f + 1}/{args.folds * args.repeats}  ({time.time() - t0:.0f}s)")
    return pd.DataFrame(records), pd.DataFrame(rule_rows), imp_store


def build_tables(folds_df, imp_all, args):
    ds_names = list(folds_df["dataset"].unique())
    # ---------------- Table 1: main comparison ----------------
    rows = []
    for label, short, _, _ in MODELS:
        row = {"Model": label}
        for ds in ds_names:
            s = folds_df[(folds_df.dataset == ds) & (folds_df.model == label)]
            row[f"{ds}: AUC"] = f"{s.auc.mean():.3f} ± {s.auc.std():.3f}"
            row[f"{ds}: Approved default rate (%)"] = f"{100 * s.approved_default.mean():.2f} ± {100 * s.approved_default.std():.2f}"
            row[f"{ds}: Rule violations (%)"] = f"{100 * s.violation.mean():.2f} ± {100 * s.violation.std():.2f}"
        rows.append(row)
    t1 = pd.DataFrame(rows)

    # ---------------- Table 2: ablation + explanation behaviour ----------------
    rows, stab_summary = [], {}
    for ds in ds_names:
        base = folds_df[(folds_df.dataset == ds) & (folds_df.model == BASE_CELL)].sort_values("fold")
        pvals, tmp = [], []
        for label in ABLATION:
            s = folds_df[(folds_df.dataset == ds) & (folds_df.model == label)].sort_values("fold")
            imp = pd.DataFrame(imp_all[ds][label])
            st = stability_metrics(imp)
            stab_summary[(ds, label)] = st
            if label == BASE_CELL:
                tmp.append((label, None, None, s, st))
                continue
            diff, ci, p = nb_test(s.auc.values - base.auc.values, args.folds)
            pvals.append(p)
            tmp.append((label, diff, ci, s, st))
        adj = iter(holm(np.array(pvals)))
        for label, diff, ci, s, st in tmp:
            if diff is None:
                dcol, pcol = "—", "—"
            else:
                dcol = f"{diff:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"
                pcol = f"{next(adj):.3f}"
            rows.append({"Dataset": ds, "Variant": label, "ΔAUC vs data-only GBM [95% CI]": dcol,
                         "p (Holm)": pcol, "Rule violations (%)": f"{100 * s.violation.mean():.2f}",
                         "Rank stability (Spearman ρ)": f"{st['spearman']:.3f}",
                         "Top-5 overlap (Jaccard)": f"{st['jaccard_top5']:.3f}",
                         "Top-3 importance share": f"{st['top3_share']:.3f}"})
    return t1, pd.DataFrame(rows), stab_summary


def short_name(label):
    return {m[0]: m[1] for m in MODELS}[label]


def figure1(folds_df, out, tag):
    ds_names = list(folds_df["dataset"].unique())
    fig, axes = plt.subplots(1, len(ds_names), figsize=(7.2, 3.1), sharey=False)
    marks = {"LR": "o", "RF": "s", "GBM": "^", "GBM+K": "D", "GBM+M": "P", "GBM+K+M": "*"}
    for ax, ds in zip(np.atleast_1d(axes), ds_names):
        for label, short, _, uses_k in MODELS:
            s = folds_df[(folds_df.dataset == ds) & (folds_df.model == label)]
            x, y = 100 * s.violation.mean(), s.auc.mean()
            col = ORANGE if uses_k else BLUE
            ax.errorbar(x, y, yerr=s.auc.std(), fmt="none", ecolor=col, elinewidth=0.8, capsize=2, zorder=2)
            ax.scatter(x, y, s=70 if short == "GBM+K+M" else 42, marker=marks[short], color=col,
                       edgecolor="white", linewidth=0.8, zorder=3)
            ax.annotate(short, (x, y), xytext=(6, -11 if short == "GBM+M" else 5),
                        textcoords="offset points", fontsize=7, color=INK)
        ax.set_title(ds, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Domain-rule violation rate (%)  ← better")
        ax.set_ylabel("ROC-AUC (mean ± SD over folds)")
        xmax = 100 * folds_df[folds_df.dataset == ds].groupby("model").violation.mean().max()
        ax.set_xlim(-0.05 * max(xmax, 1), 1.18 * max(xmax, 1))
    h = [plt.Line2D([], [], marker="o", ls="", color=BLUE, label="data-only"),
         plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="uses domain knowledge")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
    if tag:
        fig.suptitle(tag, color="#c00", fontsize=9, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"figure1_accuracy_vs_rule_violations.{ext}")
    plt.close(fig)


def figure2(imp_all, out, tag, top_n=8):
    ds_names = list(imp_all.keys())
    fig, axes = plt.subplots(1, len(ds_names), figsize=(7.2, 3.4))
    base_l, aug_l = BASE_CELL, MODELS[-1][0]
    for ax, ds in zip(np.atleast_1d(axes), ds_names):
        b, a = pd.DataFrame(imp_all[ds][base_l]), pd.DataFrame(imp_all[ds][aug_l])
        feats = list(b.mean().nlargest(top_n).index)[::-1]
        y = np.arange(len(feats))
        ax.barh(y + 0.19, b.mean()[feats], height=0.36, color=BLUE, label="GBM (data-only)",
                xerr=b.std()[feats], error_kw=dict(elinewidth=0.7, capsize=1.5, ecolor=INK2))
        ax.barh(y - 0.19, a.mean()[feats], height=0.36, color=ORANGE, label="GBM+K+M (knowledge-augmented)",
                xerr=a.std()[feats], error_kw=dict(elinewidth=0.7, capsize=1.5, ecolor=INK2))
        ax.set_yticks(y)
        ax.set_yticklabels(feats, fontsize=7)
        ax.set_title(ds, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Permutation importance (AUC drop, mean ± SD over folds)")
        ax.grid(axis="y", visible=False)
    axes_flat = np.atleast_1d(axes)
    h, l = axes_flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
    if tag:
        fig.suptitle(tag, color="#c00", fontsize=9, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"figure2_feature_attribution.{ext}")
    plt.close(fig)


def graphical_abstract(folds_df, t2_numbers, out, tag):
    ds_names = list(folds_df["dataset"].unique())
    fig = plt.figure(figsize=(7.2, 2.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)

    def box(x, y, w, h, text, fc, ec=INK2, fs=7, bold=False):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", linespacing=1.35)

    ax.text(1, 37.5, "Does domain knowledge make credit-decision AI more trustworthy?", fontsize=9,
            fontweight="bold", color=INK, va="center")
    box(1, 22, 22, 10, "Credit data\nGerman Credit · Taiwan default", "#eef2f7", fs=6.5)
    box(1, 7, 22, 10, "Domain rules\nrisk must not fall as delays,\nloan size or amount owed rise", "#fdf2e3", fs=6.2)
    box(28, 22, 27, 10, "Data-only GBM", "#e7edf5", ec=BLUE, bold=True)
    box(28, 7, 27, 10, "Knowledge-augmented GBM\n(rule features +\nmonotone constraints)", "#fdf2e3", ec=ORANGE, fs=6.2, bold=True)
    for (x1, y1, x2, y2) in [(23, 27, 28, 27), (23, 12, 28, 12), (23, 24, 28, 14)]:
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9))
    # results panel
    axb = fig.add_axes([0.71, 0.2, 0.27, 0.50])
    xs = np.arange(len(ds_names))
    va = [100 * folds_df[(folds_df.dataset == d) & (folds_df.model == BASE_CELL)].violation.mean() for d in ds_names]
    vd = [100 * folds_df[(folds_df.dataset == d) & (folds_df.model == MODELS[-1][0])].violation.mean() for d in ds_names]
    axb.bar(xs - 0.19, va, 0.36, color=BLUE, label="data-only")
    axb.bar(xs + 0.19, vd, 0.36, color=ORANGE, label="knowledge-augmented")
    for x, a, d_ in zip(xs, va, vd):
        axb.text(x - 0.19, a, f"{a:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK)
        axb.text(x + 0.19, d_, f"{d_:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    axb.set_xticks(xs)
    axb.set_xticklabels([f"{d}\nΔAUC {t2_numbers[d]:+.3f}" for d in ds_names], fontsize=6.3)
    axb.set_ylabel("Rule violations (%)", fontsize=6.5)
    axb.tick_params(labelsize=6.3)
    axb.set_ylim(0, max(va + vd + [1]) * 1.5)
    axb.legend(frameon=False, fontsize=6, loc="upper right")
    axb.set_title("Result", fontsize=7, color=INK, loc="left")
    ax.annotate("", (63, 20), (56, 20), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9))
    if tag:
        ax.text(50, 1.5, tag, color="#c00", fontsize=7, ha="center")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"graphical_abstract.{ext}", bbox_inches=None)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="synthetic-data code test, NOT results")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or ("smoke_out" if args.smoke else "results"))
    out.mkdir(parents=True, exist_ok=True)
    tag = "SMOKE TEST - SYNTHETIC DATA - NOT RESULTS" if args.smoke else ""
    t_start = time.time()
    np.random.seed(args.seed)

    print("== Loading data ==")
    if args.smoke:
        rng = np.random.default_rng(0)
        raws = {"German Credit": synth_german(500, rng), "Taiwan Default": synth_taiwan(3000, rng)}
    else:
        raws = {"German Credit": load_german_raw(Path(args.data_dir)),
                "Taiwan Default": load_taiwan_raw(Path(args.data_dir))}
    prepared, data_info = {}, []
    for spec, prep in ((GERMAN, prepare_german), (TAIWAN, prepare_taiwan)):
        X, y = prep(raws[spec.name])
        data_info.append(check_dataset(spec.name, X, y, args.smoke))
        prepared[spec.name] = (spec, X, y)

    print("== Running experiment ==")
    all_folds, all_rules, imp_all = [], [], {}
    for name, (spec, X, y) in prepared.items():
        fdf, rdf, imp_store = run_dataset(spec, X, y, args)
        all_folds.append(fdf)
        all_rules.append(rdf)
        imp_all[name] = imp_store
    folds_df, rules_df = pd.concat(all_folds), pd.concat(all_rules)

    print("== Tables and figures ==")
    t1, t2, stab = build_tables(folds_df, imp_all, args)
    t1.to_csv(out / "table1_main_comparison.csv", index=False)
    t2.to_csv(out / "table2_ablation_explainability.csv", index=False)
    folds_df.to_csv(out / "fold_level_metrics.csv", index=False)
    rr = (rules_df.groupby(["dataset", "model", "rule"])[["violations", "tests"]].sum().reset_index())
    rr["violation_rate_pct"] = 100 * rr.violations / rr.tests
    rr.to_csv(out / "violations_by_rule.csv", index=False)
    imp_rows = [dict(dataset=ds, model=m, fold=i, feature=f, importance=v)
                for ds, d in imp_all.items() for m, lst in d.items()
                for i, s in enumerate(lst) for f, v in s.items()]
    pd.DataFrame(imp_rows).to_csv(out / "importance_by_fold.csv", index=False)

    delta_auc = {}
    for ds in prepared:
        a = folds_df[(folds_df.dataset == ds) & (folds_df.model == BASE_CELL)].sort_values("fold").auc.values
        d = folds_df[(folds_df.dataset == ds) & (folds_df.model == MODELS[-1][0])].sort_values("fold").auc.values
        delta_auc[ds] = float(np.mean(d - a))
    figure1(folds_df, out, tag)
    figure2(imp_all, out, tag)
    graphical_abstract(folds_df, delta_auc, out, tag)

    summary = {
        "smoke_test": args.smoke,
        "settings": {"folds": args.folds, "repeats": args.repeats, "seed": args.seed,
                     "n_evaluations_per_model": args.folds * args.repeats},
        "data": data_info,
        "table1": t1.to_dict(orient="records"),
        "table2": t2.to_dict(orient="records"),
        "delta_auc_knowledge_augmented_vs_data_only": delta_auc,
        "constrained_models_max_violation_pct": {
            f"{ds}|{m}": float(100 * folds_df[(folds_df.dataset == ds) & (folds_df.model == m)].violation.max())
            for ds in prepared for m in ("GBM + monotone constraints", MODELS[-1][0])},
        "environment": {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__,
                        "platform": platform.platform(), "nvidia_smi_found": bool(shutil.which("nvidia-smi")),
                        "cpu_count": __import__("os").cpu_count()},
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    (out / "run_info.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\n== TABLE 1 ==\n", t1.to_string(index=False))
    print("\n== TABLE 2 ==\n", t2.to_string(index=False))
    print(f"\nDone in {summary['runtime_seconds']}s. Outputs in: {out.resolve()}")


if __name__ == "__main__":
    main()
