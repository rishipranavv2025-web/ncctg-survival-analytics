import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import DATA_PATH, DATA_URL, FEATURES, MODEL_PATH, RESULTS_DIR, DISPLAY_NAMES
from survival_models import ExponentialPHModel


RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5
COX_PENALIZER = 0.01
PARAMETRIC_PENALIZER = 0.01

for directory in [DATA_PATH.parent, MODEL_PATH, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def load_data():
    expected = {
        "time", "status", "age", "sex", "ph.ecog",
        "ph.karno", "pat.karno", "meal.cal", "wt.loss"
    }

    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH)
            df = df.drop(columns=["rownames", "Unnamed: 0"], errors="ignore")
            if expected.issubset(df.columns):
                print(f"Loaded dataset from {DATA_PATH}")
                return df
            print("Local dataset is missing required columns; trying download.")
        except Exception as exc:
            print(f"Could not read local dataset: {exc}")

    urls = list(dict.fromkeys([
        DATA_URL,
        "https://raw.githubusercontent.com/CamDavidsonPilon/lifelines/master/lifelines/datasets/lung.csv"
    ]))

    for url in urls:
        try:
            print(f"Downloading dataset from {url}")
            df = pd.read_csv(url)
            df = df.drop(columns=["rownames", "Unnamed: 0"], errors="ignore")
            if expected.issubset(df.columns):
                df.to_csv(DATA_PATH, index=False)
                print(f"Saved dataset to {DATA_PATH}")
                return df
            print("Downloaded file did not contain all required columns.")
        except Exception as exc:
            print(f"Download failed: {exc}")

    raise RuntimeError(
        "Could not load a valid NCCTG lung dataset. Check data/lung.csv and your internet connection."
    )


def prepare_data(raw):
    df = raw.copy()
    required = ["time", "status"] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    status = pd.to_numeric(df["status"], errors="coerce")
    status_values = set(status.dropna().unique())

    if status_values.issubset({1, 2}) and 2 in status_values:
        # Original NCCTG/R survival dataset: 1=censored, 2=death.
        df["event"] = (status == 2).astype(int)
    elif status_values.issubset({0, 1}):
        # Converted coding: 0=censored, 1=death.
        df["event"] = (status == 1).astype(int)
    else:
        raise ValueError(f"Unexpected status values: {sorted(status_values)}")

    # Original NCCTG coding: 1=male, 2=female. Some prepared copies may
    # already use model coding: 0=male, 1=female. Detect the representation.
    sex = pd.to_numeric(df["sex"], errors="coerce")
    sex_values = set(sex.dropna().unique())
    if sex_values.issubset({1, 2}) and 2 in sex_values:
        df["sex"] = sex.map({1: 0, 2: 1})
    elif sex_values.issubset({0, 1}):
        df["sex"] = sex
    else:
        raise ValueError(f"Unexpected sex values: {sorted(sex_values)}")

    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["time", "event"]).copy()
    df = df[df["time"] > 0].copy()
    df["event"] = df["event"].astype(int)

    if df.empty:
        raise ValueError("No valid observations remain after outcome cleaning.")
    if df["event"].nunique() < 2:
        raise ValueError("The dataset must contain both event and censored observations.")

    print(f"Patients: {len(df)}")
    print(f"Observed events/deaths: {int(df['event'].sum())}")
    print(f"Censored observations: {int((df['event'] == 0).sum())}")
    return df


def impute_from_training(train_df, other_df):
    train_df = train_df.copy()
    other_df = other_df.copy()
    medians = {}

    for column in FEATURES:
        median = train_df[column].median()
        if pd.isna(median):
            raise ValueError(f"Cannot impute {column}: training values are all missing.")
        medians[column] = float(median)
        train_df[column] = train_df[column].fillna(median)
        other_df[column] = other_df[column].fillna(median)

    return train_df, other_df, medians


def fit_models(train_df):
    cox = CoxPHFitter(penalizer=COX_PENALIZER)
    cox.fit(train_df[["time", "event"] + FEATURES],
            duration_col="time", event_col="event")

    weibull = WeibullAFTFitter(penalizer=PARAMETRIC_PENALIZER)
    weibull.fit(train_df[["time", "event"] + FEATURES],
                duration_col="time", event_col="event")

    exp_model = ExponentialPHModel().fit(
        train_df[FEATURES].to_numpy(dtype=float),
        train_df["time"].to_numpy(dtype=float),
        train_df["event"].to_numpy(dtype=int),
        feature_names=FEATURES
    )
    return cox, weibull, exp_model


def model_c_indices(models, eval_df):
    cox, weibull, exp_model = models
    X = eval_df[FEATURES]
    durations = eval_df["time"].to_numpy(dtype=float)
    events = eval_df["event"].to_numpy(dtype=int)

    # lifelines concordance_index expects higher predicted values to mean longer survival.
    scores = {
        "Cox PH": -cox.predict_partial_hazard(X).to_numpy().ravel(),
        "Weibull AFT": weibull.predict_median(X).to_numpy().ravel(),
        "Exponential PH": -exp_model.predict_hazard(X.to_numpy(dtype=float)).ravel()
    }

    output = {}
    for name, score in scores.items():
        score = np.asarray(score, dtype=float)
        valid = np.isfinite(score) & np.isfinite(durations)
        try:
            if valid.sum() < 2:
                output[name] = np.nan
            else:
                output[name] = float(concordance_index(
                    durations[valid], score[valid], events[valid]
                ))
        except (ZeroDivisionError, ValueError):
            output[name] = np.nan
    return output


def cross_validate_training_data(train_df):
    """Choose the model using only the training partition, never the final test set."""
    splitter = StratifiedKFold(
        n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    fold_scores = {name: [] for name in ["Cox PH", "Weibull AFT", "Exponential PH"]}

    for fold, (fit_idx, val_idx) in enumerate(
        splitter.split(train_df[FEATURES], train_df["event"]), start=1
    ):
        fold_train_raw = train_df.iloc[fit_idx].copy()
        fold_val_raw = train_df.iloc[val_idx].copy()
        fold_train, fold_val, _ = impute_from_training(fold_train_raw, fold_val_raw)

        models = fit_models(fold_train)
        scores = model_c_indices(models, fold_val)

        print(f"CV fold {fold}: " + ", ".join(
            f"{name}={scores[name]:.4f}" if np.isfinite(scores[name]) else f"{name}=NA"
            for name in fold_scores
        ))

        for name, score in scores.items():
            if np.isfinite(score):
                fold_scores[name].append(float(score))

    means = {
        name: (float(np.mean(values)) if values else None)
        for name, values in fold_scores.items()
    }
    if not any(value is not None for value in means.values()):
        raise RuntimeError("Cross-validation did not produce any valid C-index scores.")

    selected = max(
        (name for name, value in means.items() if value is not None),
        key=lambda name: means[name]
    )
    return means, fold_scores, selected


def save_km_plot(df):
    km = KaplanMeierFitter()
    km.fit(df["time"], event_observed=df["event"], label="NCCTG Lung Cancer")

    km_table = pd.DataFrame({
        "time": km.survival_function_.index,
        "survival": km.survival_function_.iloc[:, 0].values
    })
    km_table.to_csv(RESULTS_DIR / "kaplan_meier.csv", index=False)

    plt.figure(figsize=(9, 5.5))
    km.plot_survival_function(ci_show=True)
    plt.title("Kaplan-Meier Survival Curve")
    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "kaplan_meier.png", dpi=180)
    plt.close()
    return km


def save_model_comparison_plot(metrics_df):
    plot_df = metrics_df.dropna(subset=["c_index"])
    if plot_df.empty:
        return

    plt.figure(figsize=(9, 5.5))
    bars = plt.bar(plot_df["Model"], plot_df["c_index"])
    plt.ylim(0, 1)
    plt.ylabel("Held-out test C-index")
    plt.title("Survival Model Comparison")
    plt.grid(axis="y", alpha=0.2)

    for bar, value in zip(bars, plot_df["c_index"]):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.015,
                 f"{value:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=180)
    plt.close()


def clean_metric(value):
    if value is None:
        return None
    try:
        return float(value) if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def main():
    print("=" * 65)
    print("LUNG CANCER SURVIVAL ANALYSIS")
    print("=" * 65)

    df = prepare_data(load_data())
    model_df = df[["time", "event"] + FEATURES].copy()

    # This split is reserved for the final, one-time test evaluation.
    train_raw, test_raw = train_test_split(
        model_df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=model_df["event"]
    )
    train_raw = train_raw.copy()
    test_raw = test_raw.copy()

    print(f"Training partition: {len(train_raw)}")
    print(f"Held-out test partition: {len(test_raw)}")

    # Model choice is made only by cross-validation within the training partition.
    cv_means, cv_fold_scores, selected_model = cross_validate_training_data(train_raw)
    print("\nMean training-only cross-validation C-index:")
    for name, value in cv_means.items():
        print(f"  {name}: {value if value is not None else 'NA'}")
    print(f"Selected using training-only CV: {selected_model}")

    # Fit final candidate models on the complete training partition.
    train_df, test_df, medians = impute_from_training(train_raw, test_raw)
    models = fit_models(train_df)
    cox, weibull, exp_model = models

    # Final test scores are reported for evaluation only; they do not select the model.
    test_c_indices = model_c_indices(models, test_df)

    # Cox AIC_partial_ is a partial-likelihood AIC, unlike full-likelihood AIC
    # for the parametric models. They are reported as model-specific statistics,
    # not used for selection or ranked against each other.
    metrics = {
        "Kaplan-Meier": {
            "c_index": None, "aic": None, "log_likelihood": None,
            "cv_c_index": None
        },
        "Cox PH": {
            "c_index": clean_metric(test_c_indices["Cox PH"]),
            "aic": clean_metric(cox.AIC_partial_),
            "log_likelihood": clean_metric(cox.log_likelihood_),
            "cv_c_index": clean_metric(cv_means["Cox PH"])
        },
        "Weibull AFT": {
            "c_index": clean_metric(test_c_indices["Weibull AFT"]),
            "aic": clean_metric(weibull.AIC_),
            "log_likelihood": clean_metric(weibull.log_likelihood_),
            "cv_c_index": clean_metric(cv_means["Weibull AFT"])
        },
        "Exponential PH": {
            "c_index": clean_metric(test_c_indices["Exponential PH"]),
            "aic": clean_metric(exp_model.aic_),
            "log_likelihood": clean_metric(exp_model.log_likelihood_),
            "cv_c_index": clean_metric(cv_means["Exponential PH"])
        }
    }

    print("\nHeld-out test C-index (not used for model selection):")
    for name in ["Cox PH", "Weibull AFT", "Exponential PH"]:
        print(f"  {name}: {metrics[name]['c_index']}")

    # Save final fitted models.
    joblib.dump(cox, MODEL_PATH / "cox.pkl")
    joblib.dump(weibull, MODEL_PATH / "weibull.pkl")
    joblib.dump(exp_model, MODEL_PATH / "exponential.pkl")

    km = save_km_plot(df)
    joblib.dump(km, MODEL_PATH / "kaplan_meier.pkl")

    metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
    metrics_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    save_model_comparison_plot(metrics_df)

    metadata = {
        "features": FEATURES,
        "display_names": dict(DISPLAY_NAMES),
        "medians": medians,
        "selected_model": selected_model,
        "selection_method": f"Highest mean {CV_SPLITS}-fold stratified CV C-index on training partition",
        "cv_mean_c_index": {k: clean_metric(v) for k, v in cv_means.items()},
        "cv_fold_c_index": {
            k: [clean_metric(v) for v in values] for k, values in cv_fold_scores.items()
        },
        "metrics": metrics,
        "rows": int(len(df)),
        "events": int(df["event"].sum()),
        "censored": int((df["event"] == 0).sum()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "imputation": "Median values fitted separately within each CV training fold; final medians fitted on final training partition only",
        "c_index_definition": "lifelines concordance_index; higher predicted score means longer survival. Risk scores are negated where needed.",
        "aic_note": "Cox AIC_partial_ uses partial likelihood and is not directly comparable to full-likelihood AIC values from parametric models."
    }
    (MODEL_PATH / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nTraining complete.")
    print(f"Patients: {len(df)} | Events: {int(df['event'].sum())} | Censored: {int((df['event'] == 0).sum())}")
    print(f"Train/test: {len(train_df)}/{len(test_df)}")
    print(f"Frozen model for the website: {selected_model}")
    print(f"Saved models and metadata to: {MODEL_PATH}")
    print("Run the website with: py app.py")


if __name__ == "__main__":
    main()