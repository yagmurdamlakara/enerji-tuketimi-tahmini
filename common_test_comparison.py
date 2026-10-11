"""Compare saved model predictions and exact-lag baselines on shared targets."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from clean_data_rerun import CONFIG, MODELS, SEEDS
from reference_baselines import (
    METHOD_LAGS,
    TARGET,
    add_exact_references,
    load_and_prepare_hourly,
)

LABELS = {
    "persistence": "Persistence",
    "daily_24h": "24 saat önce",
    "weekly_168h": "168 saat önce",
}


def read_predictions(path):
    """Reject ambiguous timestamps and invalid energy values."""
    frame = pd.read_csv(path)
    required = {"target_timestamp", "actual_kwh"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"Missing columns in {path.name}: {required - set(frame.columns)}"
        )
    index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("target_timestamp"), errors="raise")
    )
    if (
        index.hasnans
        or index.has_duplicates
        or index.tz is not None
        or not index.is_monotonic_increasing
    ):
        raise ValueError(f"Invalid, duplicate or unordered timestamps in {path.name}.")
    frame.index = index
    for column in ["actual_kwh", "prediction_kwh"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if not np.isfinite(frame[column]).all():
                raise ValueError(f"Non-finite {column} in {path.name}.")
    return frame


def metrics(actual, prediction):
    return dict(
        MAE_kWh=float(mean_absolute_error(actual, prediction)),
        RMSE_kWh=float(np.sqrt(mean_squared_error(actual, prediction))),
        R2=float(r2_score(actual, prediction)),
    )


def evaluate(hourly, results_dir):
    """Calculate each seed's metrics before taking the five-seed mean."""
    summary = json.loads((results_dir / "run_summary.json").read_text())
    if summary.get("status") != "complete" or summary.get("completed_runs") != 20:
        raise ValueError("A completed 20-run training archive is required.")
    for key, expected in CONFIG.items():
        if summary.get(key) != expected:
            raise ValueError(f"Training protocol differs for {key}.")
    if summary.get("minimum_valid_minutes_per_variable") != 60:
        raise ValueError("Training archive does not use the accepted 60-minute rule.")
    target_path = results_dir / "model_test_targets.csv"
    target_hash = hashlib.sha256(target_path.read_bytes()).hexdigest()
    if target_hash != summary.get("test_targets_sha256"):
        raise ValueError(
            "The saved test-target file hash does not match the training run."
        )
    targets = read_predictions(target_path)
    test_count = next(
        row["samples"] for row in summary["split_counts"] if row["split"] == "test"
    )
    if len(targets) != test_count:
        raise ValueError("Test-target count differs from the training audit.")
    actual = hourly[TARGET].reindex(targets.index)
    if actual.isna().any() or not np.allclose(
        actual, targets.actual_kwh, rtol=1e-7, atol=1e-8
    ):
        raise ValueError("Saved target energy differs from the cleaned raw data.")
    references = add_exact_references(hourly, targets.index).set_index(
        "target_timestamp"
    )
    if not references.continuous_24h_input_and_target.all():
        raise ValueError("Saved model targets include a discontinuous input window.")
    expected_files = {
        f"{model}_seed_{seed}_predictions.csv" for model in MODELS for seed in SEEDS
    }
    present_files = {path.name for path in results_dir.glob("*_predictions.csv")}
    if present_files != expected_files:
        raise ValueError(
            f"Unexpected prediction archive: missing={expected_files-present_files}, extra={present_files-expected_files}"
        )
    predictions = {}
    for model in MODELS:
        for seed in SEEDS:
            path = results_dir / f"{model}_seed_{seed}_predictions.csv"
            frame = read_predictions(path)
            if not {"prediction_kwh", "model", "seed"}.issubset(frame.columns):
                raise ValueError(f"Missing model prediction columns in {path.name}.")
            if not frame.index.equals(targets.index):
                raise ValueError(
                    f"Target timestamps differ in {path.name}; do not hide missing predictions with an intersection."
                )
            if not frame.model.eq(model).all() or not frame.seed.eq(seed).all():
                raise ValueError(f"Model/seed metadata differs in {path.name}.")
            if not np.allclose(frame.actual_kwh, actual, rtol=1e-7, atol=1e-8):
                raise ValueError(f"Actual energy differs in {path.name}.")
            predictions[(model, seed)] = frame.prediction_kwh
    # Model files are complete; only absent exact baseline history removes targets.
    available = references[
        [f"{method}_history_available" for method in METHOD_LAGS]
    ].all(axis=1)
    common = references.index[available]
    if len(common) < 2 or actual.loc[common].nunique() < 2:
        raise ValueError("Insufficient shared targets or constant target energy.")
    shared = pd.DataFrame({"actual_kwh": actual.loc[common]}, index=common)
    rows, seed_rows, variability = [], [], []
    for method in METHOD_LAGS:
        column = f"{method}_prediction_kwh"
        shared[column] = references.loc[common, column]
        rows.append(
            dict(
                method=LABELS[method],
                evaluated_targets=len(common),
                **metrics(shared.actual_kwh, shared[column]),
            )
        )
    for model in MODELS:
        model_rows = []
        for seed in SEEDS:
            column = f"{model}_seed_{seed}_prediction_kwh"
            shared[column] = predictions[(model, seed)].loc[common]
            row = dict(
                model=model,
                seed=seed,
                evaluated_targets=len(common),
                **metrics(shared.actual_kwh, shared[column]),
            )
            model_rows.append(row)
            seed_rows.append(row)
        scores = pd.DataFrame(model_rows)[["MAE_kWh", "RMSE_kWh", "R2"]]
        rows.append(
            dict(method=model, evaluated_targets=len(common), **scores.mean().to_dict())
        )
        variability.append(
            dict(
                model=model,
                **{f"{key}_std": value for key, value in scores.std(ddof=1).items()},
            )
        )
    audit = dict(
        original_test_targets=len(targets),
        common_test_targets=len(common),
        excluded_targets=len(targets) - len(common),
        missing_history_by_method={
            method: int((~references[f"{method}_history_available"]).sum())
            for method in METHOD_LAGS
        },
        first_common_target=str(common[0]),
        last_common_target=str(common[-1]),
        model_aggregation="Arithmetic mean of five per-seed metrics; predictions are not ensembled.",
        seeds=SEEDS,
        models=MODELS,
        completed_prediction_files=len(predictions),
        raw_sha256=summary["raw_sha256"],
        test_targets_sha256=target_hash,
        input_file_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(results_dir.glob("*_predictions.csv"))
        },
        status="complete",
    )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(seed_rows),
        pd.DataFrame(variability),
        shared,
        audit,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            "Choose an empty output directory; existing results are preserved."
        )
    summary = json.loads((args.results_dir / "run_summary.json").read_text())
    with args.data.open("rb") as raw_file:
        raw_hash = hashlib.file_digest(raw_file, "sha256").hexdigest()
    if raw_hash != summary.get("raw_sha256"):
        raise ValueError("Raw dataset hash differs from the completed training run.")
    hourly, preprocessing = load_and_prepare_hourly(args.data)
    table, seeds, variability, shared, audit = evaluate(hourly, args.results_dir)
    audit["preprocessing"] = preprocessing
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        args.output_dir / "common_test_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    seeds.to_csv(args.output_dir / "common_per_seed_metrics.csv", index=False)
    variability.to_csv(args.output_dir / "common_model_variability.csv", index=False)
    shared.to_csv(args.output_dir / "common_test_predictions.csv", index=True)
    shared[["actual_kwh"]].to_csv(
        args.output_dir / "common_test_targets.csv", index=True
    )
    audit["common_targets_sha256"] = hashlib.sha256(
        (args.output_dir / "common_test_targets.csv").read_bytes()
    ).hexdigest()
    (args.output_dir / "comparison_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(table.to_string(index=False), flush=True)
    print(
        f"Shared targets: {audit['common_test_targets']}; excluded: {audit['excluded_targets']}"
    )


if __name__ == "__main__":
    main()
