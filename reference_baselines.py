#!/usr/bin/env python3
"""Evaluate time-aligned persistence baselines without training any model.

Use --test-targets CSV for model-aligned evaluation, or explicitly select
--cleaned-ratio for a new 70/15/15 split that is not comparable to old scores.
CSV columns: target_timestamp, actual_kwh (optional, checked if present).
Exact timestamp lookups; no interpolation and no extra continuity exclusions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RAW_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
MEAN_COLUMNS = RAW_COLUMNS[:4]
SUM_COLUMNS = RAW_COLUMNS[4:]
TARGET = "Energy_kWh"
MIN_VALID_MINUTES = 60
LOOKBACK_HOURS = 24
EXPECTED_HOURLY_ROWS = 34_085
EXPECTED_CONTINUOUS_WINDOWS = 32_449
EXPECTED_CONTINUOUS_CANDIDATES = 34_061

METHOD_LAGS = {
    "persistence": 1,
    "daily_24h": 24,
    "weekly_168h": 168,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("household_power_consumption.txt"),
        help="Path to the original semicolon-delimited UCI text file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reference_baseline_results"),
        help="Directory for the metrics, predictions, and audit summary.",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--test-targets", type=Path,
                           help="CSV of the actual model evaluation targets.")
    selection.add_argument("--cleaned-ratio", action="store_true",
                           help="Explicitly use a NEW cleaned-data 70/15/15 split.")
    return parser.parse_args()


def load_and_prepare_hourly(data_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read raw minute data and apply the accepted hourly preprocessing."""
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found: {data_path.resolve()}\n"
            "Download household_power_consumption.txt from the UCI dataset "
            "and pass its path with --data."
        )

    raw = pd.read_csv(
        data_path,
        sep=";",
        na_values=["?"],
        low_memory=False,
        dtype={"Date": "string", "Time": "string"},
    )
    required_columns = {"Date", "Time", *RAW_COLUMNS}
    missing_columns = sorted(required_columns.difference(raw.columns))
    if missing_columns:
        raise ValueError(f"Required raw-data columns are missing: {missing_columns}")

    timestamps = pd.to_datetime(
        raw["Date"].str.strip() + " " + raw["Time"].str.strip(),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    if timestamps.isna().any():
        examples = raw.loc[timestamps.isna(), ["Date", "Time"]].head(5)
        raise ValueError(
            "Could not parse one or more raw timestamps. Examples:\n"
            f"{examples.to_string(index=False)}"
        )

    minute_data = raw[RAW_COLUMNS].apply(pd.to_numeric, errors="coerce")
    minute_data.index = pd.DatetimeIndex(timestamps, name="datetime")
    minute_data = minute_data.sort_index()

    if minute_data.index.has_duplicates:
        duplicate_count = int(minute_data.index.duplicated().sum())
        raise ValueError(f"Found {duplicate_count} duplicate minute timestamps.")

    # Count actual, nonmissing minute observations before any aggregation.
    valid_minutes = minute_data.resample("h").count()
    keep_hour = valid_minutes.ge(MIN_VALID_MINUTES).all(axis=1)

    hourly_all = minute_data[MEAN_COLUMNS].resample("h").mean()
    hourly_all[SUM_COLUMNS] = minute_data[SUM_COLUMNS].resample("h").sum(min_count=1)

    hourly = hourly_all.loc[keep_hour].copy()
    hourly[TARGET] = hourly["Global_active_power"] * 1.0

    if hourly.isna().any().any():
        raise AssertionError("Missing values remain after the 60-minute hourly filter.")

    # This is a diagnostic assertion, not a row cut or a correction.
    if len(hourly) != EXPECTED_HOURLY_ROWS:
        print(
            "NOTICE: hourly row count differs from the 5 October reference "
            f"({len(hourly):,} vs {EXPECTED_HOURLY_ROWS:,}). "
            "The run will use the observed data without forcing the old count."
        )

    audit = {
        "raw_rows": int(len(raw)),
        "raw_start": str(minute_data.index.min()),
        "raw_end": str(minute_data.index.max()),
        "raw_missing_values_by_column": {
            str(k): int(v) for k, v in minute_data.isna().sum().items()
        },
        "hourly_bins_before_filter": int(len(valid_minutes)),
        "hourly_rows_retained": int(len(hourly)),
        "hourly_rows_removed": int((~keep_hour).sum()),
        "hourly_missing_cells_after_filter": int(hourly.isna().sum().sum()),
        "minimum_valid_minutes_per_variable": MIN_VALID_MINUTES,
        "hourly_rule": {
            "mean": MEAN_COLUMNS,
            "sum": SUM_COLUMNS,
            "interpolation": "none",
        },
        "energy_definition": (
            "Energy_kWh = hourly mean Global_active_power (kW) * 1 hour"
        ),
    }
    return hourly, audit


def add_exact_references(
    hourly: pd.DataFrame, target_timestamps: pd.DatetimeIndex
) -> pd.DataFrame:
    """Join actual historical targets at exact elapsed-hour timestamps."""
    target_timestamps = pd.DatetimeIndex(target_timestamps, name="target_timestamp")
    actual = hourly[TARGET]
    records = pd.DataFrame(index=target_timestamps)
    records.index.name = "target_timestamp"
    records["actual_kwh"] = actual.reindex(target_timestamps).to_numpy()

    # A valid 24-hour input plus target requires 25 consecutive timestamps.
    continuity_columns = []
    for step in range(LOOKBACK_HOURS + 1):
        expected_time = target_timestamps - pd.Timedelta(hours=LOOKBACK_HOURS - step)
        observed = hourly[TARGET].reindex(expected_time).notna().to_numpy()
        continuity_columns.append(observed)

    continuous = np.column_stack(continuity_columns).all(axis=1)
    records["continuous_24h_input_and_target"] = continuous

    for method, lag_hours in METHOD_LAGS.items():
        reference_timestamps = target_timestamps - pd.Timedelta(hours=lag_hours)
        records[f"{method}_reference_timestamp"] = reference_timestamps
        records[f"{method}_prediction_kwh"] = (
            actual.reindex(reference_timestamps).to_numpy()
        )
        records[f"{method}_history_available"] = records[
            f"{method}_prediction_kwh"
        ].notna().to_numpy()

    return records.reset_index()


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.all(y_true == y_true[0]):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def summarize_metrics(
    predictions: pd.DataFrame, candidate_count: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score supplied targets when history exists; continuity is diagnostic only."""
    continuous = predictions["continuous_24h_input_and_target"]
    continuity_valid_count = int(continuous.sum())
    continuity_invalid_count = int(candidate_count - continuity_valid_count)

    if predictions["actual_kwh"].isna().any():
        raise ValueError("Some supplied test targets are absent from cleaned data; "
                         "do not silently alter the model evaluation set.")
    rows = []
    evaluated_by_method: dict[str, np.ndarray] = {}
    for method in METHOD_LAGS:
        prediction_column = f"{method}_prediction_kwh"
        has_reference = predictions[f"{method}_history_available"]
        evaluated = has_reference
        y_true = predictions.loc[evaluated, "actual_kwh"].to_numpy(dtype=float)
        y_pred = predictions.loc[evaluated, prediction_column].to_numpy(dtype=float)
        evaluated_by_method[method] = evaluated.to_numpy()

        missing_history = int((~has_reference).sum())
        rows.append(
            {
                "method": method,
                "lag_hours": METHOD_LAGS[method],
                "test_targets_total": candidate_count,
                "continuity_valid_targets": continuity_valid_count,
                "continuity_invalid_targets": continuity_invalid_count,
                "evaluated_targets": int(evaluated.sum()),
                "missing_history_targets": missing_history,
                "MAE_kWh": (
                    float(mean_absolute_error(y_true, y_pred))
                    if len(y_true)
                    else float("nan")
                ),
                "RMSE_kWh": (
                    float(np.sqrt(mean_squared_error(y_true, y_pred)))
                    if len(y_true)
                    else float("nan")
                ),
                "R2": safe_r2(y_true, y_pred),
            }
        )

    common = np.logical_and.reduce(list(evaluated_by_method.values()))
    common_true = predictions.loc[common, "actual_kwh"].to_numpy(dtype=float)
    common_rows = []
    for method in METHOD_LAGS:
        common_pred = predictions.loc[
            common, f"{method}_prediction_kwh"
        ].to_numpy(dtype=float)
        common_rows.append(
            {
                "method": method,
                "common_evaluated_targets": int(common.sum()),
                "MAE_kWh": float(mean_absolute_error(common_true, common_pred)) if len(common_true) else float("nan"),
                "RMSE_kWh": float(np.sqrt(mean_squared_error(common_true, common_pred))) if len(common_true) else float("nan"),
                "R2": safe_r2(common_true, common_pred),
            }
        )

    summary = {
        "continuity_valid_test_targets": continuity_valid_count,
        "continuity_invalid_test_targets": continuity_invalid_count,
        "common_evaluated_targets": int(common.sum()),
    }
    return pd.DataFrame(rows), {**summary, "common_metrics": common_rows}


def verify_missing_hour_example() -> None:
    """Guard against treating rows around a missing hour as adjacent in time."""
    index = pd.date_range("2020-01-01 13:00", periods=5, freq="h")
    toy = pd.DataFrame({TARGET: [1.0, 2.0, 4.0, 5.0]}, index=index.delete(2))
    target_time = pd.DatetimeIndex([pd.Timestamp("2020-01-01 17:00")])
    check = add_exact_references(toy, target_time).iloc[0]

    assert check["persistence_prediction_kwh"] == 4.0
    assert pd.isna(check["daily_24h_prediction_kwh"])
    assert not bool(check["continuous_24h_input_and_target"])
    missing = add_exact_references(toy, pd.DatetimeIndex([index[3]])).iloc[0]
    assert pd.isna(missing["persistence_prediction_kwh"])
    print("Missing-hour check: exact lag lookup and continuity diagnostic passed.")


def main() -> None:
    args = parse_args()
    verify_missing_hour_example()

    hourly, audit = load_and_prepare_hourly(args.data)

    # Match the notebook's current chronological row split on the cleaned table.
    row_count = len(hourly)
    train_end = int(row_count * 0.70)
    validation_end = int(row_count * 0.85)
    test_df = hourly.iloc[validation_end:].copy()
    if test_df.empty:
        raise ValueError("The chronological test split is empty.")

    if args.test_targets is not None:
        supplied = pd.read_csv(args.test_targets)
        if "target_timestamp" not in supplied or supplied.empty:
            raise ValueError("CSV must contain a nonempty target_timestamp column.")
        target_timestamps = pd.DatetimeIndex(
            pd.to_datetime(supplied["target_timestamp"], errors="raise"),
            name="target_timestamp")
        if target_timestamps.hasnans or target_timestamps.has_duplicates:
            raise ValueError("Test target timestamps must be valid and unique.")
        if not target_timestamps.is_monotonic_increasing or target_timestamps.tz is not None:
            raise ValueError("Use chronological, timezone-naive dataset timestamps.")
        actual = hourly[TARGET].reindex(target_timestamps)
        if actual.isna().any():
            raise ValueError("Some model targets are not present in cleaned data.")
        if "actual_kwh" in supplied:
            expected = pd.to_numeric(supplied["actual_kwh"], errors="raise").to_numpy()
            if not np.isfinite(expected).all() or not np.allclose(
                    actual.to_numpy(), expected, rtol=1e-5, atol=1e-6):
                raise ValueError("Model true targets differ from cleaned target values.")
        test_df = hourly.loc[target_timestamps].copy()
        split_description = "explicit model target timestamps from CSV"
        comparison_note = "Aligned to supplied CSV; caller must supply the actual model evaluation targets."
    else:
        target_timestamps = pd.DatetimeIndex(test_df.index, name="target_timestamp")
        split_description = "NEW chronological cleaned-data 70/15/15 row split"
        comparison_note = "NOT verified against previous model targets; do not compare to old model scores."
    print(comparison_note)
    predictions = add_exact_references(hourly, target_timestamps)
    metrics, metric_audit = summarize_metrics(predictions, len(test_df))
    common_metrics = pd.DataFrame(metric_audit["common_metrics"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "reference_baseline_metrics.csv"
    common_metrics_path = args.output_dir / "reference_baseline_common_metrics.csv"
    predictions_path = args.output_dir / "reference_baseline_predictions.csv"
    summary_path = args.output_dir / "reference_baseline_run_summary.json"

    metrics.to_csv(metrics_path, index=False, float_format="%.10g")
    common_metrics.to_csv(common_metrics_path, index=False, float_format="%.10g")
    predictions.to_csv(predictions_path, index=False, float_format="%.10g")

    audit.update(
        {
            "split_rule": split_description,
            "test_targets_file": str(args.test_targets) if args.test_targets else None,
            "train_rows": train_end if args.cleaned_ratio else None,
            "validation_rows": validation_end - train_end if args.cleaned_ratio else None,
            "test_targets_total": int(len(test_df)),
            "test_start": str(test_df.index.min()),
            "test_end": str(test_df.index.max()),
            "reference_timestamp_rule": (
                "target timestamp minus exact elapsed hours; exact timestamp join"
            ),
            "historical_lookup_scope": "all retained hourly data before target time",
            "model_comparison_status": comparison_note,
            "continuity_filter_applied": False,
            "continuity_audit": metric_audit,
            "expected_reference_checks": {
                "retained_hours_34085": len(hourly) == EXPECTED_HOURLY_ROWS,
                "candidate_windows_34061": (
                    len(hourly) - LOOKBACK_HOURS == EXPECTED_CONTINUOUS_CANDIDATES
                ),
                "continuous_windows_32449": (
                    int(
                        (
                            np.diff(
                                hourly.index.to_numpy()[
                                    np.arange(len(hourly) - LOOKBACK_HOURS)[:, None]
                                    + np.arange(LOOKBACK_HOURS + 1)[None, :]
                                ],
                                axis=1,
                            )
                            == np.timedelta64(1, "h")
                        )
                        .all(axis=1)
                        .sum()
                    )
                    == EXPECTED_CONTINUOUS_WINDOWS
                ),
            },
            "outputs": {
                "metrics_csv": str(metrics_path),
                "common_metrics_csv": str(common_metrics_path),
                "predictions_csv": str(predictions_path),
                "summary_json": str(summary_path),
            },
        }
    )
    summary_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nPreprocessing and test split audit")
    print(f"Retained hours : {len(hourly):,}")
    print(f"Test targets   : {len(test_df):,}")
    print(f"Test period    : {test_df.index.min()} → {test_df.index.max()}")
    print(
        "Continuous test targets: "
        f"{metric_audit['continuity_valid_test_targets']:,}"
    )
    print(
        "Discontinuous test targets: "
        f"{metric_audit['continuity_invalid_test_targets']:,}"
    )
    print("\nBaseline metrics (kWh for MAE/RMSE; R² unitless)")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nCommon-target metrics")
    print(common_metrics.to_string(index=False))
    print(f"\nSaved results under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

