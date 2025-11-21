#!/usr/bin/env python3
"""Evaluate Imputation Accuracy for exported _wexp CSV files.

Given an export file that contains both imputed scoreboard fields and
corresponding expected values (columns prefixed with 'expected_'), this script
computes per-field accuracy by comparing equality of the imputed values to the
expected ones.

Usage:
    python scripts/evaluate_imputation_accuracy.py --input exports/imputed_data_game_6_20251113_160543_wexp.csv

By default, outputs a small Markdown-like summary. Optionally writes detailed
per-row comparison to a new CSV when --output is supplied.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import pandas as pd


ESSENTIAL_FIELDS = ["game_time", "period", "play_clock"]
IMPUTED_FLAG_TEMPLATE = "imputed_{field}"


def compare_fields(df: pd.DataFrame) -> Dict[str, float]:
    """Compute accuracy for essential fields that have expected counterparts."""
    scores: Dict[str, float] = {}

    for field in ESSENTIAL_FIELDS:
        expected_col = f"expected_{field}"
        flag_col = IMPUTED_FLAG_TEMPLATE.format(field=field)

        if field not in df.columns or expected_col not in df.columns or flag_col not in df.columns:
            continue

        mask = df[flag_col] == True
        if mask.sum() == 0:
            scores[field] = float("nan")
            continue

        comparisons = df.loc[mask, field] == df.loc[mask, expected_col]
        total = comparisons.notna().sum()
        if total == 0:
            scores[field] = float("nan")
        else:
            correct = comparisons.sum()
            scores[field] = correct / total

        df.loc[mask, f"accuracy_{field}"] = comparisons.astype(float)

    return scores


def build_breakdown(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for field in ESSENTIAL_FIELDS:
        expected_col = f"expected_{field}"
        flag_col = IMPUTED_FLAG_TEMPLATE.format(field=field)

        if field not in df.columns or expected_col not in df.columns or flag_col not in df.columns:
            continue

        mask = df[flag_col] == True
        if mask.sum() == 0:
            continue

        comparisons = df.loc[mask, field] == df.loc[mask, expected_col]
        valid_mask = comparisons.notna()
        total = valid_mask.sum()
        correct = comparisons[valid_mask].sum()
        incorrect = total - correct

        summary[field] = {
            "total": total,
            "correct": int(correct),
            "incorrect": int(incorrect),
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate imputation accuracy for _wexp export files")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the _wexp CSV file inside exports/"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to save detailed comparison (CSV)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        exports_path = Path("exports") / args.input
        if exports_path.exists():
            input_path = exports_path

    if not input_path.exists():
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)

    df = pd.read_csv(input_path)
    scores = compare_fields(df)
    breakdown = build_breakdown(df)

    print("\n=== Imputation Accuracy Summary ===")
    for field in ESSENTIAL_FIELDS:
        if field in scores:
            accuracy = scores[field]
            if pd.isna(accuracy):
                print(f"{field:>12}: n/a (no comparable rows)")
            else:
                print(f"{field:>12}: {accuracy:.2%}")
        else:
            print(f"{field:>12}: n/a (missing columns)")

    print("\n--- Breakdown ---")
    for field in ESSENTIAL_FIELDS:
        if field not in breakdown:
            print(f"{field:>12}: n/a")
            continue
        stats = breakdown[field]
        print(
            f"{field:>12}: total={stats['total']}, "
            f"correct={stats['correct']}, incorrect={stats['incorrect']}"
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nDetailed comparisons written to {output_path}")


if __name__ == "__main__":
    main()
