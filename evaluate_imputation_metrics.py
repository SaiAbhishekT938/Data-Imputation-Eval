#!/usr/bin/env python3
"""Evaluate Imputation Metrics from Matched Columns.

This script analyzes CSV files with _wmatch suffix that contain:
- imputed_<essential_field> (boolean): Whether imputation was performed
- matched_<essential_field> (boolean): Whether imputed value matched expected

For each essential field, it creates a confusion matrix and computes:
- Accuracy
- Precision
- Recall
- F1 Score

Confusion Matrix Structure:
    For imputed rows (imputed_<field> == TRUE):
        TP (True Positive):  matched=TRUE  (correct imputation)
        FP (False Positive): matched=FALSE (incorrect imputation)
    
    For non-imputed rows (imputed_<field> == FALSE):
        TN (True Negative):  No imputation needed (original was present)
        FN (False Negative): Not applicable in this context

Usage:
    python scripts/evaluate_imputation_metrics.py --input exports/imputed_data_game_6_20251113_215047_wexp_wmatch.csv
    python scripts/evaluate_imputation_metrics.py --input exports/file_wmatch.csv --output reports/metrics_report.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np


ESSENTIAL_FIELDS = ["game_time", "period", "play_clock"]
IMPUTED_FLAG_TEMPLATE = "imputed_{field}"
MATCHED_PREFIX = "matched_"


def build_confusion_matrix(df: pd.DataFrame, field: str) -> Dict[str, int]:
    """
    Build confusion matrix for a single field.
    
    Args:
        df: DataFrame with imputed_<field> and matched_<field> columns
        field: Field name (e.g., 'game_time')
        
    Returns:
        Dictionary with TP, FP, TN, FN counts
    """
    imputed_col = IMPUTED_FLAG_TEMPLATE.format(field=field)
    matched_col = f"{MATCHED_PREFIX}{field}"
    
    # Check required columns exist
    if imputed_col not in df.columns:
        return None
    if matched_col not in df.columns:
        return None
    
    # Convert boolean columns to proper boolean type
    imputed_series = df[imputed_col].astype(bool)
    matched_series = df[matched_col].astype(bool)
    
    # True Positive: imputed=TRUE, matched=TRUE (correct imputation)
    tp = ((imputed_series == True) & (matched_series == True)).sum()
    
    # False Positive: imputed=TRUE, matched=FALSE (incorrect imputation)
    fp = ((imputed_series == True) & (matched_series == False)).sum()
    
    # True Negative: imputed=FALSE (no imputation needed)
    # Note: We consider non-imputed rows as TN (original data was present)
    tn = (imputed_series == False).sum()
    
    # False Negative: Not applicable in imputation context
    # (We don't have a concept of "should have imputed but didn't")
    fn = 0
    
    return {
        'TP': int(tp),
        'FP': int(fp),
        'TN': int(tn),
        'FN': int(fn),
        'total': len(df)
    }


def calculate_metrics(confusion_matrix: Dict[str, int]) -> Dict[str, float]:
    """
    Calculate accuracy, precision, recall, and F1 score from confusion matrix.
    
    Args:
        confusion_matrix: Dictionary with TP, FP, TN, FN counts
        
    Returns:
        Dictionary with calculated metrics
    """
    tp = confusion_matrix['TP']
    fp = confusion_matrix['FP']
    tn = confusion_matrix['TN']
    fn = confusion_matrix['FN']
    
    # Accuracy: (TP + TN) / (TP + FP + TN + FN)
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    # Precision: TP / (TP + FP) - Of all imputations, how many were correct?
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall: TP / (TP + FN) - Of all cases that should be imputed correctly, how many were?
    # Note: In imputation context, FN=0, so recall = TP/TP = 1.0 if TP>0, else 0.0
    # However, a more meaningful recall would be: TP / (TP + FP) = Precision
    # Or: TP / (total imputed) = TP / (TP + FP)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Additional metrics
    # Imputation Rate: (TP + FP) / total (how many rows were imputed)
    imputation_rate = (tp + fp) / total if total > 0 else 0.0
    
    # Correct Imputation Rate: TP / (TP + FP) (same as precision)
    correct_imputation_rate = precision
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'imputation_rate': float(imputation_rate),
        'correct_imputation_rate': float(correct_imputation_rate)
    }


def print_confusion_matrix_table(field: str, cm: Dict[str, int], metrics: Dict[str, float]):
    """Print formatted confusion matrix table."""
    print(f"\n{'=' * 80}")
    print(f"CONFUSION MATRIX: {field.upper()}")
    print(f"{'=' * 80}")
    
    # Confusion matrix table
    print("\nConfusion Matrix:")
    print(" " * 20 + "Predicted")
    print(" " * 20 + "Imputed=TRUE" + " " * 10 + "Imputed=FALSE")
    print(" " * 10 + "Matched=TRUE" + f"  {cm['TP']:>6} (TP)" + f"  {cm['FN']:>6} (FN)")
    print(" " * 10 + "Matched=FALSE" + f" {cm['FP']:>6} (FP)" + f"  {cm['TN']:>6} (TN)")
    
    print(f"\nTotal Rows: {cm['total']}")
    print(f"  Imputed: {cm['TP'] + cm['FP']} (TP + FP)")
    print(f"  Not Imputed: {cm['TN']} (TN)")
    
    print(f"\nMetrics:")
    print(f"  Accuracy:              {metrics['accuracy']:>6.2%}")
    print(f"  Precision:             {metrics['precision']:>6.2%}  (TP / (TP + FP))")
    print(f"  Recall:                {metrics['recall']:>6.2%}  (TP / (TP + FN))")
    print(f"  F1 Score:              {metrics['f1_score']:>6.2%}")
    print(f"  Imputation Rate:       {metrics['imputation_rate']:>6.2%}  ((TP + FP) / Total)")
    print(f"  Correct Imputation:    {metrics['correct_imputation_rate']:>6.2%}  (TP / (TP + FP))")
    
    print(f"\nInterpretation:")
    print(f"  - {cm['TP']} imputations were correct (matched expected)")
    print(f"  - {cm['FP']} imputations were incorrect (did not match expected)")
    print(f"  - {cm['TN']} rows did not require imputation")
    if cm['TP'] + cm['FP'] > 0:
        success_rate = cm['TP'] / (cm['TP'] + cm['FP']) * 100
        print(f"  - Imputation Success Rate: {success_rate:.1f}% ({cm['TP']}/{cm['TP'] + cm['FP']})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate imputation metrics from matched columns",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to CSV file with _wmatch suffix (e.g., exports/file_wmatch.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to save JSON report"
    )
    args = parser.parse_args()
    
    # Resolve input path
    input_path = Path(args.input)
    if not input_path.exists():
        # Try in exports directory
        exports_path = Path("exports") / args.input
        if exports_path.exists():
            input_path = exports_path
        else:
            print(f"❌ Input file not found: {args.input}")
            sys.exit(1)
    
    print(f"📂 Reading file: {input_path}")
    
    # Read CSV file
    df = pd.read_csv(input_path)
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Check for required columns
    print("\n🔍 Checking for required columns...")
    missing_fields = []
    for field in ESSENTIAL_FIELDS:
        imputed_col = IMPUTED_FLAG_TEMPLATE.format(field=field)
        matched_col = f"{MATCHED_PREFIX}{field}"
        
        if imputed_col not in df.columns:
            print(f"⚠️  Missing: {imputed_col}")
            missing_fields.append(field)
        elif matched_col not in df.columns:
            print(f"⚠️  Missing: {matched_col}")
            missing_fields.append(field)
        else:
            print(f"✅ Found: {imputed_col}, {matched_col}")
    
    if missing_fields:
        print(f"\n❌ Missing required columns for fields: {missing_fields}")
        print("   Ensure file has _wmatch suffix and contains imputed_ and matched_ columns")
        sys.exit(1)
    
    # Build confusion matrices and calculate metrics
    print("\n📊 Building confusion matrices and calculating metrics...")
    results = {}
    
    for field in ESSENTIAL_FIELDS:
        cm = build_confusion_matrix(df, field)
        if cm is None:
            print(f"⚠️  Skipping {field}: missing required columns")
            continue
        
        metrics = calculate_metrics(cm)
        results[field] = {
            'confusion_matrix': cm,
            'metrics': metrics
        }
        
        # Print results
        print_confusion_matrix_table(field, cm, metrics)
    
    if not results:
        print("\n❌ No valid fields found for analysis")
        sys.exit(1)
    
    # Print overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    summary_data = []
    for field in ESSENTIAL_FIELDS:
        if field in results:
            m = results[field]['metrics']
            cm = results[field]['confusion_matrix']
            summary_data.append({
                'field': field,
                'accuracy': m['accuracy'],
                'precision': m['precision'],
                'recall': m['recall'],
                'f1_score': m['f1_score'],
                'imputation_rate': m['imputation_rate'],
                'correct_imputation_rate': m['correct_imputation_rate'],
                'total_imputed': cm['TP'] + cm['FP'],
                'correct_imputed': cm['TP'],
                'incorrect_imputed': cm['FP']
            })
    
    # Print summary table
    print(f"\n{'Field':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Imputed':<10} {'Correct':<10}")
    print("-" * 80)
    for data in summary_data:
        print(f"{data['field']:<15} {data['accuracy']:>10.2%} {data['precision']:>10.2%} "
              f"{data['recall']:>10.2%} {data['f1_score']:>10.2%} "
              f"{data['total_imputed']:>8} {data['correct_imputed']:>8}")
    
    # Calculate averages
    if summary_data:
        avg_accuracy = np.mean([d['accuracy'] for d in summary_data])
        avg_precision = np.mean([d['precision'] for d in summary_data])
        avg_recall = np.mean([d['recall'] for d in summary_data])
        avg_f1 = np.mean([d['f1_score'] for d in summary_data])
        
        print("-" * 80)
        print(f"{'Average':<15} {avg_accuracy:>10.2%} {avg_precision:>10.2%} "
              f"{avg_recall:>10.2%} {avg_f1:>10.2%}")
    
    # Save JSON report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj) if not np.isnan(obj) else None
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        json_results = convert_to_native(results)
        json_results['summary'] = summary_data
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n✅ Metrics report saved to: {output_path}")
    
    # Save CSV report with _report suffix
    csv_report_path = input_path.parent / f"{input_path.stem}_report.csv"
    csv_report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build structured CSV report with matrix format
    csv_lines = []
    
    # Header
    csv_lines.append("Imputation Metrics Report")
    csv_lines.append("")
    
    # Confusion Matrices Section
    csv_lines.append("CONFUSION MATRICES")
    csv_lines.append("")
    
    for field in ESSENTIAL_FIELDS:
        if field in results:
            cm = results[field]['confusion_matrix']
            
            # Field header
            csv_lines.append(f"Field: {field.upper()}")
            csv_lines.append("")
            
            # Confusion Matrix as actual matrix
            csv_lines.append("Actual\\Predicted,Imputed=TRUE,Imputed=FALSE")
            csv_lines.append(f"Matched=TRUE,{cm['TP']} (TP),{cm['FN']} (FN)")
            csv_lines.append(f"Matched=FALSE,{cm['FP']} (FP),{cm['TN']} (TN)")
            csv_lines.append("")
            csv_lines.append(f"Total Rows,{cm['total']}")
            csv_lines.append(f"Total Imputed,{cm['TP'] + cm['FP']}")
            csv_lines.append("")
    
    # Metrics Section
    csv_lines.append("METRICS")
    csv_lines.append("")
    csv_lines.append("Field,Accuracy,Precision,Recall,F1 Score,Imputation Rate,Correct Imputation Rate")
    
    for field in ESSENTIAL_FIELDS:
        if field in results:
            m = results[field]['metrics']
            csv_lines.append(f"{field},{m['accuracy']:.4f},{m['precision']:.4f},{m['recall']:.4f},"
                           f"{m['f1_score']:.4f},{m['imputation_rate']:.4f},{m['correct_imputation_rate']:.4f}")
    
    csv_lines.append("")
    
    # Summary Section
    if summary_data:
        csv_lines.append("SUMMARY")
        csv_lines.append("")
        csv_lines.append("Metric,Value")
        
        avg_accuracy = np.mean([d['accuracy'] for d in summary_data])
        avg_precision = np.mean([d['precision'] for d in summary_data])
        avg_recall = np.mean([d['recall'] for d in summary_data])
        avg_f1 = np.mean([d['f1_score'] for d in summary_data])
        
        csv_lines.append(f"Average Accuracy,{avg_accuracy:.4f}")
        csv_lines.append(f"Average Precision,{avg_precision:.4f}")
        csv_lines.append(f"Average Recall,{avg_recall:.4f}")
        csv_lines.append(f"Average F1 Score,{avg_f1:.4f}")
        csv_lines.append("")
    
    # Write to CSV file
    with open(csv_report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(csv_lines))
    
    print(f"\n✅ CSV report saved to: {csv_report_path}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

