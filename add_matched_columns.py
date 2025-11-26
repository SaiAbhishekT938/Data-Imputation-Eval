#!/usr/bin/env python3
"""Add Matched Columns to Imputation Evaluation Files.

This script adds matched_<essential_field> columns to CSV/Excel files that contain
imputation evaluation data. For each essential field, it creates a boolean column
indicating whether the imputed value matches the expected value (only for rows
where imputation was performed).

Logic:
    matched_<field> = TRUE if:
        - imputed_<field> == TRUE AND
        - <field> == expected_<field>
    Otherwise: FALSE

Usage:
    python scripts/add_matched_columns.py --input exports/imputed_data_game_6_20251113_215047_wexp.csv
    python scripts/add_matched_columns.py --input exports/file.csv --output exports/custom_output.csv
    python scripts/add_matched_columns.py --input exports/file.xlsx

Output:
    By default, creates a new CSV file with "_wmatch" suffix.
    Example: input.csv -> input_wmatch.csv
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd


ESSENTIAL_FIELDS = ["game_time", "period", "play_clock"]
IMPUTED_FLAG_TEMPLATE = "imputed_{field}"
EXPECTED_PREFIX = "expected_"
MATCHED_PREFIX = "matched_"


def add_matched_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add matched_<essential_field> columns to the DataFrame.
    
    For each essential field:
    - matched_<field> = TRUE if imputed_<field> == TRUE AND <field> == expected_<field>
    - matched_<field> = FALSE otherwise
    
    Args:
        df: DataFrame with imputed, expected, and imputed flag columns
        
    Returns:
        DataFrame with added matched_ columns
    """
    df = df.copy()
    
    for field in ESSENTIAL_FIELDS:
        imputed_flag_col = IMPUTED_FLAG_TEMPLATE.format(field=field)
        expected_col = f"{EXPECTED_PREFIX}{field}"
        matched_col = f"{MATCHED_PREFIX}{field}"
        
        # Check if required columns exist
        if field not in df.columns:
            print(f"⚠️  Warning: Column '{field}' not found. Skipping matched_{field}.")
            continue
        
        if imputed_flag_col not in df.columns:
            print(f"⚠️  Warning: Column '{imputed_flag_col}' not found. Skipping matched_{field}.")
            continue
        
        if expected_col not in df.columns:
            print(f"⚠️  Warning: Column '{expected_col}' not found. Skipping matched_{field}.")
            continue
        
        # Initialize matched column with FALSE
        df[matched_col] = False
        
        # Find rows where imputation was performed
        imputed_mask = df[imputed_flag_col] == True
        
        if imputed_mask.sum() == 0:
            print(f"ℹ️  No imputed rows found for {field}. All matched_{field} values set to FALSE.")
            continue
        
        # For imputed rows, check if imputed value matches expected value
        imputed_rows = df.loc[imputed_mask]
        
        # Compare values (handle NaN/None cases)
        for idx in imputed_rows.index:
            imputed_value = df.loc[idx, field]
            expected_value = df.loc[idx, expected_col]
            imputed_flag = df.loc[idx, imputed_flag_col]
            
            # Only set to TRUE if imputed_flag is TRUE and values match
            if imputed_flag == True:
                # Handle NaN/None/empty string comparisons
                if pd.isna(imputed_value) and pd.isna(expected_value):
                    df.loc[idx, matched_col] = True
                elif pd.isna(imputed_value) or pd.isna(expected_value):
                    df.loc[idx, matched_col] = False
                elif str(imputed_value).strip() == str(expected_value).strip():
                    df.loc[idx, matched_col] = True
                else:
                    df.loc[idx, matched_col] = False
            else:
                df.loc[idx, matched_col] = False
        
        # Count matches
        matched_count = df.loc[imputed_mask, matched_col].sum()
        total_imputed = imputed_mask.sum()
        print(f"✅ {field}: {matched_count}/{total_imputed} imputed values match expected ({matched_count/total_imputed*100:.1f}%)")
    
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add matched_<essential_field> columns to imputation evaluation files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV/Excel file (e.g., exports/imputed_data_game_6_20251113_215047_wexp.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to save output CSV file. If not provided, creates input_wmatch.csv"
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
    
    # Read file (support both CSV and Excel)
    if input_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(input_path)
        file_type = 'excel'
    else:
        df = pd.read_csv(input_path)
        file_type = 'csv'
    
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Add matched columns
    print("\n🔍 Adding matched columns...")
    df_with_matched = add_matched_columns(df)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
        # Ensure output is CSV
        if output_path.suffix.lower() not in ['.csv']:
            output_path = output_path.with_suffix('.csv')
    else:
        # Default: add _wmatch suffix and always output as CSV
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}_wmatch.csv"
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Always save as CSV
    print(f"\n💾 Saving to: {output_path}")
    df_with_matched.to_csv(output_path, index=False)
    
    print(f"✅ Successfully added matched columns. Output saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for field in ESSENTIAL_FIELDS:
        matched_col = f"{MATCHED_PREFIX}{field}"
        if matched_col in df_with_matched.columns:
            total_matched = df_with_matched[matched_col].sum()
            total_rows = len(df_with_matched)
            print(f"{matched_col}: {total_matched} TRUE out of {total_rows} total rows")
    
    print("\n" + "=" * 80)
    print("New columns added:")
    for field in ESSENTIAL_FIELDS:
        matched_col = f"{MATCHED_PREFIX}{field}"
        if matched_col in df_with_matched.columns:
            print(f"  ✅ {matched_col}")
        else:
            print(f"  ⚠️  {matched_col} (skipped - missing required columns)")


if __name__ == "__main__":
    main()

