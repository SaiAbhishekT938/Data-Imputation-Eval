#!/usr/bin/env python3
"""
Bulk Data Extractor for IMPUTED Tables

Extracts frame-wise data from imputed game tables (_imputed suffix) to CSV.
Includes all original columns PLUS imputation tracking metadata.

Usage:
    python bulk_data_extractor_imputed.py
    
Configuration:
    Edit game_name variable in __main__ section (line ~140)
"""

from utils.database_util import DatabaseUtility
from sqlalchemy import text


def get_imputed_table_name(game_name: str) -> str:
    """Generate imputed table name from game name."""
    # Replace spaces and special characters with underscores
    safe_name = ''.join(c if c.isalnum() else '_' for c in game_name.lower())
    # Ensure it starts with a letter
    if safe_name and not safe_name[0].isalpha():
        safe_name = 'game_' + safe_name
    return f"game_frames_{safe_name}_imputed"


def extract_bulk_data_imputed(game_name: str, max_rows: int = 50000):
    """
    Extract bulk data from the IMPUTED game table with all columns.
    
    Args:
        game_name: Name of the game
        max_rows: Maximum number of rows to extract (default: 50000)
    
    Returns:
        List of dictionaries containing the extracted data
    """
    db_util = DatabaseUtility()
    table_name = get_imputed_table_name(game_name)
    
    # Ensure connection is established
    db_util._ensure_connection()
    
    # First, check if the imputed table exists
    with db_util.get_connection() as conn:
        check_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = :table_name
        );
        """
        result = conn.execute(text(check_query), {"table_name": table_name})
        table_exists = result.scalar()
        
        if not table_exists:
            print(f"❌ ERROR: Imputed table '{table_name}' does not exist!")
            print(f"")
            print(f"To create the imputed table, run:")
            print(f"  python scripts/impute_scoreboard_data.py --game \"{game_name}\"")
            return None
    
    original_table_name = table_name[:-9] if table_name.endswith("_imputed") else None
    frame_path_available = False
    
    with db_util.get_connection() as conn:
        if original_table_name:
            column_check_query = """
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = :table_name 
                AND column_name = 'frame_path'
            );
            """
            result = conn.execute(text(column_check_query), {"table_name": table_name})
            frame_path_available = bool(result.scalar())
        else:
            print("⚠️ Unable to derive original table name. Frame paths may be unavailable.")
    
        # SQL query for MINIMAL imputed table
        # This table contains ONLY essential fields + imputation tracking
        if frame_path_available:
            query = f"""
            SELECT 
                frame_number,
                frame_path,
                -- Corrected essential fields
                game_time,
                period,
                play_clock,
                -- Imputation flags (what was changed)
                imputed_game_time,
                imputed_period,
                imputed_play_clock,
                -- Original values (before correction)
                original_game_time,
                original_period,
                original_play_clock,
                -- Imputation metadata
                imputation_confidence,
                imputation_method,
                imputation_timestamp
            FROM {table_name}
            ORDER BY frame_number ASC
            LIMIT {max_rows};
            """
        else:
            join_clause = ""
            if original_table_name:
                join_clause = f"""
                LEFT JOIN {original_table_name} original
                    ON original.frame_number = imputed.frame_number
                """
            query = f"""
            SELECT 
                imputed.frame_number,
                original.frame_path,
                -- Corrected essential fields
                imputed.game_time,
                imputed.period,
                imputed.play_clock,
                -- Imputation flags (what was changed)
                imputed.imputed_game_time,
                imputed.imputed_period,
                imputed.imputed_play_clock,
                -- Original values (before correction)
                imputed.original_game_time,
                imputed.original_period,
                imputed.original_play_clock,
                -- Imputation metadata
                imputed.imputation_confidence,
                imputed.imputation_method,
                imputed.imputation_timestamp
            FROM {table_name} imputed
            {join_clause}
            ORDER BY imputed.frame_number ASC
            LIMIT {max_rows};
            """
            if not join_clause:
                print("⚠️ Frame path column not found and original table unavailable; frame_path will be NULL in export.")
            else:
                print(f"ℹ️ Frame path column missing in imputed table; pulling values from original table '{original_table_name}'.")
        
        print(f"Executing bulk data extraction for IMPUTED table: {table_name}")
        print(f"Query: {query}")
        
        # Execute query and fetch all results
        result = conn.execute(text(query))
        rows = result.fetchall()
        columns = result.keys()
        
        print(f"✅ Found {len(rows)} rows (limited to {max_rows})")
        
        # Convert to list of dictionaries for better readability
        results = []
        for row in rows:
            results.append(dict(zip(columns, row)))
        
        # Print sample of results
        if results:
            print(f"\n📊 Sample of first 3 rows:")
            for i, row in enumerate(results[:3]):
                print(f"\nRow {i+1}:")
                print(f"  Frame: {row.get('frame_number')}")
                if 'frame_path' in row:
                    print(f"  Frame Path: {row.get('frame_path')}")
                print(f"  Game Time: {row.get('game_time')} (imputed: {row.get('imputed_game_time')})")
                print(f"  Period: {row.get('period')} (imputed: {row.get('imputed_period')})")
                print(f"  Play Clock: {row.get('play_clock')} (imputed: {row.get('imputed_play_clock')})")
                if row.get('imputed_game_time') or row.get('imputed_period') or row.get('imputed_play_clock'):
                    print(f"  Confidence: {row.get('imputation_confidence')}")
                    print(f"  Method: {row.get('imputation_method')}")
        
        return results


def save_to_csv(data: list, game_name: str, filename: str = None):
    """
    Save extracted imputed data to CSV file.
    
    Args:
        data: List of dictionaries containing the data
        game_name: Game name for filename
        filename: Output filename (optional)
    """
    if not data:
        print("❌ No data to save")
        return
    
    import os
    from datetime import datetime
    from pandas import DataFrame
    
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_game_name = game_name.replace(" ", "_").replace("/", "_")
        filename = f"imputed_data_{safe_game_name}_{timestamp}.csv"
    
    # Ensure output directory exists
    output_dir = "exports"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    # Convert to DataFrame
    df = DataFrame(data)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    
    print(f"\n✅ Data saved to: {filepath}")
    print(f"📊 Total rows saved: {len(data)}")
    
    # Print imputation statistics
    imputed_game_time = sum(1 for row in data if row.get('imputed_game_time'))
    imputed_period = sum(1 for row in data if row.get('imputed_period'))
    imputed_play_clock = sum(1 for row in data if row.get('imputed_play_clock'))
    
    print(f"\n📈 Imputation Statistics:")
    print(f"  Game Time Imputed: {imputed_game_time} frames")
    print(f"  Period Imputed: {imputed_period} frames")
    print(f"  Play Clock Imputed: {imputed_play_clock} frames")
    
    total_imputed = len([row for row in data if (
        row.get('imputed_game_time') or 
        row.get('imputed_period') or 
        row.get('imputed_play_clock')
    )])
    print(f"  Total Frames with Imputations: {total_imputed}")
    
    return filepath


def print_summary_statistics(data: list):
    """
    Print detailed summary statistics for the imputed data.
    
    Args:
        data: List of dictionaries containing the data
    """
    if not data:
        return
    
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    print(f"\n📊 Overall:")
    print(f"  Total Frames: {len(data)}")
    
    # Count by period
    period_counts = {}
    for row in data:
        period = row.get('period', 'Unknown')
        if period:
            period_counts[period] = period_counts.get(period, 0) + 1
    
    print(f"\n🏈 Frames by Period:")
    for period, count in sorted(period_counts.items()):
        print(f"  {period}: {count}")
    
    # Count processed frames
    processed_count = sum(1 for row in data if row.get('is_processed_frame', False))
    print(f"\n✅ Processed Frames: {processed_count}/{len(data)} ({processed_count/len(data)*100:.1f}%)")
    
    # Imputation details
    imputed_game_time = sum(1 for row in data if row.get('imputed_game_time'))
    imputed_period = sum(1 for row in data if row.get('imputed_period'))
    imputed_play_clock = sum(1 for row in data if row.get('imputed_play_clock'))
    
    print(f"\n🔧 Imputation Details:")
    print(f"  Game Time Imputed: {imputed_game_time} ({imputed_game_time/len(data)*100:.1f}%)")
    print(f"  Period Imputed: {imputed_period} ({imputed_period/len(data)*100:.1f}%)")
    print(f"  Play Clock Imputed: {imputed_play_clock} ({imputed_play_clock/len(data)*100:.1f}%)")
    
    # Calculate average confidence for imputed frames
    confidence_values = [
        row.get('imputation_confidence', 0) 
        for row in data 
        if row.get('imputation_confidence') is not None
    ]
    
    if confidence_values:
        avg_confidence = sum(confidence_values) / len(confidence_values)
        print(f"\n📈 Confidence:")
        print(f"  Average Confidence: {avg_confidence:.2f}")
        print(f"  Frames with Confidence Data: {len(confidence_values)}")
    
    # Imputation methods breakdown
    methods = {}
    for row in data:
        method = row.get('imputation_method')
        if method:
            methods[method] = methods.get(method, 0) + 1
    
    if methods:
        print(f"\n🔍 Imputation Methods Used:")
        for method, count in sorted(methods.items(), key=lambda x: x[1], reverse=True):
            print(f"  {method}: {count}")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # ========================================
    # CONFIGURATION: Set your game name here
    # ========================================
    game_name = "game_9"  # Change this to your game name
    max_rows = 50000       # Maximum rows to extract (adjust as needed)
    
    print(f"\n{'='*60}")
    print(f"IMPUTED DATA EXTRACTOR")
    print(f"{'='*60}")
    print(f"Game: {game_name}")
    print(f"Max Rows: {max_rows}")
    print(f"{'='*60}\n")
    
    # Extract data from imputed table
    data = extract_bulk_data_imputed(game_name, max_rows=max_rows)
    
    if data:
        # Save to CSV
        csv_path = save_to_csv(data, game_name)
        
        # Print summary statistics
        print_summary_statistics(data)
        
        print(f"✅ Extraction complete!")
        print(f"📁 CSV file: {csv_path}")
        
    else:
        print("\n❌ No data found or table doesn't exist")
        print("\nTroubleshooting:")
        print("1. Verify the imputed table exists:")
        print("   SELECT table_name FROM information_schema.tables")
        print("   WHERE table_name LIKE '%_imputed';")
        print("\n2. If table doesn't exist, create it first:")
        print(f"   python scripts/impute_scoreboard_data.py --game \"{game_name}\"")

