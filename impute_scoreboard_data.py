#!/usr/bin/env python3
"""
Scoreboard Data Imputation Script

Intelligent data quality system that:
1. Detects anomalies in scoreboard fields (game_time, period, play_clock)
2. Imputes missing fields using pattern analysis
3. Creates new imputed tables without modifying originals
4. Generates detailed validation reports

Usage:
    python scripts/impute_scoreboard_data.py --game "Duke vs SMU"
    python scripts/impute_scoreboard_data.py --game "Duke vs SMU" --dry-run
    python scripts/impute_scoreboard_data.py --all-games
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from config.applicationPropertiesConfig import ApplicationPropertiesConfig
from utils.database_util import DatabaseUtility
from scripts.data_quality.anomaly_detector import AnomalyDetector
from scripts.data_quality.field_imputer import FieldImputer
from scripts.data_quality.period_handler import PeriodHandler
from scripts.data_quality.table_manager import TableManager
from scripts.data_quality.validation import ValidationReporter


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('imputation.log')
    ]
)
logger = logging.getLogger(__name__)


class ScoreboardDataImputer:
    """Main class for scoreboard data imputation."""
    
    def __init__(self, game_name: str, config: dict):
        """
        Initialize imputer.
        
        Args:
            game_name: Name of the game to process
            config: Configuration dictionary with:
                - window_size: Window for anomaly detection (default: 5)
                - imputation_window: Window for imputation (default: 3)
                - min_confidence: Minimum confidence for imputation (default: 0.7)
                - max_consecutive_missing: Max consecutive frames to impute (default: 10)
        """
        self.game_name = game_name
        self.config = config
        
        # Initialize utilities
        self.db_utility = DatabaseUtility()
        self.anomaly_detector = AnomalyDetector(window_size=config.get('window_size', 5))
        self.field_imputer = FieldImputer(
            window_size=config.get('imputation_window', 3),
            min_confidence=config.get('min_confidence', 0.7),
            max_consecutive_missing=config.get('max_consecutive_missing', 10)
        )
        self.period_handler = PeriodHandler()
        self.table_manager = TableManager(self.db_utility)
        self.reporter = ValidationReporter()
        
        # Table names
        self.original_table = self._get_table_name(game_name)
        self.imputed_table = f"{self.original_table}_imputed"
    
    def run(self, dry_run: bool = False) -> bool:
        """
        Execute the imputation process with performance timing.
        
        Args:
            dry_run: If True, only analyze without writing to database
            
        Returns:
            True if successful
        """
        # Initialize timing tracker
        timings = {}
        pipeline_start = time.perf_counter()
        
        try:
            logger.info("=" * 80)
            logger.info(f"🚀 Starting imputation pipeline for game: {self.game_name}")
            logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
            logger.info("=" * 80)
            
            # ========== Step 1: Load data ==========
            logger.info("\n📥 Step 1: Loading data from database...")
            step_start = time.perf_counter()
            
            df = self._load_data()
            if df is None or len(df) == 0:
                logger.error("No data found for game")
                return False
            
            step_duration = time.perf_counter() - step_start
            timings['data_loading'] = step_duration
            logger.info(f"✅ Loaded {len(df)} frames")
            logger.info(f"⏱️  Step 1 completed in {step_duration:.2f} seconds")
            
            # ========== Step 2: Detect anomalies ==========
            logger.info("\n🔍 Step 2: Detecting anomalies...")
            step_start = time.perf_counter()
            
            df_with_anomalies = self.anomaly_detector.detect_anomalies(df)
            
            step_duration = time.perf_counter() - step_start
            timings['anomaly_detection'] = step_duration
            logger.info(f"⏱️  Step 2 completed in {step_duration:.2f} seconds")
            
            # ========== Step 3: Detect period transitions ==========
            logger.info("\n🔄 Step 3: Detecting period transitions...")
            step_start = time.perf_counter()
            
            transitions = self.period_handler.detect_transitions(df_with_anomalies)
            df_with_anomalies = self.period_handler.mark_transition_frames(df_with_anomalies, transitions)
            
            step_duration = time.perf_counter() - step_start
            timings['period_transition_detection'] = step_duration
            logger.info(f"⏱️  Step 3 completed in {step_duration:.2f} seconds")
            
            # ========== Step 4: Impute missing fields ==========
            logger.info("\n🔧 Step 4: Imputing missing fields...")
            step_start = time.perf_counter()
            
            df_imputed = self.field_imputer.impute_missing_fields(df_with_anomalies)
            
            step_duration = time.perf_counter() - step_start
            timings['field_imputation'] = step_duration
            logger.info(f"⏱️  Step 4 completed in {step_duration:.2f} seconds")
            
            # ========== Step 5: Generate report ==========
            logger.info("\n📊 Step 5: Generating validation report...")
            step_start = time.perf_counter()
            
            report = self.reporter.generate_report(
                self.game_name,
                df_with_anomalies,
                df_imputed,
                self.original_table,
                self.imputed_table
            )
            
            step_duration = time.perf_counter() - step_start
            timings['report_generation'] = step_duration
            
            # Print summary
            report.print_summary()
            logger.info(f"⏱️  Step 5 completed in {step_duration:.2f} seconds")
            
            # ========== Step 6: Write to database ==========
            if not dry_run:
                logger.info("\n💾 Step 6: Writing imputed data to database...")
                step_start = time.perf_counter()
                
                # Create imputed table
                table_create_start = time.perf_counter()
                self.table_manager.create_imputed_table(self.original_table)
                table_create_duration = time.perf_counter() - table_create_start
                logger.info(f"   ├─ Table creation: {table_create_duration:.2f} seconds")
                
                # Write data
                data_write_start = time.perf_counter()
                self.table_manager.write_imputed_data(df_imputed, self.imputed_table)
                data_write_duration = time.perf_counter() - data_write_start
                logger.info(f"   └─ Data writing: {data_write_duration:.2f} seconds")
                
                step_duration = time.perf_counter() - step_start
                timings['database_write'] = step_duration
                logger.info(f"✅ Successfully created imputed table: {self.imputed_table}")
                logger.info(f"⏱️  Step 6 completed in {step_duration:.2f} seconds")
            else:
                logger.info("\n⏭️  Step 6: Skipped (dry run mode)")
                timings['database_write'] = 0.0
            
            # ========== Step 7: Export reports ==========
            logger.info("\n📁 Step 7: Exporting reports...")
            step_start = time.perf_counter()
            
            self._export_reports(report, df_with_anomalies)
            
            step_duration = time.perf_counter() - step_start
            timings['report_export'] = step_duration
            logger.info(f"⏱️  Step 7 completed in {step_duration:.2f} seconds")
            
            # ========== Final Summary ==========
            total_duration = time.perf_counter() - pipeline_start
            timings['total'] = total_duration
            
            logger.info("\n" + "=" * 80)
            logger.info("⏱️  PERFORMANCE SUMMARY")
            logger.info("=" * 80)
            
            # Calculate percentages
            for step_name, duration in timings.items():
                if step_name != 'total' and total_duration > 0:
                    percentage = (duration / total_duration) * 100
                    step_display_name = step_name.replace('_', ' ').title()
                    logger.info(f"{step_display_name:.<40} {duration:>8.2f}s ({percentage:>5.1f}%)")
            
            logger.info("-" * 80)
            logger.info(f"{'Total Pipeline Time':.<40} {total_duration:>8.2f}s (100.0%)")
            logger.info("=" * 80)
            
            logger.info("\n✅ Imputation pipeline completed successfully!")
            return True
            
        except Exception as e:
            total_duration = time.perf_counter() - pipeline_start
            logger.error(f"\n❌ Imputation failed after {total_duration:.2f} seconds: {e}", exc_info=True)
            return False
    
    def _load_data(self) -> pd.DataFrame:
        """Load game data from database (only columns needed for imputation)."""
        try:
            # Ensure connection is established
            self.db_utility._ensure_connection()
            
            # 🚀 PERFORMANCE FIX: Only select columns needed for imputation
            # Excludes massive frame_features column (MB per row)
            query = f"""
            SELECT 
                frame_number,
                frame_path,
                frame_path_url,
                game_time,
                period,
                play_clock,
                is_processed_frame,
                timestamp_seconds,
                processing_timestamp,
                created_at,
                updated_at
            FROM {self.original_table} 
            ORDER BY frame_number ASC;
            """
            
            with self.db_utility.get_connection() as conn:
                df = pd.read_sql(query, conn)
            
            logger.info(f"Loaded {len(df)} frames with {len(df.columns)} columns (excluded frame_features for performance)")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return None
    
    def _get_table_name(self, game_name: str) -> str:
        """Get sanitized table name."""
        return self.db_utility._get_table_name(game_name)
    
    def _export_reports(self, report, anomalies_df):
        """Export reports to files."""
        # Create reports directory
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Generate filename base
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_game_name = self.game_name.replace(" ", "_").replace("/", "_")
        base_filename = f"imputation_{safe_game_name}_{timestamp}"
        
        # Export JSON report
        json_path = reports_dir / f"{base_filename}.json"
        report.to_json(str(json_path))
        
        # Export CSV report
        csv_path = reports_dir / f"{base_filename}.csv"
        report.to_csv(str(csv_path))
        
        # Export anomalies CSV
        anomalies_path = reports_dir / f"anomalies_{safe_game_name}_{timestamp}.csv"
        self.reporter.export_anomalies_csv(anomalies_df, str(anomalies_path))
        
        logger.info(f"Reports exported to: {reports_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Impute missing scoreboard data and correct anomalies",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--game',
        type=str,
        help='Game name to process (e.g., "Duke vs SMU")'
    )
    
    parser.add_argument(
        '--all-games',
        action='store_true',
        help='Process all games in database'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Analyze without writing to database'
    )
    
    parser.add_argument(
        '--window-size',
        type=int,
        default=5,
        help='Window size for anomaly detection (default: 5)'
    )
    
    parser.add_argument(
        '--imputation-window',
        type=int,
        default=5,
        help='Window size for imputation (default: 5 frames = ±1.7s at 3 FPS)'
    )
    
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.6,
        help='Minimum confidence threshold for imputation (default: 0.75 = 75%%, range: 0.0-1.0)'
    )
    
    parser.add_argument(
        '--max-consecutive-missing',
        type=int,
        default=6,
        help='Maximum consecutive frames with missing fields to impute (default: 5 frames = ~1.7s at 3 FPS). '
             'Longer sequences are skipped (likely commercials/timeouts/pauses)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not args.game and not args.all_games:
        parser.error("Either --game or --all-games must be specified")
    
    if args.game and args.all_games:
        parser.error("Cannot specify both --game and --all-games")
    
    # Validate confidence range
    if not (0.0 <= args.min_confidence <= 1.0):
        parser.error("--min-confidence must be between 0.0 and 1.0")
    
    # Build configuration
    config = {
        'window_size': args.window_size,
        'imputation_window': args.imputation_window,
        'min_confidence': args.min_confidence,
        'max_consecutive_missing': args.max_consecutive_missing
    }
    
    # Process games
    if args.all_games:
        logger.error("--all-games not yet implemented. Please specify --game for now.")
        sys.exit(1)
    else:
        # Process single game
        imputer = ScoreboardDataImputer(args.game, config)
        success = imputer.run(dry_run=args.dry_run)
        
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

