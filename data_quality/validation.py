#!/usr/bin/env python3
"""
Validation & Reporting - Generate detailed reports for imputation results.

Provides:
- Imputation statistics
- Confidence distribution
- Per-period summaries
- CSV/JSON export
"""

import logging
from typing import Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, asdict
import pandas as pd
import json


logger = logging.getLogger(__name__)


@dataclass
class ImputationReport:
    """Comprehensive imputation report."""
    
    game_name: str
    timestamp: str
    total_frames: int
    frames_with_anomalies: int
    frames_imputed: int
    
    # Field-level statistics
    game_time_anomalies: int
    period_anomalies: int
    play_clock_anomalies: int
    
    game_time_imputed: int
    period_imputed: int
    play_clock_imputed: int
    
    # Confidence distribution
    high_confidence_count: int  # >= 0.8
    medium_confidence_count: int  # 0.6 - 0.8
    low_confidence_count: int  # < 0.6
    
    avg_confidence: float
    
    # Per-period summaries
    period_summaries: Dict[str, Dict[str, int]]
    
    # Table information
    original_table: str
    imputed_table: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, output_path: str):
        """Export report as JSON."""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Saved JSON report: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON report: {e}")
    
    def to_csv(self, output_path: str):
        """Export report as CSV."""
        try:
            # Convert to DataFrame for CSV export
            report_data = {
                'Metric': [],
                'Value': []
            }
            
            report_dict = self.to_dict()
            for key, value in report_dict.items():
                if key != 'period_summaries':  # Handle separately
                    report_data['Metric'].append(key)
                    report_data['Value'].append(str(value))
            
            df = pd.DataFrame(report_data)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved CSV report: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save CSV report: {e}")
    
    def print_summary(self):
        """Print console summary."""
        print("\n" + "="*60)
        print(f"IMPUTATION REPORT: {self.game_name}")
        print(f"Timestamp: {self.timestamp}")
        print("="*60)
        print(f"\n📊 OVERVIEW:")
        print(f"  Total Frames: {self.total_frames}")
        print(f"  Frames with Anomalies: {self.frames_with_anomalies}")
        print(f"  Frames Imputed: {self.frames_imputed}")
        print(f"\n🔍 ANOMALIES DETECTED:")
        print(f"  Game Time: {self.game_time_anomalies}")
        print(f"  Period: {self.period_anomalies}")
        print(f"  Play Clock: {self.play_clock_anomalies}")
        print(f"\n🔧 FIELDS IMPUTED:")
        print(f"  Game Time: {self.game_time_imputed}")
        print(f"  Period: {self.period_imputed}")
        print(f"  Play Clock: {self.play_clock_imputed}")
        print(f"\n📈 CONFIDENCE DISTRIBUTION:")
        print(f"  High (≥0.8): {self.high_confidence_count}")
        print(f"  Medium (0.6-0.8): {self.medium_confidence_count}")
        print(f"  Low (<0.6): {self.low_confidence_count}")
        print(f"  Average: {self.avg_confidence:.2f}")
        print(f"\n🏈 PER-PERIOD SUMMARY:")
        for period, stats in self.period_summaries.items():
            print(f"  {period}:")
            print(f"    Total Frames: {stats.get('total_frames', 0)}")
            print(f"    Imputed: {stats.get('imputed_count', 0)}")
        print(f"\n💾 TABLES:")
        print(f"  Original: {self.original_table}")
        print(f"  Imputed: {self.imputed_table}")
        print("="*60 + "\n")


class ValidationReporter:
    """Generate validation reports for imputation."""
    
    def __init__(self):
        """Initialize validation reporter."""
        pass
    
    def generate_report(self, game_name: str, original_df: pd.DataFrame, 
                       imputed_df: pd.DataFrame, original_table: str, 
                       imputed_table: str) -> ImputationReport:
        """
        Generate comprehensive imputation report.
        
        Args:
            game_name: Name of the game
            original_df: Original DataFrame (with anomaly flags)
            imputed_df: Imputed DataFrame
            original_table: Original table name
            imputed_table: Imputed table name
            
        Returns:
            ImputationReport object
        """
        logger.info("Generating imputation report...")
        
        # Count anomalies
        game_time_anomalies = original_df.get('game_time_anomaly', pd.Series([False])).sum()
        period_anomalies = original_df.get('period_anomaly', pd.Series([False])).sum()
        play_clock_anomalies = original_df.get('play_clock_anomaly', pd.Series([False])).sum()
        
        frames_with_anomalies = len(original_df[
            (original_df.get('game_time_anomaly', False) == True) |
            (original_df.get('period_anomaly', False) == True) |
            (original_df.get('play_clock_anomaly', False) == True)
        ])
        
        # Count imputations
        game_time_imputed = imputed_df.get('imputed_game_time', pd.Series([False])).sum()
        period_imputed = imputed_df.get('imputed_period', pd.Series([False])).sum()
        play_clock_imputed = imputed_df.get('imputed_play_clock', pd.Series([False])).sum()
        
        frames_imputed = len(imputed_df[
            (imputed_df.get('imputed_game_time', False) == True) |
            (imputed_df.get('imputed_period', False) == True) |
            (imputed_df.get('imputed_play_clock', False) == True)
        ])
        
        # Confidence distribution
        imputed_rows = imputed_df[
            (imputed_df.get('imputed_game_time', False) == True) |
            (imputed_df.get('imputed_period', False) == True) |
            (imputed_df.get('imputed_play_clock', False) == True)
        ]
        
        if len(imputed_rows) > 0:
            confidences = imputed_rows['imputation_confidence']
            high_conf = (confidences >= 0.8).sum()
            medium_conf = ((confidences >= 0.6) & (confidences < 0.8)).sum()
            low_conf = (confidences < 0.6).sum()
            avg_conf = confidences.mean()
        else:
            high_conf = medium_conf = low_conf = 0
            avg_conf = 0.0
        
        # Per-period summaries
        period_summaries = self._generate_period_summaries(imputed_df)
        
        report = ImputationReport(
            game_name=game_name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_frames=len(imputed_df),
            frames_with_anomalies=int(frames_with_anomalies),
            frames_imputed=int(frames_imputed),
            game_time_anomalies=int(game_time_anomalies),
            period_anomalies=int(period_anomalies),
            play_clock_anomalies=int(play_clock_anomalies),
            game_time_imputed=int(game_time_imputed),
            period_imputed=int(period_imputed),
            play_clock_imputed=int(play_clock_imputed),
            high_confidence_count=int(high_conf),
            medium_confidence_count=int(medium_conf),
            low_confidence_count=int(low_conf),
            avg_confidence=float(avg_conf),
            period_summaries=period_summaries,
            original_table=original_table,
            imputed_table=imputed_table
        )
        
        logger.info("Report generated successfully")
        return report
    
    def _generate_period_summaries(self, df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """Generate per-period summaries."""
        summaries = {}
        
        # Get unique periods
        periods = df['period'].dropna().unique()
        
        for period in periods:
            if period and period != '':
                period_df = df[df['period'] == period]
                
                imputed_count = len(period_df[
                    (period_df.get('imputed_game_time', False) == True) |
                    (period_df.get('imputed_period', False) == True) |
                    (period_df.get('imputed_play_clock', False) == True)
                ])
                
                summaries[str(period)] = {
                    'total_frames': len(period_df),
                    'imputed_count': int(imputed_count)
                }
        
        return summaries
    
    def export_anomalies_csv(self, df: pd.DataFrame, output_path: str):
        """
        Export detected anomalies to CSV for review.
        
        Args:
            df: DataFrame with anomaly flags
            output_path: Path for CSV output
        """
        try:
            # Filter rows with anomalies
            anomaly_df = df[
                (df.get('game_time_anomaly', False) == True) |
                (df.get('period_anomaly', False) == True) |
                (df.get('play_clock_anomaly', False) == True)
            ].copy()
            
            if len(anomaly_df) == 0:
                logger.info("No anomalies to export")
                return
            
            # Select relevant columns
            export_cols = [
                'frame_number', 'game_time', 'period', 'play_clock',
                'game_time_anomaly', 'suggested_game_time',
                'period_anomaly', 'suggested_period',
                'play_clock_anomaly', 'suggested_play_clock',
                'anomaly_confidence'
            ]
            
            # Filter to columns that exist
            export_cols = [col for col in export_cols if col in anomaly_df.columns]
            
            anomaly_df[export_cols].to_csv(output_path, index=False)
            logger.info(f"Exported {len(anomaly_df)} anomalies to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export anomalies: {e}")

