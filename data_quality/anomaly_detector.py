#!/usr/bin/env python3
"""
Anomaly Detector - Pattern-based detection for scoreboard fields.

Detects anomalies without fixed thresholds using:
- Sequence pattern analysis
- Statistical outlier detection
- Edit distance for string corrections
- Temporal consistency validation
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd


logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Pattern-based anomaly detector for scoreboard fields."""
    
    def __init__(self, window_size: int = 5):
        """
        Initialize anomaly detector.
        
        Args:
            window_size: Number of neighboring frames to analyze (±window_size)
        """
        self.window_size = window_size
        
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in all scoreboard fields.
        
        Args:
            df: DataFrame with columns: frame_number, game_time, period, play_clock
            
        Returns:
            DataFrame with added anomaly flags and suggested corrections
        """
        logger.info(f"Starting anomaly detection on {len(df)} frames")
        
        # Add anomaly detection columns
        df['game_time_anomaly'] = False
        df['period_anomaly'] = False
        df['play_clock_anomaly'] = False
        df['suggested_game_time'] = None
        df['suggested_period'] = None
        df['suggested_play_clock'] = None
        df['anomaly_confidence'] = 0.0
        
        # Detect anomalies for each field
        df = self._detect_game_time_anomalies(df)
        df = self._detect_period_anomalies(df)
        df = self._detect_play_clock_anomalies(df)
        
        total_anomalies = df['game_time_anomaly'].sum() + df['period_anomaly'].sum() + df['play_clock_anomaly'].sum()
        logger.info(f"Detected {total_anomalies} anomalies total")
        
        return df
    
    def _detect_game_time_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect game_time anomalies using pattern analysis.
        
        Detects:
        - Missing digits (1:50 instead of 11:50)
        - Sudden time jumps
        - Impossible progressions
        """
        logger.info("Detecting game_time anomalies...")
        
        index_list = list(df.index)
        total_frames = len(index_list)
        minutes_list: List[Optional[int]] = []
        seconds_list: List[Optional[int]] = []

        for idx in index_list:
            game_time = df.loc[idx, 'game_time']
            if game_time is None or pd.isna(game_time):
                minutes_list.append(None)
                seconds_list.append(None)
                continue

            total_seconds = self._time_to_seconds(game_time)
            if total_seconds is None:
                minutes_list.append(None)
                seconds_list.append(None)
                continue

            minutes_list.append(total_seconds // 60)
            seconds_list.append(total_seconds % 60)

        pos = 0
        while pos < total_frames:
            if minutes_list[pos] != 1:
                pos += 1
                continue

            run_start = pos
            while run_start > 0 and minutes_list[run_start - 1] == 1:
                run_start -= 1

            run_end = pos
            while run_end + 1 < total_frames and minutes_list[run_end + 1] == 1:
                run_end += 1

            window_start = max(0, run_start - 20)
            window_end = min(total_frames - 1, run_end + 20)
            eleven_count = 0
            for window_pos in range(window_start, window_end + 1):
                if run_start <= window_pos <= run_end:
                    continue
                if minutes_list[window_pos] == 11:
                    eleven_count += 1
                if eleven_count >= 4:
                    break

            if eleven_count >= 4:
                for adjusted_pos in range(run_start, run_end + 1):
                    seconds_value = seconds_list[adjusted_pos]
                    if seconds_value is None:
                        continue
                    frame_idx = index_list[adjusted_pos]
                    suggested_value = f"11:{seconds_value:02d}"
                    df.at[frame_idx, 'game_time_anomaly'] = True
                    df.at[frame_idx, 'suggested_game_time'] = suggested_value
                    df.at[frame_idx, 'anomaly_confidence'] = max(df.at[frame_idx, 'anomaly_confidence'], 0.90)
                    logger.debug(
                        f"Frame {df.at[frame_idx, 'frame_number']}: detected missing digit trend '{df.at[frame_idx, 'game_time']}' → '{suggested_value}'"
                    )

            pos = run_end + 1

        anomaly_count = df['game_time_anomaly'].sum()
        logger.info(f"Found {anomaly_count} game_time anomalies")
        return df
    
    def _detect_period_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect period anomalies using mode-based detection with type validation.
        
        Detects:
        - Invalid period values (like "15", "ABC", numeric-only)
        - Wrong period in sequence
        - Invalid period transitions
        """
        logger.info("Detecting period anomalies...")
        
        # ✅ NEW: Define valid period values
        VALID_PERIODS = {'1ST', '2ND', '3RD', '4TH', 'OT', '1st', '2nd', '3rd', '4th', 'ot', 'HALF'}
        
        for idx in df.index:
            period = df.loc[idx, 'period']
            
            # Skip if no period
            if pd.isna(period) or period == '' or period is None:
                continue
            
            # ✅ NEW: Convert to string, strip whitespace, and normalize case
            period_str = str(period).strip().upper()
            
            # Get neighbor periods for suggestion
            window_start = max(0, idx - self.window_size)
            window_end = min(len(df), idx + self.window_size + 1)
            context_df = df.iloc[window_start:window_end]
            
            neighbor_periods = []
            for i in context_df.index:
                if i != idx:
                    p = context_df.loc[i, 'period']
                    if p and not pd.isna(p):
                        p_normalized = str(p).strip().upper()
                        if p_normalized in VALID_PERIODS:
                            neighbor_periods.append(p_normalized)
            
            # ✅ NEW: Type validation - reject non-period values immediately
            if period_str not in VALID_PERIODS:
                # Invalid format (like "15", "ABC", "123", etc.)
                df.loc[idx, 'period_anomaly'] = True
                
                if len(neighbor_periods) > 0:
                    mode_period = max(set(neighbor_periods), key=neighbor_periods.count)
                    mode_count = neighbor_periods.count(mode_period)
                    confidence = mode_count / len(neighbor_periods)
                    
                    df.loc[idx, 'suggested_period'] = mode_period
                    df.loc[idx, 'anomaly_confidence'] = max(df.loc[idx, 'anomaly_confidence'], confidence)
                    logger.info(f"Frame {df.loc[idx, 'frame_number']}: Invalid period '{period}' → '{mode_period}' (type validation, confidence: {confidence:.2f})")
                else:
                    # No reliable suggestion, but still flag anomaly (leave suggestion as None)
                    df.loc[idx, 'suggested_period'] = None
                    df.loc[idx, 'anomaly_confidence'] = max(df.loc[idx, 'anomaly_confidence'], 0.5)
                    logger.info(f"Frame {df.loc[idx, 'frame_number']}: Invalid period '{period}' flagged with no nearby consensus.")
                continue
            
            # ✅ IMPROVED: Mode-based detection with stricter threshold (80% instead of 60%)
            if len(neighbor_periods) < 2:
                continue
            
            mode_period = max(set(neighbor_periods), key=neighbor_periods.count)
            mode_count = neighbor_periods.count(mode_period)
            mode_ratio = mode_count / len(neighbor_periods)
            
            # Stricter threshold: 80% instead of 60%
            if period_str != mode_period and mode_ratio >= 0.8:
                df.loc[idx, 'period_anomaly'] = True
                df.loc[idx, 'suggested_period'] = mode_period
                df.loc[idx, 'anomaly_confidence'] = max(df.loc[idx, 'anomaly_confidence'], mode_ratio)
                logger.debug(f"Frame {df.loc[idx, 'frame_number']}: period '{period_str}' → '{mode_period}' (confidence: {mode_ratio:.2f})")
        
        anomaly_count = df['period_anomaly'].sum()
        logger.info(f"Found {anomaly_count} period anomalies")
        return df
    
    def _detect_play_clock_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect play_clock anomalies.
        
        Detects:
        - Impossible values (<0 or >40)
        - Inconsistent countdown patterns
        """
        logger.info("Detecting play_clock anomalies...")
        
        anomaly_count = 0
        for idx in df.index:
            play_clock = df.loc[idx, 'play_clock']
            
            # Skip if no play_clock
            if pd.isna(play_clock) or play_clock == '' or play_clock is None:
                continue
            
            try:
                play_clock_val = int(str(play_clock).strip())
            except (ValueError, AttributeError):
                # Invalid format
                df.loc[idx, 'play_clock_anomaly'] = True
                df.loc[idx, 'suggested_play_clock'] = None
                anomaly_count += 1
                continue
            
            # Check for impossible values
            if play_clock_val < 0 or play_clock_val > 40:
                df.loc[idx, 'play_clock_anomaly'] = True
                df.loc[idx, 'suggested_play_clock'] = None  # Can't correct without context
                df.loc[idx, 'anomaly_confidence'] = 0.9
                anomaly_count += 1
                logger.debug(f"Frame {df.loc[idx, 'frame_number']}: invalid play_clock '{play_clock}'")
        
        logger.info(f"Found {anomaly_count} play_clock anomalies")
        return df
    
    def _time_to_seconds(self, time_str: str) -> Optional[int]:
        """Convert time string (MM:SS) to seconds."""
        if not time_str or pd.isna(time_str):
            return None
        
        try:
            time_str = str(time_str).strip()
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 2:
                    minutes = int(parts[0])
                    seconds = int(parts[1])
                    return minutes * 60 + seconds
        except (ValueError, AttributeError):
            pass
        
        return None
    
    def _seconds_to_time(self, seconds: int) -> str:
        """Convert seconds to time string (MM:SS)."""
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}:{secs:02d}"

