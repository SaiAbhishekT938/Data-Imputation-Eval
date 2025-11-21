#!/usr/bin/env python3
"""
Period Handler - Detects and handles period transitions in game data.

Manages:
- Period transition detection (1ST→2ND→3RD→4TH→OT)
- Time reset validation at transitions
- Per-period imputation logic
"""

import logging
from typing import List, Tuple, Dict
import pandas as pd


logger = logging.getLogger(__name__)


class PeriodHandler:
    """Handles period transitions and per-period processing."""
    
    # Standard period sequence
    PERIOD_SEQUENCE = ['1ST', '2ND', '3RD', '4TH']
    
    def __init__(self):
        """Initialize period handler."""
        pass
    
    def detect_transitions(self, df: pd.DataFrame) -> List[int]:
        """
        Detect frame indices where period transitions occur.
        
        Args:
            df: DataFrame with period column
            
        Returns:
            List of frame indices where transitions occur
        """
        transitions = []
        
        prev_period = None
        for idx in df.index:
            current_period = df.loc[idx, 'period']
            
            if pd.isna(current_period) or current_period == '':
                continue
            
            if prev_period and prev_period != current_period:
                transitions.append(idx)
                logger.debug(f"Period transition detected at frame {df.loc[idx, 'frame_number']}: {prev_period} → {current_period}")
            
            prev_period = current_period
        
        logger.info(f"Detected {len(transitions)} period transitions")
        return transitions
    
    def split_by_period(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split DataFrame into separate DataFrames per period.
        
        Args:
            df: DataFrame with period column
            
        Returns:
            Dictionary mapping period name to DataFrame
        """
        periods = {}
        
        # Get unique periods
        unique_periods = df['period'].dropna().unique()
        
        for period in unique_periods:
            if period and period != '':
                period_df = df[df['period'] == period].copy()
                periods[period] = period_df
                logger.debug(f"Period '{period}': {len(period_df)} frames")
        
        logger.info(f"Split data into {len(periods)} periods")
        return periods
    
    def validate_period_sequence(self, df: pd.DataFrame) -> bool:
        """
        Validate that periods follow logical sequence.
        
        Args:
            df: DataFrame with period column
            
        Returns:
            True if sequence is valid
        """
        unique_periods = df['period'].dropna().unique()
        
        # Get order of periods as they appear
        period_order = []
        prev_period = None
        for idx in df.index:
            current_period = df.loc[idx, 'period']
            if current_period and not pd.isna(current_period):
                if current_period != prev_period:
                    period_order.append(current_period)
                    prev_period = current_period
        
        # Validate sequence
        for i in range(len(period_order) - 1):
            current = period_order[i]
            next_period = period_order[i + 1]
            
            if current in self.PERIOD_SEQUENCE and next_period in self.PERIOD_SEQUENCE:
                current_idx = self.PERIOD_SEQUENCE.index(current)
                next_idx = self.PERIOD_SEQUENCE.index(next_period)
                
                if next_idx <= current_idx:
                    logger.warning(f"Invalid period sequence: {current} → {next_period}")
                    return False
        
        logger.info("Period sequence is valid")
        return True
    
    def mark_transition_frames(self, df: pd.DataFrame, transitions: List[int]) -> pd.DataFrame:
        """
        Mark frames that are at period transitions.
        
        Args:
            df: DataFrame
            transitions: List of transition frame indices
            
        Returns:
            DataFrame with is_transition column added
        """
        df['is_transition'] = False
        for idx in transitions:
            df.loc[idx, 'is_transition'] = True
        
        return df

