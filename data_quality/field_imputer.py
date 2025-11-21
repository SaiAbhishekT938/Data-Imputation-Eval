#!/usr/bin/env python3
"""
Field Imputer - Smart imputation for missing scoreboard fields.

Handles all combinations of missing fields:
- If at least 1 field exists → impute the rest
- Uses temporal neighbors and period context
- Validates imputation confidence
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class FieldImputer:
    """Smart imputation for missing scoreboard fields."""
    
    def __init__(self, window_size: int = 5, min_confidence: float = 0.6, max_consecutive_missing: int = 6):
        """
        Initialize field imputer.
        
        Args:
            window_size: Number of neighboring frames to use for imputation (default: 5 = ±1.7s at 3 FPS)
            min_confidence: Minimum confidence threshold for imputation (default: 0.75 = 75%)
            max_consecutive_missing: Max consecutive frames with missing fields to impute (default: 5 = 1.7s at 3 FPS)
                                     Longer sequences likely indicate commercials/timeouts/pauses
        """
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.max_consecutive_missing = max_consecutive_missing
        
        logger.info(f"FieldImputer initialized: window_size={window_size} (±{window_size/3:.1f}s), "
                   f"min_confidence={min_confidence} ({min_confidence*100:.0f}%), "
                   f"max_consecutive_missing={max_consecutive_missing} (~{max_consecutive_missing/3:.1f}s)")
    
    def impute_missing_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing scoreboard fields with consecutive missing detection.
        
        Args:
            df: DataFrame with scoreboard data
            
        Returns:
            DataFrame with imputed values and tracking columns
        """
        logger.info(f"Starting field imputation on {len(df)} frames")
        
        # Add tracking columns
        df['imputed_game_time'] = False
        df['imputed_period'] = False
        df['imputed_play_clock'] = False
        df['imputation_confidence'] = 0.0
        df['imputation_method'] = ''
        
        # Store original values before imputation
        df['original_game_time'] = df['game_time']
        df['original_period'] = df['period']
        df['original_play_clock'] = df['play_clock']
        
        # 🎯 STEP 0: Apply anomaly-driven corrections before standard imputation
        df = self._apply_anomaly_corrections(df)
        
        # 🎯 STEP 1: Identify consecutive missing field sequences
        consecutive_missing_sequences = self._identify_consecutive_missing_sequences(df)
        
        # 🎯 STEP 2: Mark frames to skip (in long consecutive missing sequences)
        skip_indices = set()
        for start_idx, end_idx, length in consecutive_missing_sequences:
            if length > self.max_consecutive_missing:
                skip_indices.update(range(start_idx, end_idx + 1))
                logger.info(f"Skipping imputation for frames {start_idx}-{end_idx} "
                           f"({length} consecutive missing - likely commercial/pause)")
        
        logger.info(f"Identified {len(consecutive_missing_sequences)} consecutive missing sequences")
        logger.info(f"Skipping {len(skip_indices)} frames in long sequences (>{self.max_consecutive_missing} consecutive)")
        
        # 🎯 STEP 3: Impute for each frame (except skipped ones)
        imputed_count = 0
        skipped_count = 0
        low_confidence_skipped = 0
        
        for idx in df.index:
            # Skip if in a long consecutive missing sequence
            if idx in skip_indices:
                skipped_count += 1
                continue
            
            # Check if frame has at least one field
            if self._has_at_least_one_field(df.loc[idx]):
                result = self._impute_frame(df, idx)
                if result == 'imputed':
                    imputed_count += 1
                elif result == 'low_confidence':
                    low_confidence_skipped += 1
        
        logger.info(f"✅ Imputed fields for {imputed_count} frames")
        logger.info(f"⏭️  Skipped {skipped_count} frames (long consecutive missing sequences)")
        logger.info(f"📊 Skipped {low_confidence_skipped} frames (low confidence < {self.min_confidence})")
        
        return df

    def _apply_anomaly_corrections(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply corrections suggested by anomaly detection.
        
        - If a suggestion exists, apply it immediately with high confidence.
        - If no suggestion exists, clear the field so standard imputation can fill it.
        """
        anomaly_columns_present = {
            'game_time': 'game_time_anomaly' in df.columns,
            'period': 'period_anomaly' in df.columns,
            'play_clock': 'play_clock_anomaly' in df.columns,
        }
        
        def _append_method(existing: str, new: str) -> str:
            if not existing:
                return new
            if new in existing.split(','):
                return existing
            return f"{existing},{new}"
        
        for idx in df.index:
            # Game Time
            if anomaly_columns_present['game_time'] and bool(df.loc[idx, 'game_time_anomaly']):
                suggestion = df.loc[idx, 'suggested_game_time']
                if suggestion:
                    df.loc[idx, 'game_time'] = suggestion
                    df.loc[idx, 'imputed_game_time'] = True
                    df.loc[idx, 'imputation_confidence'] = max(
                        df.loc[idx, 'imputation_confidence'],
                        df.loc[idx, 'anomaly_confidence'] if 'anomaly_confidence' in df.columns else 0.95
                    )
                    df.loc[idx, 'imputation_method'] = _append_method(df.loc[idx, 'imputation_method'], 'anomaly_correction')
                else:
                    # Treat as missing to allow standard imputation
                    df.loc[idx, 'game_time'] = ''
            
            # Period
            if anomaly_columns_present['period'] and bool(df.loc[idx, 'period_anomaly']):
                suggestion = df.loc[idx, 'suggested_period']
                if suggestion:
                    df.loc[idx, 'period'] = suggestion
                    df.loc[idx, 'imputed_period'] = True
                    df.loc[idx, 'imputation_confidence'] = max(
                        df.loc[idx, 'imputation_confidence'],
                        df.loc[idx, 'anomaly_confidence'] if 'anomaly_confidence' in df.columns else 0.90
                    )
                    df.loc[idx, 'imputation_method'] = _append_method(df.loc[idx, 'imputation_method'], 'anomaly_correction')
                else:
                    df.loc[idx, 'period'] = ''
            
            # Play Clock
            if anomaly_columns_present['play_clock'] and bool(df.loc[idx, 'play_clock_anomaly']):
                suggestion = df.loc[idx, 'suggested_play_clock']
                if suggestion:
                    df.loc[idx, 'play_clock'] = suggestion
                    df.loc[idx, 'imputed_play_clock'] = True
                    df.loc[idx, 'imputation_confidence'] = max(
                        df.loc[idx, 'imputation_confidence'],
                        df.loc[idx, 'anomaly_confidence'] if 'anomaly_confidence' in df.columns else 0.85
                    )
                    df.loc[idx, 'imputation_method'] = _append_method(df.loc[idx, 'imputation_method'], 'anomaly_correction')
                else:
                    df.loc[idx, 'play_clock'] = ''
        
        return df
    
    def _identify_consecutive_missing_sequences(self, df: pd.DataFrame) -> List[tuple]:
        """
        Identify sequences of consecutive frames with ALL essential fields missing.
        
        Returns:
            List of tuples: (start_idx, end_idx, length)
        """
        sequences = []
        in_sequence = False
        sequence_start = None
        
        for idx in df.index:
            all_missing = not self._has_at_least_one_field(df.loc[idx])
            
            if all_missing:
                if not in_sequence:
                    # Start new sequence
                    in_sequence = True
                    sequence_start = idx
            else:
                if in_sequence:
                    # End current sequence
                    sequence_end = idx - 1
                    length = sequence_end - sequence_start + 1
                    sequences.append((sequence_start, sequence_end, length))
                    in_sequence = False
                    sequence_start = None
        
        # Handle sequence at end of data
        if in_sequence and sequence_start is not None:
            sequence_end = df.index[-1]
            length = sequence_end - sequence_start + 1
            sequences.append((sequence_start, sequence_end, length))
        
        return sequences
    
    def _has_at_least_one_field(self, row: pd.Series) -> bool:
        """
        Check if row has at least one essential field for imputation.
        
        ✅ NEW CONSTRAINT: Only impute if period OR play_clock exists
        (game_time alone is not enough - might be OCR error)
        """
        has_period = not pd.isna(row['period']) and row['period'] != ''
        has_play_clock = not pd.isna(row['play_clock']) and row['play_clock'] != ''
        
        # ✅ STRICTER: Only period or play_clock can trigger imputation
        # If only game_time exists → skip (might be misread, no context confidence)
        return has_period or has_play_clock
    
    def _impute_frame(self, df: pd.DataFrame, idx: int) -> str:
        """
        Impute missing fields for a single frame with confidence enforcement.
        
        Returns:
            'imputed' if any field was imputed successfully
            'low_confidence' if skipped due to low confidence
            'no_imputation' if no imputation was needed
        """
        imputed_any = False
        low_confidence_attempts = 0
        
        # Get context
        period = df.loc[idx, 'period']
        game_time = df.loc[idx, 'game_time']
        play_clock = df.loc[idx, 'play_clock']
        
        # Impute game_time if missing
        if pd.isna(game_time) or game_time == '':
            imputed_value, confidence, method = self._impute_game_time(df, idx, period)
            if imputed_value:
                if confidence >= self.min_confidence:
                    df.loc[idx, 'game_time'] = imputed_value
                    df.loc[idx, 'imputed_game_time'] = True
                    df.loc[idx, 'imputation_confidence'] = confidence
                    df.loc[idx, 'imputation_method'] = method
                    imputed_any = True
                    logger.debug(f"Frame {df.loc[idx, 'frame_number']}: ✅ Imputed game_time='{imputed_value}' "
                               f"(method: {method}, confidence: {confidence:.2f})")
                else:
                    low_confidence_attempts += 1
                    logger.debug(f"Frame {df.loc[idx, 'frame_number']}: ⏭️ Skipped game_time imputation "
                               f"(confidence {confidence:.2f} < threshold {self.min_confidence})")
        
        # Impute period if missing
        if pd.isna(period) or period == '':
            imputed_value, confidence, method = self._impute_period(df, idx)
            if imputed_value:
                if confidence >= self.min_confidence:
                    df.loc[idx, 'period'] = imputed_value
                    df.loc[idx, 'imputed_period'] = True
                    df.loc[idx, 'imputation_confidence'] = max(df.loc[idx, 'imputation_confidence'], confidence)
                    df.loc[idx, 'imputation_method'] += f",{method}" if df.loc[idx, 'imputation_method'] else method
                    imputed_any = True
                    logger.debug(f"Frame {df.loc[idx, 'frame_number']}: ✅ Imputed period='{imputed_value}' "
                               f"(method: {method}, confidence: {confidence:.2f})")
                else:
                    low_confidence_attempts += 1
                    logger.debug(f"Frame {df.loc[idx, 'frame_number']}: ⏭️ Skipped period imputation "
                               f"(confidence {confidence:.2f} < threshold {self.min_confidence})")
        
        # Impute play_clock if missing
        if pd.isna(play_clock) or play_clock == '':
            imputed_value, confidence, method = self._impute_play_clock(df, idx, period)
            if imputed_value:
                if confidence >= self.min_confidence:
                    df.loc[idx, 'play_clock'] = imputed_value
                    df.loc[idx, 'imputed_play_clock'] = True
                    df.loc[idx, 'imputation_confidence'] = max(df.loc[idx, 'imputation_confidence'], confidence)
                    df.loc[idx, 'imputation_method'] += f",{method}" if df.loc[idx, 'imputation_method'] else method
                    imputed_any = True
                    logger.debug(f"Frame {df.loc[idx, 'frame_number']}: ✅ Imputed play_clock='{imputed_value}' "
                               f"(method: {method}, confidence: {confidence:.2f})")
                else:
                    low_confidence_attempts += 1
                    logger.debug(f"Frame {df.loc[idx, 'frame_number']}: ⏭️ Skipped play_clock imputation "
                               f"(confidence {confidence:.2f} < threshold {self.min_confidence})")
        
        # Return status
        if imputed_any:
            return 'imputed'
        elif low_confidence_attempts > 0:
            return 'low_confidence'
        else:
            return 'no_imputation'
    
    def _impute_game_time(self, df: pd.DataFrame, idx: int, period: Optional[str]) -> tuple:
        """
        Impute game_time using TRUE linear interpolation.
        
        Game time DECREASES: 15:00 → 14:59 → 14:58 → ... → 0:00
        At 3 FPS, approximately 1 second decreases per frame (real-time).
        
        Returns:
            (imputed_value, confidence, method)
        """
        # Get neighbors
        neighbors = self._get_neighbors(df, idx, period)
        
        if len(neighbors) == 0:
            return None, 0.0, ''
        
        # Extract game_times from neighbors with seconds conversion
        neighbor_times = []
        for n_idx in neighbors:
            t = df.loc[n_idx, 'game_time']
            if t and not pd.isna(t):
                t_seconds = self._time_to_seconds(t)
                if t_seconds is not None:
                    neighbor_times.append((n_idx, t, t_seconds))
        
        if len(neighbor_times) == 0:
            return None, 0.0, ''
        
        # Find closest before and after
        closest_before = None
        closest_after = None
        
        for n_idx, t_str, t_sec in neighbor_times:
            if n_idx < idx:
                if closest_before is None or n_idx > closest_before[0]:
                    closest_before = (n_idx, t_str, t_sec)
            elif n_idx > idx:
                if closest_after is None or n_idx < closest_after[0]:
                    closest_after = (n_idx, t_str, t_sec)
        
        index_list = list(df.index)
        frames_per_step = 2  # Countdown cadence: each second appears twice

        def clamp_to_range(seconds_value: int) -> int:
            return max(0, min(15 * 60, seconds_value))

        def find_anchor_position(start_pos: int, step: int) -> Optional[Tuple[int, int]]:
            pos = start_pos + step
            while 0 <= pos < len(index_list):
                label = index_list[pos]
                if df.loc[label, 'imputed_game_time']:
                    pos += step
                    continue
                value = df.loc[label, 'game_time']
                if value and not pd.isna(value):
                    seconds = self._time_to_seconds(value)
                    if seconds is not None:
                        return pos, seconds
                pos += step
            return None

        current_pos = index_list.index(idx)
        before_anchor = find_anchor_position(current_pos, -1)
        after_anchor = find_anchor_position(current_pos, 1)

        if not before_anchor and not after_anchor:
            return None, 0.0, ''

        # Ignore zero anchors when the opposite side has valid non-zero data
        if before_anchor and before_anchor[1] == 0 and after_anchor and after_anchor[1] > 0:
            before_anchor = None
        if after_anchor and after_anchor[1] == 0 and before_anchor and before_anchor[1] > 0:
            after_anchor = None

        if not before_anchor and not after_anchor:
            return None, 0.0, ''

        def count_repeats(anchor_pos: int, step: int, target_seconds: int) -> int:
            count = 0
            pos = anchor_pos
            while 0 <= pos < len(index_list):
                label = index_list[pos]
                value = df.loc[label, 'game_time']
                if value is None or pd.isna(value):
                    break
                seconds = self._time_to_seconds(value)
                if seconds != target_seconds:
                    break
                count += 1
                pos += step
            return count

        def steps_from_before(distance: int) -> int:
            if distance <= 0:
                return 0
            return (distance + frames_per_step - 1) // frames_per_step

        def steps_from_after(distance: int) -> int:
            if distance <= 0:
                return 0
            return (distance + frames_per_step - 1) // frames_per_step

        if before_anchor and after_anchor:
            before_pos, before_seconds = before_anchor
            after_pos, after_seconds = after_anchor

            frames_from_before = current_pos - before_pos
            total_distance = after_pos - before_pos
            diff_seconds = before_seconds - after_seconds
            existing_left = count_repeats(before_pos, -1, before_seconds)
            existing_right = count_repeats(after_pos, 1, after_seconds)

            steps_left = max(0, (frames_from_before + frames_per_step - 1) // frames_per_step)
            candidate_left = clamp_to_range(before_seconds - steps_left)

            frames_to_after = after_pos - current_pos
            steps_right = max(0, (frames_to_after + frames_per_step - 1) // frames_per_step)
            candidate_right = clamp_to_range(after_seconds + steps_right)

            if diff_seconds > 0:
                candidate_left = max(after_seconds, min(before_seconds, candidate_left))
                candidate_right = max(after_seconds, min(before_seconds, candidate_right))

            interpolated_seconds = max(after_seconds, min(before_seconds, min(candidate_left, candidate_right)))

            interpolated_seconds = clamp_to_range(int(round(interpolated_seconds)))
            lower_bound = after_seconds if diff_seconds > 0 and after_seconds is not None else 0
            prev_repeat = self._count_prev_same_game_time(df, idx, interpolated_seconds)
            if prev_repeat >= frames_per_step:
                interpolated_seconds = clamp_to_range(max(0, interpolated_seconds - 1))

            interpolated_str = self._seconds_to_time(interpolated_seconds)

            return interpolated_str, 0.90, 'linear_interpolation'

        if before_anchor:
            before_pos, before_seconds = before_anchor
            repeat_count = count_repeats(before_pos, -1, before_seconds)
            if repeat_count > 5:
                return None, 0.0, ''
            frames_from_before = current_pos - before_pos
            steps = max(0, (frames_from_before + frames_per_step - 1) // frames_per_step)
            estimated_seconds = clamp_to_range(before_seconds - steps)
            prev_repeat = self._count_prev_same_game_time(df, idx, estimated_seconds)
            if prev_repeat >= frames_per_step:
                estimated_seconds = clamp_to_range(max(0, estimated_seconds - 1))

            estimated_str = self._seconds_to_time(estimated_seconds)
            confidence = 0.70 if frames_from_before <= frames_per_step * 3 else 0.65
            return estimated_str, confidence, 'backward_extrapolation'

        if after_anchor:
            after_pos, after_seconds = after_anchor
            repeat_count = count_repeats(after_pos, 1, after_seconds)
            if repeat_count > 5:
                return None, 0.0, ''
            frames_to_after = after_pos - current_pos
            steps = max(0, (frames_to_after + frames_per_step - 1) // frames_per_step)
            estimated_seconds = clamp_to_range(after_seconds + steps)
            prev_repeat = self._count_prev_same_game_time(df, idx, estimated_seconds)
            if prev_repeat >= frames_per_step:
                estimated_seconds = clamp_to_range(max(0, estimated_seconds - 1))

            estimated_str = self._seconds_to_time(estimated_seconds)
            confidence = 0.70 if frames_to_after <= frames_per_step * 3 else 0.65
            return estimated_str, confidence, 'forward_extrapolation'

        return None, 0.0, ''
    
    def _time_to_seconds(self, time_str: str) -> Optional[int]:
        """Convert time string (MM:SS) to seconds."""
        try:
            if ':' in str(time_str):
                parts = str(time_str).split(':')
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
        return f"{minutes:02d}:{secs:02d}"
    
    def _impute_period(self, df: pd.DataFrame, idx: int) -> tuple:
        """
        Impute period using mode from neighbors.
        
        Returns:
            (imputed_value, confidence, method)
        """
        # Get neighbors (don't filter by period since we're imputing it)
        window_start = max(0, idx - self.window_size)
        window_end = min(len(df), idx + self.window_size + 1)
        neighbors_df = df.iloc[window_start:window_end]
        
        # Get valid periods from neighbors
        neighbor_periods = []
        for n_idx in neighbors_df.index:
            if n_idx != idx:
                p = neighbors_df.loc[n_idx, 'period']
                if p and not pd.isna(p):
                    neighbor_periods.append(p)
        
        if len(neighbor_periods) == 0:
            return None, 0.0, ''
        
        # Use mode (most common)
        from collections import Counter
        period_counts = Counter(neighbor_periods)
        mode_period = period_counts.most_common(1)[0][0]
        mode_count = period_counts[mode_period]
        
        confidence = mode_count / len(neighbor_periods)
        method = 'mode_voting'
        
        return mode_period, confidence, method
    
    def _impute_play_clock(self, df: pd.DataFrame, idx: int, period: Optional[str]) -> tuple:
        """
        Impute play_clock using countdown pattern modeling.
        
        Play clock counts DOWN: 40 → 39 → 38 → ... → 0 (then resets to 40 or 25)
        At 3 FPS, approximately 1 second decreases per frame.
        
        Returns:
            (imputed_value, confidence, method)
        """
        # Get neighbors
        neighbors = self._get_neighbors(df, idx, period)
        
        if len(neighbors) == 0:
            return None, 0.0, ''
        
        # Get play_clocks from neighbors
        neighbor_clocks = []
        for n_idx in neighbors:
            pc = df.loc[n_idx, 'play_clock']
            if pc and not pd.isna(pc):
                try:
                    pc_int = int(str(pc).strip())
                    if 0 <= pc_int <= 40:  # Valid range
                        neighbor_clocks.append((n_idx, pc_int))
                except (ValueError, AttributeError):
                    pass
        
        if len(neighbor_clocks) == 0:
            return None, 0.0, ''
        
        # Find closest before and after
        closest_before = None
        closest_after = None
        
        for n_idx, pc_int in neighbor_clocks:
            if n_idx < idx:
                if closest_before is None or n_idx > closest_before[0]:
                    closest_before = (n_idx, pc_int)
            elif n_idx > idx:
                if closest_after is None or n_idx < closest_after[0]:
                    closest_after = (n_idx, pc_int)
        
        # Ignore zero anchors when the opposite side has valid non-zero data
        if closest_before and closest_before[1] == 0 and closest_after and closest_after[1] > 0:
            closest_before = None
        if closest_after and closest_after[1] == 0 and closest_before and closest_before[1] > 0:
            closest_after = None

        if not closest_before and not closest_after:
            return None, 0.0, ''

        # Helper to estimate average countdown rate (seconds per frame)
        def estimate_frames_per_decrement(default: int = 2) -> int:
            frames_needed = []
 
            # Look backwards for rate
            last_value = None
            last_idx = None
            for look_idx in range(idx - 1, max(df.index[0], idx - 15) - 1, -1):
                value = df.loc[look_idx, 'play_clock']
                if value is None or pd.isna(value):
                    continue
                try:
                    value_int = int(str(value).strip())
                except (ValueError, AttributeError):
                    continue
                if not (0 <= value_int <= 40):
                    continue
                if last_value is not None and value_int != last_value:
                    frame_distance = last_idx - look_idx
                    if frame_distance > 0:
                        diff = last_value - value_int
                        if diff > 0:
                            frames_per_step = max(1, round(frame_distance / diff))
                            frames_needed.append(frames_per_step)
                            break
                last_value = value_int
                last_idx = look_idx
 
            # Look forward for rate
            last_value = None
            last_idx = None
            for look_idx in range(idx + 1, min(df.index[-1], idx + 15) + 1):
                value = df.loc[look_idx, 'play_clock']
                if value is None or pd.isna(value):
                    continue
                try:
                    value_int = int(str(value).strip())
                except (ValueError, AttributeError):
                    continue
                if not (0 <= value_int <= 40):
                    continue
                if last_value is not None and value_int != last_value:
                    frame_distance = look_idx - last_idx
                    if frame_distance > 0:
                        diff = last_value - value_int
                        if diff > 0:
                            frames_per_step = max(1, round(frame_distance / diff))
                            frames_needed.append(frames_per_step)
                            break
                last_value = value_int
                last_idx = look_idx
 
            if frames_needed:
                frames_needed.sort()
                return max(1, min(4, frames_needed[len(frames_needed) // 2]))
            return default

        local_frames_per_step = estimate_frames_per_decrement()

        def count_repeats(label, step: int, target_value: int) -> int:
            count = 0
            pos = df.index.get_loc(label)
            while 0 <= pos < len(df.index):
                current_label = df.index[pos]
                value = df.loc[current_label, 'play_clock']
                if value is None or pd.isna(value):
                    break
                try:
                    value_int = int(str(value).strip())
                except (ValueError, AttributeError):
                    break
                if value_int != target_value:
                    break
                count += 1
                pos += step
            return count

        def steps_from_before(distance: int) -> int:
            if distance <= 0:
                return 0
            return (distance + frames_per_step - 1) // frames_per_step

        def steps_from_after(distance: int) -> int:
            if distance <= 0:
                return 0
            return (distance + frames_per_step - 1) // frames_per_step

        # Countdown modeling with dynamic rate
        if closest_before and closest_after:
            idx_before, clock_before = closest_before
            idx_after, clock_after = closest_after
 
            frame_distance_total = idx_after - idx_before
            if frame_distance_total <= 0:
                estimated_clock = clock_before
                frames_per_step = local_frames_per_step
                steps = 0
            else:
                diff = clock_before - clock_after
                if diff > 0 and frame_distance_total >= diff:
                    ratio = max(1, round(frame_distance_total / diff))
                else:
                    ratio = local_frames_per_step

                frames_per_step = max(1, min(local_frames_per_step, ratio))

                frames_from_before = idx - idx_before
                steps = frames_from_before // frames_per_step
                estimated_clock = clock_before - steps
 
            estimated_clock = max(0, min(40, int(estimated_clock)))
 
            confidence = 0.80 if closest_before[1] != closest_after[1] else 0.65
            method = 'countdown_interpolation'
 
            logger.debug(
                f"Frame {idx}: frames_per_step={frames_per_step}, steps={steps}, estimated_clock={estimated_clock}"
            )
            return str(estimated_clock), confidence, method
 
        # Backward extrapolation with dynamic rate
        if closest_before:
            idx_before, clock_before = closest_before
            if clock_before == 0 and not closest_after:
                return None, 0.0, ''
            repeat_count = count_repeats(idx_before, -1, clock_before)
            if repeat_count > 5:
                return None, 0.0, ''
            frame_distance = idx - idx_before
            frames_per_step = local_frames_per_step
            steps = steps_from_before(frame_distance)
            estimated_clock = clock_before - steps
            estimated_clock = max(0, min(40, int(estimated_clock)))
            prev_repeat = self._count_prev_same_play_clock(df, idx, estimated_clock)
            if prev_repeat >= frames_per_step:
                estimated_clock = max(0, estimated_clock - 1)
 
            confidence = 0.65 if frame_distance <= 4 else 0.60
            method = 'backward_countdown'
 
            logger.debug(
                f"Frame {idx}: Backward frames_per_step={frames_per_step}, steps={steps}, estimated_clock={estimated_clock}"
            )
            return str(estimated_clock), confidence, method
 
        elif closest_after:
            idx_after, clock_after = closest_after
            if clock_after == 0 and not closest_before:
                return None, 0.0, ''
            repeat_count = count_repeats(idx_after, 1, clock_after)
            if repeat_count > 5:
                return None, 0.0, ''
            frame_distance = idx_after - idx
            frames_per_step = local_frames_per_step
            steps = steps_from_after(frame_distance)
            estimated_clock = clock_after + steps
            estimated_clock = max(0, min(40, int(estimated_clock)))
            prev_repeat = self._count_prev_same_play_clock(df, idx, estimated_clock)
            if prev_repeat >= frames_per_step:
                estimated_clock = max(0, estimated_clock - 1)
 
            confidence = 0.65 if frame_distance <= 4 else 0.60
            method = 'forward_countdown'
 
            logger.debug(
                f"Frame {idx}: Forward frames_per_step={frames_per_step}, steps={steps}, estimated_clock={estimated_clock}"
            )
            return str(estimated_clock), confidence, method
 
        return None, 0.0, ''
    
    def _get_neighbors(self, df: pd.DataFrame, idx: int, period: Optional[str]) -> List[int]:
        """Get neighbor frame indices, optionally filtered by period."""
        window_start = max(0, idx - self.window_size)
        window_end = min(len(df), idx + self.window_size + 1)
        neighbors_df = df.iloc[window_start:window_end]
        
        # Filter by same period if available
        if period and not pd.isna(period):
            period_neighbors = neighbors_df[neighbors_df['period'] == period]
            if len(period_neighbors) >= 2:  # Need at least 2 neighbors
                return [i for i in period_neighbors.index if i != idx]
        
        # Otherwise use all neighbors
        return [i for i in neighbors_df.index if i != idx]

    def _count_prev_same_game_time(self, df: pd.DataFrame, idx: int, seconds_value: int) -> int:
        count = 0
        pos = df.index.get_loc(idx) - 1
        while pos >= 0:
            label = df.index[pos]
            value = df.loc[label, 'game_time']
            if value is None or pd.isna(value):
                break
            candidate = self._time_to_seconds(value)
            if candidate != seconds_value:
                break
            count += 1
            pos -= 1
        return count

    def _count_prev_same_play_clock(self, df: pd.DataFrame, idx: int, target_value: int) -> int:
        count = 0
        pos = df.index.get_loc(idx) - 1
        while pos >= 0:
            label = df.index[pos]
            value = df.loc[label, 'play_clock']
            if value is None or pd.isna(value):
                break
            try:
                candidate = int(str(value).strip())
            except (ValueError, AttributeError):
                break
            if candidate != target_value:
                break
            count += 1
            pos -= 1
        return count

