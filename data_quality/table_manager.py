#!/usr/bin/env python3
"""
Table Manager - Manages database tables for imputed data.

Responsibilities:
- Create new imputed tables with _imputed suffix
- Add tracking columns for imputation metadata
- Preserve original tables
- Manage table schema
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
from sqlalchemy import text


logger = logging.getLogger(__name__)


class TableManager:
    """Manages database tables for imputed scoreboard data."""
    
    def __init__(self, db_utility):
        """
        Initialize table manager.
        
        Args:
            db_utility: DatabaseUtility instance
        """
        self.db_utility = db_utility
        # Ensure connection is established
        self.db_utility._ensure_connection()
    
    def create_imputed_table(self, original_table_name: str) -> str:
        """
        Create a minimal imputed table with ONLY essential fields + tracking.
        
        Args:
            original_table_name: Name of the original table
            
        Returns:
            Name of the new imputed table
        """
        imputed_table_name = f"{original_table_name}_imputed"
        
        logger.info(f"Creating minimal imputed table: {imputed_table_name}")
        
        try:
            # Check if original table exists
            if not self._table_exists(original_table_name):
                raise ValueError(f"Original table '{original_table_name}' does not exist")
            
            # Drop imputed table if it already exists
            if self._table_exists(imputed_table_name):
                logger.warning(f"Imputed table '{imputed_table_name}' already exists, dropping...")
                self._drop_table(imputed_table_name)
            
            # Create minimal table with ONLY essential fields
            # No copying from original - we'll insert only what we need
            create_query = f"""
            CREATE TABLE {imputed_table_name} (
                frame_number INTEGER PRIMARY KEY,
                frame_path TEXT,
                
                -- Corrected essential fields
                game_time VARCHAR(20),
                period VARCHAR(10),
                play_clock VARCHAR(20),
                
                -- Imputation flags
                imputed_game_time BOOLEAN DEFAULT FALSE,
                imputed_period BOOLEAN DEFAULT FALSE,
                imputed_play_clock BOOLEAN DEFAULT FALSE,
                
                -- Original values (before correction)
                original_game_time VARCHAR(20),
                original_period VARCHAR(10),
                original_play_clock VARCHAR(20),
                
                -- Imputation metadata
                imputation_confidence DECIMAL(3,2),
                imputation_method VARCHAR(100),
                imputation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            with self.db_utility.get_connection() as conn:
                conn.execute(text(create_query))
                conn.commit()
            
            logger.info(f"Created minimal imputed table with 14 columns (essential fields only)")
            
            # No need to add tracking columns - already in schema
            
            logger.info(f"Successfully created imputed table: {imputed_table_name}")
            return imputed_table_name
            
        except Exception as e:
            logger.error(f"Failed to create imputed table: {e}")
            raise
    
    def _add_tracking_columns(self, table_name: str):
        """Add imputation tracking columns to table."""
        logger.info(f"Adding tracking columns to {table_name}")
        
        # Only add columns that will actually be stored
        # Exclude temporary analysis columns (anomaly detection flags)
        tracking_columns = [
            # Imputation flags
            ("imputed_game_time", "BOOLEAN DEFAULT FALSE"),
            ("imputed_period", "BOOLEAN DEFAULT FALSE"),
            ("imputed_play_clock", "BOOLEAN DEFAULT FALSE"),
            # Original values
            ("original_game_time", "VARCHAR(20)"),
            ("original_period", "VARCHAR(10)"),
            ("original_play_clock", "VARCHAR(20)"),
            # Imputation metadata
            ("imputation_confidence", "DECIMAL(3,2)"),
            ("imputation_method", "VARCHAR(100)"),
            ("imputation_timestamp", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
        ]
        
        # Note: Anomaly detection columns (game_time_anomaly, etc.) are NOT added
        # They are temporary analysis columns dropped before writing
        
        try:
            with self.db_utility.get_connection() as conn:
                for col_name, col_type in tracking_columns:
                    # Check if column already exists
                    if not self._column_exists(table_name, col_name):
                        alter_query = f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type};"
                        conn.execute(text(alter_query))
                        logger.debug(f"Added column: {col_name}")
                
                conn.commit()
            
            logger.info("Successfully added tracking columns")
            
        except Exception as e:
            logger.error(f"Failed to add tracking columns: {e}")
            raise
    
    def write_imputed_data(self, df: pd.DataFrame, table_name: str):
        """
        Write ONLY essential fields + tracking data to minimal imputed table.
        
        Args:
            df: DataFrame with imputed data
            table_name: Name of the imputed table
        """
        logger.info(f"Writing {len(df)} rows to minimal imputed table: {table_name}")
        
        try:
            # Add imputation timestamp
            df['imputation_timestamp'] = datetime.now()
            
            # Select ONLY the columns we need for the minimal table
            essential_columns = [
                'frame_number',
                'frame_path',
                # Corrected values
                'game_time',
                'period',
                'play_clock',
                # Imputation flags
                'imputed_game_time',
                'imputed_period',
                'imputed_play_clock',
                # Original values
                'original_game_time',
                'original_period',
                'original_play_clock',
                # Metadata
                'imputation_confidence',
                'imputation_method',
                'imputation_timestamp'
            ]
            
            # Ensure all required columns exist; if not, create with null values
            missing_columns = [col for col in essential_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing expected columns in imputed dataframe: {missing_columns}. Filling with NULL values.")
                for col in missing_columns:
                    df[col] = None
            
            # Keep only essential columns
            df_minimal = df[essential_columns].copy()
            
            logger.info(f"Created minimal dataset with {len(essential_columns)} columns (all other columns excluded)")
            
            # Truncate table first
            with self.db_utility.get_connection() as conn:
                conn.execute(text(f"TRUNCATE TABLE {table_name};"))
                conn.commit()
            
            # Write data - much faster now with only 14 columns!
            df_minimal.to_sql(
                table_name,
                self.db_utility._engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=500  # Can use larger chunks now (14 cols × 500 rows = 7,000 params)
            )
            
            logger.info(f"Successfully wrote data to {table_name}")
            
        except Exception as e:
            logger.error(f"Failed to write imputed data: {e}")
            raise
    
    def _table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        try:
            query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = :table_name
            );
            """
            with self.db_utility.get_connection() as conn:
                result = conn.execute(text(query), {"table_name": table_name})
                return result.scalar()
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")
            return False
    
    def _column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if column exists in table."""
        try:
            query = """
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = :table_name 
                AND column_name = :column_name
            );
            """
            with self.db_utility.get_connection() as conn:
                result = conn.execute(text(query), {
                    "table_name": table_name,
                    "column_name": column_name
                })
                return result.scalar()
        except Exception as e:
            logger.error(f"Error checking column existence: {e}")
            return False
    
    def _drop_table(self, table_name: str):
        """Drop table."""
        try:
            with self.db_utility.get_connection() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name};"))
                conn.commit()
            logger.info(f"Dropped table: {table_name}")
        except Exception as e:
            logger.error(f"Error dropping table: {e}")
            raise
    
    def get_imputed_table_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Get statistics about imputed table.
        
        Returns:
            Dictionary with statistics
        """
        try:
            query = f"""
            SELECT 
                COUNT(*) as total_rows,
                SUM(CASE WHEN imputed_game_time THEN 1 ELSE 0 END) as imputed_game_time_count,
                SUM(CASE WHEN imputed_period THEN 1 ELSE 0 END) as imputed_period_count,
                SUM(CASE WHEN imputed_play_clock THEN 1 ELSE 0 END) as imputed_play_clock_count,
                AVG(imputation_confidence) as avg_confidence
            FROM {table_name}
            WHERE imputed_game_time OR imputed_period OR imputed_play_clock;
            """
            
            with self.db_utility.get_connection() as conn:
                result = conn.execute(text(query))
                row = result.fetchone()
                
                if row:
                    return {
                        'total_rows': row[0],
                        'imputed_game_time': row[1] or 0,
                        'imputed_period': row[2] or 0,
                        'imputed_play_clock': row[3] or 0,
                        'avg_confidence': float(row[4]) if row[4] else 0.0
                    }
                
                return {}
                
        except Exception as e:
            logger.error(f"Error getting table stats: {e}")
            return {}

