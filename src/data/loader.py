"""
Data Loading and Extraction Module

Production-grade data loader for the EdTech Student Success Pipeline.
Handles data extraction from various formats and validates data integrity.

Author: Alyona Carolina Ivanova Araujo
License: MIT
"""

# --- Standard library imports ---
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# --- Third-party libraries ---
# Data manipulation and analysis
import pandas as pd

# Database and query
import duckdb

logger = logging.getLogger(__name__)


class DataLoader:
    """Production-grade data loader with validation and error handling."""
    
    def __init__(self, data_path: Path):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to data directory
            
        Raises:
            ValueError: If data path doesn't exist
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {data_path}")
        logger.info(f"DataLoader initialized with path: {data_path}")
    
    def load_student_data(
        self,
        filename: str,
        format: str = "parquet",
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Load student data from file.
        
        Args:
            filename: Name of data file
            format: File format ('parquet', 'csv', 'xlsx')
            validate: Whether to validate data integrity
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If data validation fails
        """
        file_path = self.data_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            if format == "parquet":
                df = pd.read_parquet(file_path)
            elif format == "csv":
                df = pd.read_csv(file_path)
            elif format == "xlsx":
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Loaded {len(df)} rows from {filename}")
            
            if validate:
                self._validate_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {filename}: {e}")
            raise
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate data integrity.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Check for essential columns
        required_columns = ['student_id', 'course_id']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        logger.info(f"Data validation passed. Shape: {df.shape}")
    
    def query_duckdb(self, query: str) -> pd.DataFrame:
        """
        Execute DuckDB SQL query on data.
        
        Args:
            query: SQL query string
            
        Returns:
            Query result as DataFrame
        """
        try:
            conn = duckdb.connect(':memory:')
            result = conn.execute(query).fetch_df()
            logger.info(f"DuckDB query executed successfully")
            return result
        except Exception as e:
            logger.error(f"DuckDB query error: {e}")
            raise


class DataAggregator:
    """Aggregate data across multiple sources."""
    
    @staticmethod
    def aggregate_student_metrics(
        engagement: pd.DataFrame,
        performance: pd.DataFrame,
        progress: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate multiple data sources into unified student metrics.
        
        Args:
            engagement: Student engagement data
            performance: Quiz/assessment performance data
            progress: Course progress data
            
        Returns:
            Aggregated metrics DataFrame
        """
        # Merge on student_id and course_id
        merged = engagement.merge(
            performance,
            on=['student_id', 'course_id'],
            how='inner'
        ).merge(
            progress,
            on=['student_id', 'course_id'],
            how='inner'
        )
        
        logger.info(f"Aggregated data: {len(merged)} records")
        return merged
