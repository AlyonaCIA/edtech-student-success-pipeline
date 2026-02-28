"""Data Loading and Processing Module

This module handles all data operations for the EdTech Student Success Pipeline.
Provides production-grade data loading, validation, and processing capabilities.

Classes:
    DataLoader - Load data from various sources with validation
    DataAggregator - Aggregate and process student data

Usage:
    from src.data.loader import DataLoader
    
    loader = DataLoader(Path("data/raw"))
    df = loader.load_student_data()
"""

__all__ = ["DataLoader", "DataAggregator"]
