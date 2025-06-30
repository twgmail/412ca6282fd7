"""
Diabetes Prediction Pipeline

This script implements a machine learning pipeline for diabetes prediction using:
- Random undersampling for class balancing
- Feature selection with SelectKBest
- Random Forest classification with cross-validation
"""

import sys
import os
import logging
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'dataset_path': 'datasets/c99d9bc33649/c99d9bc33649_50.csv',
    'target_column': 'Diabetes_binary',
    'n_features': 10,
    'cv_folds': 5,
    'random_state': 42,
    'rf_n_estimators': 50,  # Reduced for faster execution
    'test_size': 0.2
}


def setup_paths() -> Path:
    """Setup and return project root path."""
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    sys.path.append(str(parent_dir))
    
    try:
        from utils import get_project_root
        return get_project_root()
    except ImportError:
        logger.warning("Could not import utils.get_project_root, using current directory")
        return Path.cwd()


def load_and_validate_data(data_path: Path) -> pd.DataFrame:
    """
    Load and validate the dataset.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        Loaded and validated DataFrame
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If required columns are missing
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    
    # Validate required columns
    if CONFIG['target_column'] not in data.columns:
        raise ValueError(f"Target column '{CONFIG['target_column']}' not found in dataset")
    
    # Check for missing values
    missing_values = data.isnull().sum().sum()
    if missing_values > 0:
        logger.warning(f"Dataset contains {missing_values} missing values")
    
    logger.info(f"Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
    logger.info(f"Target distribution:\n{data[CONFIG['target_column']].value_counts()}")
    
    return data


def prepare_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data by separating features and target.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Tuple of (features, target)
    """
    # Separate features and target
    X = data.drop(CONFIG['target_column'], axis=1)
    y = data[CONFIG['target_column']]
    
    class_counts = dict(zip(*np.unique(y, return_counts=True)))
    logger.info(f"Class distribution: {class_counts}")
    
    # Check if data is balanced (within 5% tolerance)
    counts = list(class_counts.values())
    balance_ratio = min(counts) / max(counts)
    
    if balance_ratio > 0.95:
        logger.info("Data is well-balanced (no undersampling needed in pipeline)")
    else:
        logger.info("Data is imbalanced (undersampling will be applied in pipeline)")
    
    return X, y


def create_pipeline(balance_ratio: float) -> ImbPipeline:
    """
    Create ML pipeline with proper cross-validation to prevent data leakage.
    
    Args:
        balance_ratio: Ratio of minority to majority class
        
    Returns:
        Configured pipeline
    """
    steps = []
    
    # Only add undersampling if data is imbalanced
    if balance_ratio <= 0.95:
        logger.info("Adding undersampling to pipeline")
        steps.append(('sampler', RandomUnderSampler(random_state=CONFIG['random_state'])))
    else:
        logger.info("Skipping undersampling (data is balanced)")
    
    # Add feature selection and classifier
    steps.extend([
        ('selector', SelectKBest(f_classif, k=CONFIG['n_features'])),
        ('classifier', RandomForestClassifier(
            n_estimators=CONFIG['rf_n_estimators'],
            random_state=CONFIG['random_state'],
            n_jobs=-1
        ))
    ])
    
    return ImbPipeline(steps)


def train_and_evaluate_model(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Train and evaluate the Random Forest model using proper cross-validation.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        Dictionary containing evaluation results
    """
    logger.info("Creating ML pipeline...")
    
    # Check class balance
    class_counts = dict(zip(*np.unique(y, return_counts=True)))
    counts = list(class_counts.values())
    balance_ratio = min(counts) / max(counts)
    
    # Create pipeline that prevents data leakage
    pipeline = create_pipeline(balance_ratio)
    
    logger.info("Performing cross-validation (no data leakage)...")
    
    # Perform stratified cross-validation with the pipeline
    cv = StratifiedKFold(n_splits=CONFIG['cv_folds'], shuffle=True, random_state=CONFIG['random_state'])
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    
    # Train final model for feature analysis
    logger.info("Training final model for feature importance analysis...")
    pipeline.fit(X, y)
    
    # Extract feature names from the pipeline
    if hasattr(pipeline.named_steps['selector'], 'get_support'):
        selected_features = X.columns[pipeline.named_steps['selector'].get_support()].tolist()
    else:
        selected_features = list(X.columns)
    
    results = {
        'cv_scores': scores,
        'mean_cv_score': scores.mean(),
        'std_cv_score': scores.std(),
        'pipeline': pipeline,
        'feature_names': selected_features
    }
    
    return results


def print_results(results: dict) -> None:
    """Print formatted results."""
    print("\n" + "="*50)
    print("DIABETES PREDICTION MODEL RESULTS (NO DATA LEAKAGE)")
    print("="*50)
    
    print(f"\nCross-Validation Results:")
    print(f"  Individual fold scores: {results['cv_scores']}")
    print(f"  Mean accuracy: {results['mean_cv_score']:.4f}")
    print(f"  Standard deviation: {results['std_cv_score']:.4f}")
    print(f"  95% Confidence interval: [{results['mean_cv_score'] - 2*results['std_cv_score']:.4f}, "
          f"{results['mean_cv_score'] + 2*results['std_cv_score']:.4f}]")
    
    # Feature importance from final trained pipeline
    pipeline = results['pipeline']
    if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
        print(f"\nTop 5 Feature Importances:")
        importances = pipeline.named_steps['classifier'].feature_importances_
        feature_names = results['feature_names']
        indices = np.argsort(importances)[::-1][:5]
        for i, idx in enumerate(indices, 1):
            print(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    print(f"\nSelected Features ({len(results['feature_names'])}):")
    print(f"  {', '.join(results['feature_names'])}")


def main() -> None:
    """Main execution function."""
    try:
        # Setup paths
        project_root = setup_paths()
        data_path = project_root / CONFIG['dataset_path']
        
        # Load and validate data
        data = load_and_validate_data(data_path)
        
        # Prepare data (separate features and target)
        X, y = prepare_data(data)
        
        # Train and evaluate model with proper pipeline (no data leakage)
        results = train_and_evaluate_model(X, y)
        
        # Print results
        print_results(results)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()