#!/usr/bin/env python3
"""
Quick comparison of data leakage impact
"""
import sys
import os
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import pandas as pd

# Load data directly
raw_data_file = "/workspace/412ca6282fd7/datasets/c99d9bc33649/c99d9bc33649_50.csv"
data = pd.read_csv(raw_data_file)

X = data.drop("Diabetes_binary", axis=1)
y = data["Diabetes_binary"]

print("=== DATA LEAKAGE COMPARISON ===\n")

# Method 1: WITH DATA LEAKAGE (original approach)
print("1. WITH DATA LEAKAGE (original method):")
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X_resampled, y_resampled)  # LEAKAGE HERE!

clf = RandomForestClassifier(random_state=42, n_estimators=50)
scores_leakage = cross_val_score(clf, X_selected, y_resampled, cv=5)

print(f"   Mean CV Score: {scores_leakage.mean():.4f} ± {scores_leakage.std():.4f}")
print(f"   Individual scores: {scores_leakage}")

# Method 2: NO DATA LEAKAGE (correct approach)
print("\n2. NO DATA LEAKAGE (correct method):")
pipeline = ImbPipeline([
    ('sampler', RandomUnderSampler(random_state=42)),
    ('selector', SelectKBest(f_classif, k=10)),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=50))
])

scores_no_leakage = cross_val_score(pipeline, X, y, cv=5)

print(f"   Mean CV Score: {scores_no_leakage.mean():.4f} ± {scores_no_leakage.std():.4f}")
print(f"   Individual scores: {scores_no_leakage}")

# Analysis
print(f"\n=== ANALYSIS ===")
print(f"Difference in mean score: {scores_leakage.mean() - scores_no_leakage.mean():.4f}")
print(f"Leakage leads to {'HIGHER' if scores_leakage.mean() > scores_no_leakage.mean() else 'LOWER'} apparent performance")
print(f"Standard deviation difference: {abs(scores_leakage.std() - scores_no_leakage.std()):.4f}")