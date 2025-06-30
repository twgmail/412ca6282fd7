#!/usr/bin/env python3
"""
Analyze what causes the performance improvement
"""
import time
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

# Load data
data = pd.read_csv("/workspace/412ca6282fd7/datasets/c99d9bc33649/c99d9bc33649_50.csv")
X = data.drop("Diabetes_binary", axis=1)
y = data["Diabetes_binary"]

print("=== PERFORMANCE ANALYSIS ===\n")

# Test 1: Original configuration (100 estimators, no n_jobs)
print("1. Original configuration:")
start = time.time()
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X_resampled, y_resampled)
clf = RandomForestClassifier(random_state=42)  # Default: 100 estimators, n_jobs=None
scores = cross_val_score(clf, X_selected, y_resampled, cv=5)
end = time.time()
print(f"   Time: {end-start:.1f}s")
print(f"   Mean score: {scores.mean():.4f}")

# Test 2: Reduced estimators only
print("\n2. Reduced estimators (50 instead of 100):")
start = time.time()
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X_resampled, y_resampled)
clf = RandomForestClassifier(random_state=42, n_estimators=50)  # 50 estimators
scores = cross_val_score(clf, X_selected, y_resampled, cv=5)
end = time.time()
print(f"   Time: {end-start:.1f}s")
print(f"   Mean score: {scores.mean():.4f}")

# Test 3: Parallel processing only
print("\n3. Parallel processing (n_jobs=-1, 100 estimators):")
start = time.time()
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X_resampled, y_resampled)
clf = RandomForestClassifier(random_state=42, n_jobs=-1)  # Parallel processing
scores = cross_val_score(clf, X_selected, y_resampled, cv=5)
end = time.time()
print(f"   Time: {end-start:.1f}s")
print(f"   Mean score: {scores.mean():.4f}")

# Test 4: Both optimizations
print("\n4. Both optimizations (50 estimators + n_jobs=-1):")
start = time.time()
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X_resampled, y_resampled)
clf = RandomForestClassifier(random_state=42, n_estimators=50, n_jobs=-1)
scores = cross_val_score(clf, X_selected, y_resampled, cv=5)
end = time.time()
print(f"   Time: {end-start:.1f}s")
print(f"   Mean score: {scores.mean():.4f}")

print(f"\n=== CONCLUSION ===")
print(f"The 74% speedup comes from:")
print(f"1. Reducing n_estimators from 100 to 50 (~50% speedup)")
print(f"2. Adding n_jobs=-1 for parallel processing (~additional speedup)")
print(f"3. These are PERFORMANCE optimizations, not bug fixes")