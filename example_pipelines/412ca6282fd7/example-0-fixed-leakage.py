import sys
import os
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import pandas as pd

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(
    project_root, "datasets", "c99d9bc33649", "c99d9bc33649_50.csv"
)
data = pd.read_csv(raw_data_file)

# Separate features and target
X = data.drop("Diabetes_binary", axis=1)
y = data["Diabetes_binary"]

# CORRECT: Create pipeline that prevents data leakage
# Feature selection and model training happen inside CV
pipeline = ImbPipeline([
    ('sampler', RandomUnderSampler(random_state=42)),
    ('selector', SelectKBest(f_classif, k=10)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Now cross-validation is done properly - no data leakage
scores = cross_val_score(pipeline, X, y, cv=5)

print("Cross-validation scores (corrected):", scores)
print("Mean CV score:", scores.mean())
print("Std CV score:", scores.std())