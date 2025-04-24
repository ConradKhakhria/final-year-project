# mypy: ignore-errors
import datasets
from joblib import Memory
import json
import numpy as np
import os
import pandas as pd
import psutil
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, f1_score, \
                            mean_absolute_error, root_mean_squared_error, \
                            classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import sys
import time
from typing import Dict, Literal, Tuple
from sklearn.preprocessing import Normalizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config

CROSS_VALIDATE = True
PROJECT_PATH = Path.home()


# Pre- and post-processing

class BucketRegressor(BaseEstimator, ClassifierMixin):
    def __init__(self, regressor, bucket_map: np.ndarray):
        self.regressor = regressor
        self.bucket_map = bucket_map


    def fit(self, X, y_numeric):
        self.reg_ = clone(self.regressor).fit(X, y_numeric)
        return self


    def predict(self, X):
        y_pred_numeric = self.reg_.predict(X)
        return self.bucket_map[y_pred_numeric]



class DatasetLoader:
    def __init__(self, seed: int | None = None, buckets: np.ndarray | None = None):
        config.debug("Loading datasets")
        self.dataset = datasets.load_dataset("blog_authorship_corpus", trust_remote_code=True)

        self.df_train = self.dataset["train"].to_pandas()
        self.df_test = pd.read_parquet(PROJECT_PATH / "results" / "experiment-1-test-set.parquet")

        self.X = {
            "train": self.df_train["text"],
            "test": self.df_test["text"]
        }

        self.y = {
            "train": { "age": self.df_train["age"], "gender": self.df_train["gender"] },
            "test":  { "age": self.df_test["age"],  "gender": self.df_test["gender"]}
        }

        self.rng = np.random.default_rng(seed)

        if buckets is None:
            self.buckets = np.array([f"{s}-{s + 5}" for s in (5 * (np.arange(0, 100) // 5))])
        else:
            self.buckets = buckets


    def get_Xy(
        self, split: str, label: str, subset_size: int | None = None, age_type: type = int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the X and y pair

        args:
        - split: 'train' or 'test'
        - label: 'age' or 'gender'
        - subset_size: the size of the random subset to use (default None: full dataset)
        - age_type: what type age should take:
            - str: bucket ages (for classification)
            - int: numeric ages (regression)

        returns:
        the tuple containing:
            - X
            - y
        """
        if split not in ["train", "test"]:
            raise NameError(f"No split named '{split}'")

        if label not in ["age", "gender"]:
            raise NameError(f"No target label '{label}'")

        X = self.X[split]
        y = self.y[split][label]

        if label == "age" and age_type is str:
            y = self.bucket_ages(y)

        if subset_size is not None:
            idxs = self.rng.choice(np.arange(len(X)), size=subset_size, replace=False)

            X = X[idxs]
            y = y[idxs]

        return X, y


    def bucket_ages(self, y: list) -> np.ndarray:
        """
        Discretises integer ages into 5-year buckets

        Assumptions: 0 <= y[i] <= 100
        """
        return self.buckets[y]


def cross_validate(
    dataset: DatasetLoader, label: Literal["age", "gender"], subset_size: int | None = None
) -> pd.DataFrame:
    """
    Performs cross validation to obtain the performance of each model

    args:
    - dataset: the dataset loader
    - label: "age" or "gender"
    - seed: the random seed to use
    - subset_size: how big a subset

    returns:
    a Pandas DataFrame recording:
        - model
        - vectoriser
        - accuracy
        - f1 macro
        - best parameters
    """
    buckets = dataset.buckets

    X, y_bucket  = dataset.get_Xy("train", label, subset_size=subset_size, age_type=str)
    _, y_numeric = dataset.get_Xy("train", label, subset_size=subset_size, age_type=int)

    # Models
    classifiers = {
        "logistic": LogisticRegression(max_iter=2000),
        "rf-classifier": RandomForestClassifier(n_jobs=1),
        "svc": LinearSVC(dual=False, max_iter=2000)
    }

    regressors = {
        "ridge": Ridge(max_iter=2000),
        "rf-regressor": RandomForestRegressor(n_jobs=1),
        "svr": SVR(max_iter=2000)
    }

    models = {
        **classifiers,
        **{ name : BucketRegressor(r, buckets) for name, r in regressors.items() }
    }

    # Vectorisers
    vectorisers = {
        "tf-idf": TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english"),
        "b-of-w": Pipeline([
            ("count", CountVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")),
            ("normaliser", Normalizer(norm='l2'))
        ])
    }

    # hyperparameters
    hyperparameters = {
        "logistic":  { "C": [0.1, 1, 10] },
        "rf-classifier": { "n_estimators": [100, 200, 500], "max_depth": [10, 20, 30] },
        "svc": { "C": [0.1, 1, 10] },
        "ridge":  { "alpha": [0.1, 1, 10] },
        "rf-regressor": { "n_estimators": [100, 200, 500], "max_depth": [10, 20, 30] },
        "svr": { "C": [0.1, 1, 10] }
    }

    # scoring
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1_macro': make_scorer(f1_score, average='macro'),
    }

    # Do the scoring
    results = []

    for vec_name, vec in vectorisers.items():
        for model_name, model in models.items():
            pipe = Pipeline([
                ("vectoriser", vec),
                ("model", model)
            ], memory=Memory(location='./cache', verbose=0))

            # Create param grid
            params = {}

            for p_name, ps in hyperparameters[model_name].items():
                params[f"model__{p_name}"] = ps

            # Cross validate
            config.debug(f"Doing CV for {model_name}:{vec_name}")
            grid = GridSearchCV(
                estimator=pipe,
                param_grid=params,
                cv=5,
                scoring=scoring,
                refit="f1_macro",
                n_jobs=-1
            )

            if model_name in regressors:
                if label == "age":
                    grid.fit(X, y_numeric)
                else:
                    continue
            else:
                grid.fit(X, y_bucket)

            idx = grid.best_index_
            results.append({
                "model": model_name,
                "vectoriser": vec_name,
                "accuracy": grid.cv_results_["mean_test_accuracy"][idx],
                "f1_macro": grid.cv_results_["mean_test_f1_macro"][idx],
                "best_params": grid.best_params_
            })

    df = pd.DataFrame(results)
    df.to_csv(f"{label}_model_selection_wrapped.csv", index=False)

    return df


if __name__ == "__main__":
    buckets = np.array([f"{s}-{s + 5}" for s in (5 * (np.arange(0, 100) // 5))])
    dataset = DatasetLoader(seed=42, buckets=buckets)

    if CROSS_VALIDATE:
        age_results = cross_validate(dataset, "age", subset_size=5000)
        gender_results = cross_validate(dataset, "gender", subset_size=5000)

        exit()
    else:
        best_lr = Pipeline([
            ("vectoriser", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
            ("model", LogisticRegression(C=1)),
        ])

        best_rf = Pipeline([
            ("vectoriser", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
            ("model", RandomForestClassifier(max_depth=30, n_estimators=200, n_jobs=4)),
        ])

        best_svm = Pipeline([
            ("vectoriser", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
            ("model", LinearSVC(C=0.1)),
        ])

    config.debug("Testing each model")
    X_train, y_train_age = dataset.get_Xy("train", "age")
    X_test, y_test_age = dataset.get_Xy("test", "age")

    config.debug("Fitting each model")
    print("fitting logistic regression")
    config.debug(psutil.virtual_memory())
    best_lr.fit(X_train, y_train_age)

    print("fitting random forest")
    config.debug(psutil.virtual_memory())
    best_rf.fit(X_train, y_train_age)

    print("fitting svm")
    config.debug(psutil.virtual_memory())
    best_svm.fit(X_train, y_train_age)

    config.debug("Running predictions")
    y_pred_lr = best_lr.predict(X_test)
    y_pred_rf = best_rf.predict(X_test)
    y_pred_svm = best_svm.predict(X_test)

    print(f"Classification report for lr:\n{classification_report(y_test_age, y_pred_lr)}")
    print(f"Classification report for rf:\n{classification_report(y_test_age, y_pred_rf)}")
    print(f"Classification report for svm:\n{classification_report(y_test_age, y_pred_svm)}")

    cm_lr = confusion_matrix(y_test_age, y_pred_lr)
    cm_rf = confusion_matrix(y_test_age, y_pred_rf)
    cm_svm = confusion_matrix(y_test_age, y_pred_svm)

    confusion_data = {
        "LogisticRegression": cm_lr.tolist(),
        "RandomForest": cm_rf.tolist(),
        "LinearSVC": cm_svm.tolist()
    }

    # Write the confusion matrices to a JSON file that can be downloaded and inspected locally
    with open("confusion_matrices.json", "w") as f:
        json.dump(confusion_data, f)







"""
Classification report for lr:
              precision    recall  f1-score   support

       10-15       0.00      0.00      0.00         2
       15-20       0.52      0.75      0.62        16
       20-25       0.22      0.18      0.20        11
       25-30       0.35      0.38      0.36        16
       30-35       0.00      0.00      0.00         1
       35-40       0.00      0.00      0.00         2
       40-45       0.00      0.00      0.00         1
       45-50       0.00      0.00      0.00         1

    accuracy                           0.40        50
   macro avg       0.14      0.16      0.15        50
weighted avg       0.33      0.40      0.36        50

/home/paperspace/final-year-project/blog-auth-env/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/paperspace/final-year-project/blog-auth-env/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/paperspace/final-year-project/blog-auth-env/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Classification report for rf:
              precision    recall  f1-score   support

       10-15       0.00      0.00      0.00         2
       15-20       0.52      0.88      0.65        16
       20-25       0.00      0.00      0.00        11
       25-30       0.48      0.69      0.56        16
       30-35       0.00      0.00      0.00         1
       35-40       0.00      0.00      0.00         2
       40-45       0.00      0.00      0.00         1
       45-50       0.00      0.00      0.00         1

    accuracy                           0.50        50
   macro avg       0.12      0.20      0.15        50
weighted avg       0.32      0.50      0.39        50

/home/paperspace/final-year-project/blog-auth-env/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/paperspace/final-year-project/blog-auth-env/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/paperspace/final-year-project/blog-auth-env/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Classification report for svm:
              precision    recall  f1-score   support

       10-15       0.00      0.00      0.00         2
       15-20       0.48      0.75      0.59        16
       20-25       0.27      0.27      0.27        11
       25-30       0.36      0.31      0.33        16
       30-35       0.00      0.00      0.00         1
       35-40       0.00      0.00      0.00         2
       40-45       0.00      0.00      0.00         1
       45-50       0.00      0.00      0.00         1

    accuracy                           0.40        50
   macro avg       0.14      0.17      0.15        50
weighted avg       0.33      0.40      0.35        50

"""
