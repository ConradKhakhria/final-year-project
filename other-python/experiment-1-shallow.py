# mypy: ignore-errors
import datasets
from joblib import Memory
import json
import numpy as np
import os
import pandas as pd
import psutil
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, f1_score, \
                            classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import sys
import time
from typing import Dict, List, Literal, Tuple
from sklearn.preprocessing import Normalizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config

CROSS_VALIDATE = False
PROJECT_PATH = Path.home()

# To reduce crazy memory usage
BUCKETS = np.array([f"{s}-{s + 5}" for s in (5 * (np.arange(0, 100) // 5))])
BUCKET_MIDPOINTS = {}

for b in BUCKETS:
    BUCKET_MIDPOINTS[b] = sum(map(int, b.split("-"))) / 2


def mae_age_scorer(estimator, X, y_true):
    y_pred = estimator.predict(X)
    y_true_mid = np.array([BUCKET_MIDPOINTS[y] for y in y_true])
    y_pred_mid = np.array([BUCKET_MIDPOINTS[y] for y in y_pred])

    return -np.mean(np.abs(y_true_mid - y_pred_mid))


class DatasetLoader:
    def __init__(self, buckets: np.ndarray, seed: int | None = None):
        config.debug("Loading datasets")
        self.dataset = datasets.load_dataset("blog_authorship_corpus", trust_remote_code=True)

        self.df_train = self.dataset["train"].to_pandas()
        self.df_test = self.dataset["validation"]

        self.X = {
            "train": self.df_train["text"],
            "test": self.df_test["text"]
        }

        self.y = {
            "train": { "age": self.df_train["age"], "gender": self.df_train["gender"] },
            "test":  { "age": self.df_test["age"],  "gender": self.df_test["gender"]}
        }

        self.buckets = buckets
        self.rng = np.random.default_rng(seed)


    def get_Xy(
        self, split: str, label: str, subset_size: int | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the X and y pair

        args:
        - split: 'train' or 'test'
        - label: 'age' or 'gender'
        - subset_size: the size of the random subset to use (default None: full dataset)

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

        if label == "age":
            y = self.buckets[y]

        if subset_size is not None:
            idxs = self.rng.choice(np.arange(len(X)), size=subset_size, replace=False)

            X = X[idxs]
            y = y[idxs]

        return X, y


def cross_validate(
    dataset: DatasetLoader, label: Literal["age", "gender"], subset_size: int | None = None
) -> pd.DataFrame:
    """
    Performs cross validation to obtain the performance of each model

    args:
    - dataset: the dataset loader
    - label: "age" or "gender"
    - subset_size: how big a subset

    returns:
    a Pandas DataFrame recording:
        - model
        - vectoriser
        - accuracy
        - f1 macro
        - mae (for age only)
        - best parameters
    """
    X, y = dataset.get_Xy("train", label, subset_size=subset_size)

    # Models
    models = {
        "logistic": LogisticRegression(max_iter=2000),
        "rf": RandomForestClassifier(n_jobs=1),
        "svc": LinearSVC()
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
        "rf": { "n_estimators": [100, 200, 500], "max_depth": [10, 20, 30] },
        "svc": { "C": [0.1, 1, 10] },
    }

    # scoring - different for age and gender
    if label == "age":
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1_macro': make_scorer(f1_score, average='macro'),
            'mae_age':  mae_age_scorer,
        }
        refit = "mae_age"  # Optimize for MAE with age
    else:
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1_macro': make_scorer(f1_score, average='macro'),
        }
        refit = "f1_macro"

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
                refit=refit,
                n_jobs=-1
            )
            grid.fit(X, y)

            idx = grid.best_index_
            result = {
                "model": model_name,
                "vectoriser": vec_name,
                "accuracy": grid.cv_results_["mean_test_accuracy"][idx],
                "f1_macro": grid.cv_results_["mean_test_f1_macro"][idx],
                "best_params": grid.best_params_
            }

            # Add MAE for age
            if label == "age":
                result["mae"] = -grid.cv_results_[f"mean_test_mae_age"][idx]
                
            results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(f"{label}_model_selection_with_metrics.csv", index=False)

    return df


def evaluate_model(model: ClassifierMixin, X_train, X_test, y_train, y_test, dataset=None, is_age=False) -> dict:
    """
    Evaluates a model with pre-vectorised text input
    
    For age, also calculates MAE based on bucket midpoints
    """
    global BUCKETS, BUCKET_MIDPOINTS

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    result = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "confusion": confusion_matrix(y_test, y_pred)
    }
    
    # For age, also calculate MAE
    if is_age and dataset is not None:
        y_test_mid = np.array([BUCKET_MIDPOINTS[y] for y in y_test])
        y_pred_mid = np.array([BUCKET_MIDPOINTS[y] for y in y_pred])
        result["mae"] = np.mean(np.abs(y_test_mid - y_pred_mid))
        
        # Also calculate adjacent-category accuracy
        y_test_indices = np.array([list(BUCKETS).index(y) for y in y_test])
        y_pred_indices = np.array([list(BUCKETS).index(y) for y in y_pred])
        adjacent_correct = np.sum(np.abs(y_test_indices - y_pred_indices) <= 1)
        result["adjacent_accuracy"] = adjacent_correct / len(y_test)
    
    return result


if __name__ == "__main__":
    buckets = np.array([f"{s}-{s + 5}" for s in (5 * (np.arange(0, 100) // 5))])
    dataset = DatasetLoader(seed=42, buckets=buckets)

    if CROSS_VALIDATE:
        age_results = cross_validate(dataset, "age", subset_size=5000)
        gender_results = cross_validate(dataset, "gender", subset_size=5000)
        
        print("Age Model Selection Results:")
        print(age_results[["model", "vectoriser", "accuracy", "f1_macro", "mae"]].sort_values("mae"))
        
        print("\nGender Model Selection Results:")
        print(gender_results[["model", "vectoriser", "accuracy", "f1_macro"]].sort_values("f1_macro", ascending=False))
    else:
        best_age_models = {
            "lr":  LogisticRegression(C=1, max_iter=2000, solver="saga", n_jobs=-1),
            "rf":  RandomForestClassifier(max_depth=30, n_estimators=500, max_samples=50_000, n_jobs=-1),
            "svc": LinearSVC(C=0.1)
        }

        best_gender_models = {
            "lr":  LogisticRegression(C=1, max_iter=2000, solver="saga", n_jobs=-1),
            "rf":  RandomForestClassifier(max_depth=30, n_estimators=500, max_samples=50_000, n_jobs=-1),
            "svc": LinearSVC(C=0.1)   
        }

        vectorisers = {
            "tf-idf": TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english'),
            "b-of-w": Pipeline([
                ("count", CountVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")),
                ("normaliser", Normalizer(norm='l2'))
            ])
        }

        vectoriser_preferences = {
            "age": {
                "svc": "tf-idf",
                "lr":  "b-of-w",
                "rf":  "tf-idf"
            },
            "gender": {
                "svc": "tf-idf",
                "lr":  "tf-idf",
                "rf":  "b-of-w"
            }
        }

        # pre-vectorise training and testing sets
        config.debug("Vectorising datasets")
        X_train, y_train_age = dataset.get_Xy("train", "age", subset_size=500)
        _, y_train_gender    = dataset.get_Xy("train", "gender", subset_size=500)
        X_test, y_test_age   = dataset.get_Xy("test", "age", subset_size=500)
        _, y_test_gender     = dataset.get_Xy("test", "gender", subset_size=500)

        X_train_vec = { name : vec.fit_transform(X_train) for name, vec in vectorisers.items() }
        X_test_vec  = { name : vec.transform(X_test) for name, vec in vectorisers.items() }

        # Get results for age models
        age_results = []

        for name, model in best_age_models.items():
            config.debug(f"Evaluating model {name} for age")
            pref_vec = vectoriser_preferences["age"][name]
            m_X_train, m_X_test = X_train_vec[pref_vec], X_test_vec[pref_vec]
            r = evaluate_model(model, m_X_train, m_X_test, y_train_age,
                               y_test_age, dataset=dataset, is_age=True)
            age_results.append({ "name": name, **r })

        # Get results for gender models
        gender_results = []

        for name, model in best_gender_models.items():
            config.debug(f"Evaluating model {name} for gender")
            pref_vec = vectoriser_preferences["gender"][name]
            m_X_train, m_X_test = X_train_vec[pref_vec], X_test_vec[pref_vec]
            r = evaluate_model(model, m_X_train, m_X_test, y_train_gender,
                               y_test_gender, dataset=dataset, is_age=False)
            gender_results.append({ "name": name, **r })

        # Create DataFrames with results
        df_age = pd.DataFrame(age_results)
        df_gender = pd.DataFrame(gender_results)

        # Save results
        age_results_for_csv = [{
            "model": r["name"],
            "accuracy": r["accuracy"],
            "f1_macro": r["f1_macro"],
            "mae": r["mae"],
            "adjacent_accuracy": r["adjacent_accuracy"]
        } for r in age_results]

        gender_results_for_csv = [{
            "model": r["name"],
            "accuracy": r["accuracy"],
            "f1_macro": r["f1_macro"]
        } for r in gender_results]

        pd.DataFrame(age_results_for_csv).to_csv(config.RESULTS_DIR / "full-testing-age-evaluation.csv")
        pd.DataFrame(gender_results_for_csv).to_csv(config.RESULTS_DIR / "full-testing-gender-evaluation.csv")
        
        # Print evaluation metrics to console
        print("\n===== AGE PREDICTION RESULTS =====")
        print(pd.DataFrame(age_results_for_csv).sort_values("mae"))
        print("\n===== GENDER PREDICTION RESULTS =====")
        print(pd.DataFrame(gender_results_for_csv).sort_values("f1_macro", ascending=False))
        
        # Print classification reports
        for name, model in best_age_models.items():
            print(f"\nClassification report for {name} (age):")
            model.fit(X_train_vec["tf-idf"], y_train_age)
            y_pred = model.predict(X_test_vec["tf-idf"])
            print(classification_report(y_test_age, y_pred, zero_division=0))
            
            # Calculate distance-based confusion analysis
            y_test_indices = np.array([list(dataset.buckets).index(y) for y in y_test_age])
            y_pred_indices = np.array([list(dataset.buckets).index(y) for y in y_pred])
            distances = np.abs(y_test_indices - y_pred_indices)
            
            print(f"Distance-based error analysis for {name}:")
            print(f"Exact match: {np.sum(distances == 0) / len(distances):.2%}")
            print(f"Off by 1 bucket: {np.sum(distances == 1) / len(distances):.2%}")
            print(f"Off by 2 buckets: {np.sum(distances == 2) / len(distances):.2%}")
            print(f"Off by >2 buckets: {np.sum(distances > 2) / len(distances):.2%}")
            print(f"Average distance (in buckets): {np.mean(distances):.2f}")



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


"""
>>> pd.read_csv("age_model_selection_wrapped.csv")
      model vectoriser  accuracy  f1_macro                                        best_params
0  logistic     tf-idf    0.3508  0.202929                                   {'model__C': 10}
1        rf     tf-idf    0.3574  0.139767  {'model__max_depth': 30, 'model__n_estimators'...
2       svc     tf-idf    0.3322  0.210396        {'model__C': 10, 'model__kernel': 'linear'}
3  logistic     b-of-w    0.3472  0.202759                                   {'model__C': 10}
4        rf     b-of-w    0.3564  0.133840  {'model__max_depth': 30, 'model__n_estimators'...
5       svc     b-of-w    0.3206  0.199015        {'model__C': 10, 'model__kernel': 'linear'}
>>> pd.read_csv("gender_model_selection_wrapped.csv")
      model vectoriser  accuracy  f1_macro                                        best_params
0  logistic     tf-idf    0.6180  0.617573                                    {'model__C': 1}
1        rf     tf-idf    0.5958  0.595127  {'model__max_depth': 30, 'model__n_estimators'...
2       svc     tf-idf    0.6202  0.619512            {'model__C': 1, 'model__kernel': 'rbf'}
3  logistic     b-of-w    0.6068  0.606310                                    {'model__C': 1}
4        rf     b-of-w    0.5982  0.597691  {'model__max_depth': 30, 'model__n_estimators'...
5       svc     b-of-w    0.6112  0.610425            {'model__C': 1, 'model__kernel': 'rbf'}

"""
