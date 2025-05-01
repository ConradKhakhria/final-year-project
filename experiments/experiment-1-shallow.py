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

import src.config as config
import src.dataset as ds


CROSS_VALIDATE = False
PROJECT_PATH = Path.home()

BUCKETS = np.array([f"{s}-{s + 5}" for s in (5 * (np.arange(0, 100) // 5))])
BUCKET_MIDPOINTS = {}

for b in BUCKETS:
    BUCKET_MIDPOINTS[b] = sum(map(int, b.split("-"))) / 2


def mae_age_scorer(estimator, X, y_true):
    y_pred = estimator.predict(X)
    y_true_mid = np.array([BUCKET_MIDPOINTS[y] for y in y_true])
    y_pred_mid = np.array([BUCKET_MIDPOINTS[y] for y in y_pred])

    return -np.mean(np.abs(y_true_mid - y_pred_mid))


def cross_validate(
    dataset: ds.DatasetLoader, label: Literal["age", "gender"], subset_size: int | None = None
) -> pd.DataFrame:
    """Performs cross validation to obtain the performance of each model

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

    models = {
        "logistic": LogisticRegression(max_iter=2000),
        "rf": RandomForestClassifier(n_jobs=1),
        "svc": LinearSVC()
    }

    vectorisers = {
        "tf-idf": TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english"),
        "b-of-w": Pipeline([
            ("count", CountVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")),
            ("normaliser", Normalizer(norm='l2'))
        ])
    }

    hyperparameters = {
        "logistic":  { "C": [0.1, 1, 10] },
        "rf": { "n_estimators": [100, 200, 500], "max_depth": [10, 20, 30] },
        "svc": { "C": [0.1, 1, 10] },
    }

    if label == "age":
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1_macro': make_scorer(f1_score, average='macro'),
            'mae_age':  mae_age_scorer,
        }
        refit = "mae_age"
    else:
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1_macro': make_scorer(f1_score, average='macro'),
        }
        refit = "f1_macro"

    results = []

    for vec_name, vec in vectorisers.items():
        for model_name, model in models.items():
            pipe = Pipeline([
                ("vectoriser", vec),
                ("model", model)
            ], memory=Memory(location='./cache', verbose=0))

            params = {}

            for p_name, ps in hyperparameters[model_name].items():
                params[f"model__{p_name}"] = ps

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
    dataset = ds.DatasetLoader(seed=42, buckets=buckets)

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
        X_train, y_train_age = dataset.get_Xy("train", "age")
        _, y_train_gender    = dataset.get_Xy("train", "gender")
        X_test, y_test_age   = dataset.get_Xy("test", "age")
        _, y_test_gender     = dataset.get_Xy("test", "gender")

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
            "adjacent_accuracy": r["adjacent_accuracy"],
            "confusion": r["confusion"]
        } for r in age_results]

        gender_results_for_csv = [{
            "model": r["name"],
            "accuracy": r["accuracy"],
            "f1_macro": r["f1_macro"],
            "confusion": r["confusion"]
        } for r in gender_results]


        with open(config.RESULTS_DIR / "age-evaluation.json", "w") as f:
            json.dump(age_results_for_csv, f)

        with open(config.RESULTS_DIR / "gender-evaluation.json", "w") as f:
            json.dump(gender_results_for_csv, f)
