# mypy: ignore-errors
import datasets
from joblib import Memory
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC, SVR
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import time
from typing import Dict, Tuple

import config

CROSS_VALIDATE = False
PROJECT_PATH = Path.home()


class DatasetLoader:
    def __init__(self):
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


    def get_Xy(self, split: str, label: str, subset_size: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
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

        if subset_size is not None:
            idxs = np.random.choice(np.arange(len(X)), size=subset_size, replace=False)

            X = X[idxs]
            y = y[idxs]

        return X, y


class CVParameterSelector:
    def __init__(self, model):
        self.model = model

        self.memory = Memory(location='./cache', verbose=0)
        self.hyperparameters = {}
        self.vectorisers = {}


    def add_parameters(self, name: str, values: list[float]):
        """
        Adds a hyperparameter and a set of candidate values
        """
        self.hyperparameters[name] = values


    def add_vectoriser(self, name: str, vec):
        """
        Adds a vectoriser
        """
        self.vectorisers[name] = vec


    def grid_search(self, X, y, cv=5) -> tuple:
        """
        Performs grid search CV on a dataset

        args:
        - X: the features to validate on
        - y: the labels "" "" ""
        - cv=5: the number of folds

        returns:
        A tuple containing
            - the untrained optimal model
            - a dict containing all of best hyperparameters
        """
        parameter_grid = []

        # Build hyperparameter grid
        for vec_name, vec in self.vectorisers.items():
            params = { "vectoriser": [vec] }

            for param, values in self.hyperparameters.items():
                params[f"model__{param}"] = values

            parameter_grid.append(params)

        # grid search
        placeholder_pipeline = Pipeline([
            ("vectoriser", list(self.vectorisers.values())[0]),
            ("model", self.model)
        ], memory=self.memory)

        config.debug("Grid search for classification")
        grid_search = GridSearchCV(placeholder_pipeline, parameter_grid, cv=5, n_jobs=-1, verbose=10)
        grid_search.fit(X, y)

        return grid_search.best_estimator_, grid_search.best_params_


class BestModelSelector:
    def __init__(self):
        self.models = {}
        self.vectorisers = {}


    def add_model(self, name: str, model):
        """
        Adds a model and its hyperparameters

        args:
        - name: the model name
        - model: the model class
        """
        self.models[name] = {
            "model": model,
            "params": {}
        }


    def add_model_params(self, model_name: str, param_name: str, params: list):
        """
        Adds a list of model parameter values
    
        args:
        - model_name: the name of the model
        - param_name: the name of the parameter
        - params: the values it can take
        """
        self.models[model_name]["params"][param_name] = params


    def add_vectoriser(self, name: str, vec):
        """
        Adds a vectoriser

        args:
        - name: the vectorisers's name
        - vec: the vectoriser
        """
        self.vectorisers[name] = vec


    def get_best_models(self, X, y, cv=5) -> Dict[str, dict]:
        """
        Does 'cv'-fold grid search cv for each model

        returns:
            a dict mapping model name to
            
                {
                    "model", grid_search.best_estimator_,
                    "params": grid_search.best_params_
                }
        """
        best_parameters = {}

        for model_name in self.models:
            model  = self.models[model_name]["model"]
            params = self.models[model_name]["params"]

            selector = CVParameterSelector(model)

            for p_name, p_values in params.items():
                selector.add_parameters(p_name, p_values)

            for vectoriser_name, vectoriser in self.vectorisers.items():
                selector.add_vectoriser(vectoriser_name, vectoriser)

            e, p = selector.grid_search(X, y, cv=cv)

            best_parameters[model_name] = {
                "model":  e,
                "params": p
            }

        return best_parameters


if __name__ == "__main__":
    dataset = DatasetLoader()

    if CROSS_VALIDATE:
        X_train_cv, y_train_gender_cv = dataset.get_Xy("train", "gender", subset_size=50_000)

        bms = BestModelSelector()
        
        # vectorisers
        bms.add_vectoriser("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english"))
        bms.add_vectoriser("bofw", CountVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english"))

        # Logistic Regression
        bms.add_model("lr", LogisticRegression(max_iter=2_000))
        bms.add_model_params("lr", "C", [0.1, 1, 10])

        # SVM
        bms.add_model("svm", LinearSVC(dual=False, max_iter=2_000))
        bms.add_model_params("svm", "C", [0.1, 1, 10])

        # Random Forest
        bms.add_model("rf", RandomForestClassifier(n_jobs=1))
        bms.add_model_params("rf", "n_estimators", [100, 200, 500])
        bms.add_model_params("rf", "max_depth", [None, 10, 20, 30])

        config.debug("Obtaining best parameters for each model")
        best_params = bms.get_best_models(X_train_cv, y_train_gender_cv)

        best_lr: LogisticRegression = best_params["lr"]["model"]
        best_rf: RandomForestClassifier = best_params["rf"]["model"]
        best_svm: LinearSVC = best_params["svm"]["model"]

        print(f'    Best params for lr:\n{best_params["lr"]["params"]}')
        print(f'    Best params for rf:\n{best_params["rf"]["params"]}')
        print(f'    Best params for svm:\n{best_params["svm"]["params"]}')
    else:
        best_lr = Pipeline([
            ("vectoriser", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
            ("model", LogisticRegression(C=1)),
        ])

        best_rf = Pipeline([
            ("vectoriser", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
            ("model", RandomForestClassifier(max_depth=None, n_estimators=200, n_jobs=-1)),
        ])

        best_svm = Pipeline([
            ("vectoriser", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
            ("model", LinearSVC(C=0.1)),
        ])

    config.debug("Testing each model")
    X_train, y_train_gender = dataset.get_Xy("train", "gender")
    X_test, y_test_gender = dataset.get_Xy("test", "gender")

    config.debug("Fitting each model")
    print(" - lr")
    best_lr.fit(X_train, y_train_gender)
    print(" - rf")
    best_rf.fit(X_train, y_train_gender)
    print(" - svm")
    best_svm.fit(X_train, y_train_gender)

    config.debug("Running predictions")
    y_pred_lr = best_lr.predict(X_test)
    y_pred_rf = best_rf.predict(X_test)
    y_pred_svm = best_svm.predict(X_test)

    print(f"Classification report for lr:\n{classification_report(y_test_gender, y_pred_lr)}")
    print(f"Classification report for rf:\n{classification_report(y_test_gender, y_pred_rf)}")
    print(f"Classification report for svm:\n{classification_report(y_test_gender, y_pred_svm)}")
