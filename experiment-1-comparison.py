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

CROSS_VALIDATE = True
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


    def grid_search(self, X, y, cv=5) -> dict:
        """
        Performs grid search CV on a dataset

        args:
        - X: the features to validate on
        - y: the labels "" "" ""
        - cv=5: the number of folds

        returns:
            a dict containing all of best hyperparameters
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

        return grid_search.best_params_


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
        """
        best_parameters = {}

        for model_name in self.models:
            model  = self.models["model"]
            params = self.models["params"]

            selector = CVParameterSelector(model)

            for p_name, p_values in params:
                selector.add_parameters(p_name, p_values)

            for vectoriser_name, vectoriser in self.vectorisers.items():
                selector.add_vectoriser(vectoriser_name, vectoriser)

            best_params = selector.grid_search(X, y, cv=cv)

            best_parameters[model_name] = best_parameters

        return best_parameters


if __name__ == "__main__":
    dataset = DatasetLoader()

    if CROSS_VALIDATE:
        X_train_cv, y_train_gender_cv = dataset.get_Xy("train", "gender", subset_size=50_000)

        bms = BestModelSelector()
        
        # vectorisers
        bms.add_vectoriser("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english"))
        bms.add_vectoriser("bofw", CountVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english"))

        # SVM
        bms.add_model("svm", LinearSVC(dual="False", max_iter=1_000))
        bms.add_model_params("svm", "C", [0.1, 1, 10])

        # Stack
        bms.add_model("stack", StackingClassifier(
            estimators=[
                ('lr', LogisticRegression(max_iter=1000)),
                ('rf', RandomForestClassifier(n_estimators=100))
            ],
            final_estimator=LogisticRegression(max_iter=1000)
        ))
        bms.add_model_params("stack", "final_estimator__C", [0.1, 1, 10])

        best_params = bms.get_best_models(X_train_cv, y_train_gender_cv)

        print(best_params)
