import datasets
from joblib import Memory
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC, SVR
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import time
from typing import Tuple

import config

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


# This class selects the best hyperparameters *for each model*
class CVModelSelector:
    def __init__(self):
        self.vectorisers = {}
        self.models = {}
        self.memory = Memory(location='./cache', verbose=0)

        self.vectoriser_parameters = {}
        self.model_parameters = {}

        self.parameter_grid = []


    def add_vectoriser(self, name: str, vec, params: dict):
        """
        Adds a vectoriser to the model selector

        args:
        - name: the name of the vectoriser
        - vec: the vectoriser
        - params: its tunably hyperparameters (dict)
        """
        self.vectorisers[name] = vec
        self.vectoriser_parameters[name] = params


    def add_model(self, name: str, model, params: dict):
        """
        Adds a model to the model selector

        args:
        - name: the name of the classifier
        - model: the model
        - params: its tunably hyperparameters (dict)
        """
        self.models[name] = model
        self.model_parameters[name] = params


    def _build_parameter_grids(self):
        """
        Builds the parameter grids from the existing vectorisers/models
        """
        # Classifier parameter grid
        for model_name, model in self.models.items():
            for vec_name, vec in self.vectorisers.items():
                params = { "kernel": [vec], "model": [model] }

                for param, values in self.vectoriser_parameters[vec_name].items():
                    params[f"kernel__{param}"] = values

                for param, values in self.model_parameters[model_name].items():
                    params[f"model__{param}"] = values

                self.parameter_grid.append(params)


    def grid_search(self, X_train, y_train, cv=5) -> pd.DataFrame:
        """
        Performs grid search CV with cv folds
        """
        self._build_parameter_grids()

        placeholder_pipeline = Pipeline([
            ("kernel", list(self.vectorisers.values())[0]),
            ("model", list(self.models.values())[0])
        ], memory=self.memory)


        config.debug("Grid search for classification")
        grid_search = GridSearchCV(placeholder_pipeline, self.parameter_grid,
                                   cv=5, n_jobs=-1, verbose=10)
        grid_search.fit(X_train, y_train)

        # find best models
        results_df = pd.DataFrame(grid_search.cv_results_)

        results_df['model_name'] = results_df['param_model'].apply(lambda x: type(x).__name__)
        results_df['kernel_name'] = results_df['param_kernel'].apply(lambda x: type(x).__name__)

        # Group by and select the best configuration based on mean_test_score.
        best_per_group = results_df.loc[results_df.groupby(['model_name', 'kernel_name'])['mean_test_score'].idxmax()]

        print("Best hyperparameters for each (model, vectoriser) combination:")
        print(best_per_group[['model_name', 'kernel_name', 'mean_test_score', 'params']])

        # Optionally, return the best configurations
        return best_per_group



config.debug("Creating models and kernels")

# kernels
cls_model_selector = CVModelSelector()

cls_model_selector.add_vectoriser(
    "tfidf",
    TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english"),
    { "max_features": [1000, 3000] }
)
cls_model_selector.add_vectoriser(
    "bofw",
    CountVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english"),
    { "max_features": [1000, 3000] }
)

cls_model_selector.add_model("svm", LinearSVC(dual="False", max_iter=1000), { "C": [0.1, 1, 10] })
cls_model_selector.add_model("lr", LogisticRegression(max_iter=1000), { "C": [0.1, 1, 10] })
cls_model_selector.add_model(
    "stack",
    StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=100))
        ],
        final_estimator=LogisticRegression(max_iter=1000)
    ),
    { "final_estimator__C": [0.1, 1, 10] }
)


if __name__ == "__main__":
    dataset = DatasetLoader()
    X_train_cv, y_train_gender_cv = dataset.get_Xy("train", "gender", subset_size=50_000)
    
    best_models = cls_model_selector.grid_search(X_train_cv, y_train_gender_cv)
    best_models.to_parquet(PROJECT_PATH / "results" / "experiment-1-comparison-output.parquet")
