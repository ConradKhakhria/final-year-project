import datasets
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC, SVR
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

import config

PROJECT_PATH = Path.home() / "UCL" / "FYP"


config.debug("Loading datasets")
dataset = datasets.load_dataset("blog_authorship_corpus", trust_remote_code=True)

df_train = dataset["train"].to_pandas()
df_test = pd.read_parquet(PROJECT_PATH / "results" / "experiment-1-test-set.parquet")

X_train, X_test = df_train["text"], df_test["text"]
y_age_train, y_age_test = df_train["age"], df_test["age"]
y_gender_train, y_gender_test = df_train["gender"], df_test["gender"]


config.debug("Creating models")
vectorisers = {
    "tf-idf": TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english"),
    "b-of-w": CountVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
}

classifiers = {
    "svm": LinearSVC(dual="False", max_iter=1000),
    "lr": LogisticRegression(max_iter=1000),
    "stack": StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=100, n_jobs=-1))
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        n_jobs=-1
    )
}

regressors = {
    "svm": SVR(),
    "ridge": Ridge(alpha=0.1)
}


if __name__ == "__main__":
    lr_tf_idf_gender = Pipeline([
        ("tfidf", vectorisers["tf-idf"]),
        ("lr", classifiers["lr"])
    ])

    config.debug("Training model")
    lr_tf_idf_gender.fit(X_train, y_gender_train)

    config.debug("Making predictions")
    y_gender_pred = lr_tf_idf_gender.predict(X_test)

    print(classification_report(y_gender_test, y_gender_pred))
