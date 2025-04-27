import datetime
import itertools
import json
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys
import torch
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, no_type_check

import config
import dataset
import model


class Experiment2:
    def __init__(self, seed: int):
        config.debug("Loading dataset")

        self.data_dir = config.CODE_DIR / "data"
        self.reddit_df = pd.read_parquet(self.data_dir / "combined-filtered-reddit-data.parquet")
        self.reddit_df["date_posted"] = pd.to_datetime(self.reddit_df["created_utc"], unit="s")

        with open(self.data_dir / "subreddit-selection.json") as f:
            self.subreddit_selection = json.load(f)

        self.rng = np.random.default_rng(seed)

    ##### Data selection #####


    @config.debug_function
    def select_test_df(
        self, relevance: Literal["relevant", "irrelevant"], num_samples: int | None = None
    ) -> pd.DataFrame:
        """
        Selects the testing df

        args:
        - relevance: whether to select the relevant or irrelevant subreddits
        - num_samples: optional - whether to take a subset
        """
        test_df = self.reddit_df[self.reddit_df["subreddit"].isin(self.subreddit_selection[relevance])]

        if num_samples is not None:
            n_posts = len(test_df)
            sample_idxs = self.rng.choice(np.arange(0, n_posts), size=num_samples, replace=False)
            test_df = test_df.iloc[sample_idxs]

        return test_df


    @config.debug_function
    def generate_demographic_inferences(
        self, test_df: pd.DataFrame, max_new_tokens: int, batch_size: int = 20,
    ) -> pd.DataFrame:
        """
        Returns the df with demographic groups inferred

        args:
        - test_df: the dataframe to add demographic inferences to
        - max_new_tokens: max number of new tokens the model can generate
        - batch_size: the size of each batch

        Post-processing:
        1. age is split into 5-year groupings
        2. if gender isn't strictly male or female, it is None
        """
        prompts = test_df.reset_index(drop=True).apply(self.row_to_prompt, axis=1).tolist()
        outputs = self.summary_model_isolator.process_prompts(
            prompts,
            batch_size=batch_size,
            cfg={
                "enforce_json": True,
                "max_new_tokens": max_new_tokens,
                "pre_prompt_name": "expt1-zero-shot.txt"
            }
        )

        json_output = model.extract_json(outputs, {"age": None, "gender": None})
        inference_df = pd.DataFrame.from_records(json_output).rename(
            columns={
                "age": "predicted_age",
                "gender": "predicted_gender"
            }
        )

        # Post-processing
        def format_age(age: Any) -> str:
            if isinstance(age, str) and age in bins:
                return age
            else:
                return "unknown"


        def format_gender(gender: Any) -> str:
            if isinstance(gender, str):
                gender = gender.lower()
                if gender in ['male', 'female']:
                    return gender
            return "unknown"


        bins = np.array([f"{5*(i // 5)}-{5*((i // 5) + 1)}" for i in range(110)])

        inference_df["predicted_age"] = inference_df["predicted_age"].apply(format_age)
        inference_df["predicted_gender"] = inference_df["predicted_gender"].apply(format_gender)

        return pd.concat([
            test_df.reset_index(drop=True),
            inference_df.reset_index(drop=True)
        ], axis=1)


    def create_balanced_post_selection(
        self, df_test: pd.DataFrame, subreddit: str, age_range: str,
        gender: str, n_posts: int
    ) -> List[pd.DataFrame]:
        """
        Yields dataframes each containing close to n_posts

        args:
        - df_test: the dataframe to select from
        - subreddit: the subreddit to select from
        - age_range: the age range to select from
        - gender: the gender to select
        - n_posts: the number of posts for each df

        All posts will be in time-order
        """
        chunks = []
        condition = (df_test["subreddit"] == subreddit) \
                  & (df_test["predicted_age"] == age_range) \
                  & (df_test["predicted_gender"] == gender)

        time_sorted = df_test[condition].sort_values(by="date_posted")
        df_len = len(time_sorted)

        for i_start in range(0, df_len, n_posts):
            chunk = time_sorted.iloc[i_start : min(df_len, i_start + n_posts)]
            chunks.append(chunk)

        return chunks


    ##### Data Preprocessing #####

    def row_to_prompt(self, row: pd.core.series.Series) -> str:
        """
        Converts a row into a string summarising the post.

        This function adds descriptions based on the data inside the row
        """
        s = f"- date posted: {row['date_posted'].strftime('%Y-%m-%d')}\n"

        for col in row.index:
            if col != "text":
                s += f"- {col}: {row[col]}\n"

        return s + f"- post contents:\n'{row['text']}'\n"


    def chunk_to_prompt(self, chunk: pd.DataFrame) -> str:
        """
        Turns a chunk of posts into a prompt string
        """
        start_date = chunk.iloc[0]["date_posted"]
        end_date = chunk.iloc[-1]["date_posted"]
        subreddit = chunk["subreddit"].iloc[0]

        prompt = f"All supplied posts will be from r/{subreddit}. " \
                 f"They were posted between {start_date} and {end_date}\n"

        post_prompts = chunk.reset_index(drop=True).apply(self.row_to_prompt, axis=1).tolist()

        return prompt + "\n".join(f"[post number {i + 1}]:\n{p}" for i, p in enumerate(post_prompts))


    ##### Trend Inference #####

    def get_trends_from_chunk(
        self, chunks: List[pd.DataFrame], max_new_tokens: int, batch_size = None
    ) -> List[str]:
        """
        Obtain an enumeration of consumer trends indicated by a chunk of posts

        args:
        - chunks: a list of dataframes of posts from a specific subreddit and timeframe
        - max_new_tokens: the max number of new tokens to generate
        - batch_size: the number of chunks to process at once

        returns:
            A string containing a bullet-pointed list of trends indicated
        """
        prompts = [self.chunk_to_prompt(c) for c in chunks]
        outputs = self.summary_model_isolator.process_prompts(
            prompts,
            batch_size=batch_size,
            cfg={
                "enforce_json": False,
                "max_new_tokens": max_new_tokens,
                "pre_prompt_name": "expt2-identify-trends-sector.txt"
            }
        )

        return outputs


    @config.debug_function
    def get_trends_from_reports(
        self, reports: List[str], subreddit: str | None, age_range: str,
        gender: str, max_new_tokens: int, batch_size: int
    ) -> str:
        """
        Uses the large model to produce a final report summarising consumer trends
        identified in the reports

        args:
        - reports: a list of reports made by the LLM
        - subreddit: the subreddit the report came from (nullable)
        - age_range: the inferred age range of the people who posted
        - gender: the inferred gender of the posters
        - max_new_tokens: the max number of new tokens the LLM can generate
        - batch_size: the number of reports to combine at each iteration
        """
        if reports == []:
            config.debug(f"For some reason we got 0 reports for {(subreddit, age_range, gender)}")
            return "<no trends for this demographic grouping>"

        layers = 1
        query_context = (
            "metadata:\n"
            f" - all posts are from r/{subreddit}\n" if subreddit else ""
            f" - the inferred age range of the posters is {age_range}\n"
            f" - the inferred gender of the posters is {gender}"
        )

        while len(reports) > 1:
            config.debug(f"Creating a new layer from {len(reports)} reports: layer = {layers}")
            new_reports = []

            for batch_start in range(0, len(reports), batch_size):
                batch_end = min(len(reports), batch_start + batch_size)
                batch = reports[batch_start : batch_end]
                batch_string = "\n".join(f"[report {i + 1}]:\n{r}" for i, r in enumerate(batch))

                new_report = self.aggregator_model_isolator.process_prompts(
                    [query_context + batch_string],
                    batch_size=20,
                    cfg={
                        "enforce_json": False,
                        "max_new_tokens": max_new_tokens,
                        "pre_prompt_name": "expt2-identify-trends-from-reports.txt"
                    }
                )

                new_reports.append(new_report)

            reports = new_reports
            layers += 1

        return reports[0]


    def dump_report(self, reports: dict, file_path: Path):
        """
        Formats a report dict for stringification and writes to a JSON file
        """
        reports_str_keys = { str(k) : v for k, v in reports.items() }

        with open(file_path, "w") as f:
            json.dump(reports_str_keys, f)


    ##### Overall Experiment #####

    def run_experiment(
        self,
        experiment_sub_heading: str,
        which_subreddits: Literal["relevant", "irrelevant"],
        demographics_max_tokens: int,
        chunk_report_max_tokens: int,
        overall_report_max_tokens: int,
        summary_model_id: str,
        aggregator_model_id: str,
        num_samples: int | None = None
    ):
        """
        Runs the second experiment

        args:
        - experiment_sub_heading:
            the name of the parent directory to put results into
        - which_subreddits:
            whether to select the relevant or irrelevant posts
        - demographics_max_tokens:
            the max number of new tokens to be used when generating demographic inferences
        - chunk_report_max_tokens:
            the max number of tokens for generating short reports
        - overall_report_max_tokens:
            the max number of tokens for generating the larger reports
        - summary_model_id:
            the name of the model to be used for summarising social media posts' consumer trends
        - aggregator_model_id:
            the name of the mdoel to be used for aggregating existing reports
        - num_samples (nullable):
            the number of samples to take from the dataset
        """
        self.summary_model_isolator = model.BatchModelIsolator(summary_model_id)
        self.aggregator_model_isolator = model.BatchModelIsolator(aggregator_model_id)

        subreddits = self.subreddit_selection[which_subreddits]
        age_ranges = np.array([f"{i}-{i + 5}" for i in np.arange(0, 100, 5)] + ["unknown"])
        gender_range = ["male", "female", "unknown"]

        output_path = config.RESULTS_DIR / experiment_sub_heading
        if not output_path.exists():
            output_path.mkdir()

        test_df = self.select_test_df(which_subreddits, num_samples=num_samples)
        test_df = self.generate_demographic_inferences(test_df, demographics_max_tokens, batch_size=40)

        # Dump metadata about demographic inference
        demographic_df = test_df[["text", "predicted_age", "predicted_gender"]]
        demographic_df.to_parquet(output_path / "demographic-inferences.parquet")

        trend_reports = {}

        # Generate reports for each (subreddit, age range, gender)
        for s, a, g in itertools.product(subreddits, age_ranges, gender_range):
            post_chunks = expt.create_balanced_post_selection(test_df, s, a, g, 25)

            if len(post_chunks) > 0:
                config.debug(f"Generating short reports for sub = {s}, ages = {a}, gender = {g}")
                trend_reports[(s, a, g)] = {
                    "reports": expt.get_trends_from_chunk(post_chunks, chunk_report_max_tokens, 4),
                    "start_date": str(post_chunks[0].iloc[0]["date_posted"]),
                    "end_date": str(post_chunks[-1].iloc[-1]["date_posted"])
                }

        expt.dump_report(trend_reports, output_path / "experiment-2-trend-reports.json")
        expt.summary_model_isolator.kill_batch_worker()

        # Produce larger reports
        larger_reports = {}

        for (s, a, g), entry in trend_reports.items():
            config.output(f"Reports for subreddit r/{s} with ages = {a} and gender = {g}:")
            config.output(f" - date range: {entry['start_date']} to {entry['end_date']}")
            config.output(f" - number of reports: {len(entry['reports'])}")
            config.output(f" - total text: {len(' '.join(entry['reports']))}")

            new_report = self.get_trends_from_reports(entry['reports'], s, a, g,
                                                      overall_report_max_tokens, 4)
            larger_reports[(s, a, g)] = new_report

        expt.dump_report(larger_reports, output_path / "experiment-2-subreddit-overall-reports.json")

        # Create overall reports for each demographic segment
        demographic_segment_reports = {}

        for a, g in itertools.product(age_ranges, gender_range):
            reports = []

            for s in subreddits:
                if (r := larger_reports.get((s, a, g), None)) is not None:
                    reports.append(r)

            if len(reports) > 0:
                overall_report = self.get_trends_from_reports(reports, None, a, g,
                                                              overall_report_max_tokens, 4)
                demographic_segment_reports[(a, g)] = overall_report

        config.output("Done with experiment!")

        expt.dump_report(demographic_segment_reports, output_path / "experiment-2-demographic-reports.json")
        expt.aggregator_model_isolator.kill_batch_worker()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    expt = Experiment2(42)

    # List LLMs to sample
    model_ids = {
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "llama-2": "meta-llama/Llama-2-7b-chat-hf",
        "deepseek": "deepseek-ai/deepseek-llm-7b-chat"
    }

    for name, model_id in model_ids.items():
        expt.run_experiment(
            experiment_sub_heading="expt2-relevant-subs-general-trends-all",
            which_subreddits="irrelevant",
            demographics_max_tokens=30,
            chunk_report_max_tokens=200,
            overall_report_max_tokens=2000,
            summary_model_id=model_id,
            aggregator_model_id=model_id,
            num_samples=100
        )

    exit()
