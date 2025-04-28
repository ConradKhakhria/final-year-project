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

    
    # ===== Data Selection ===== #

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


    # ===== Stage 1 ===== #

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
        prompts = test_df.reset_index(drop=True).apply(self.row_to_string, axis=1).tolist()
        outputs = self.model_isolator.process_prompts(
            prompts,
            batch_size=batch_size,
            cfg={
                "structure_header": "{",
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


    # ===== Stage 2 ===== #

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


    def get_trends_from_chunk(
        self, chunks: List[pd.DataFrame], max_new_tokens: int, batch_size = None
    ) -> dict:
        """
        Obtain an enumeration of consumer trends indicated by a chunk of posts

        args:
        - chunks: a list of dataframes of posts from a specific subreddit and timeframe
        - max_new_tokens: the max number of new tokens to generate
        - batch_size: the number of chunks to process at once

        returns:
            A dict containing:
                - trends: a list of detected consumer trends
                - errors: the number of invalid outputs
                - chunk_size: the size of the chunk
        """
        prompts = [self.chunk_to_string(c) for c in chunks]
        outputs = self.model_isolator.process_prompts(
            prompts,
            batch_size=batch_size,
            cfg={
                "structure_header": "{\n    \"trends\": [",
                "max_new_tokens": max_new_tokens,
                "pre_prompt_name": "expt2-stage-2.txt"
            }
        )

        json_output = model.extract_json(outputs, {"trends": [], "format-error": True})

        trends: List[dict] = []
        errors = 0

        for o in json_output:
            trends.extend(o["trends"])
            if o.get("format-error", False):
                errors += 1

        return {
            "trends": trends,
            "errors": errors,
            "chunk_size": len(chunks)
        }


    # ===== Stage 3 ===== #

    @config.debug_function
    def hierarchical_summarisation(
        self,
        group: tuple,
        reports: List[dict],
        max_new_tokens: int,
        batch_size: int,
    ) -> Tuple[str, int]:
        """
        Uses the large model to produce a final report summarising consumer trends
        identified in the reports

        args:
        - group: the demographic group that the trend reports are derived from
        - reports: a list of reports made by the LLM
        - max_new_tokens: the max number of new tokens the LLM can generate
        - batch_size: the number of reports to combine at each iteration

        returns:
            A tuple containing the final report and the number of errors encountered
            during summarisation
        """
        age_range, gender = group

        if reports == []:
            config.debug(f"For some reason we got 0 reports for {group}")
            return ("no reports supplied as input", 1)

        query_header = (
            "METADATA\n"
            f" - the inferred age range of the posters is {age_range}\n"
            f" - the inferred gender of the posters is {gender}"
        )

        layers = 1
        failed_output_counter = 0

        current_reports_count = len(reports)
        previous_reports_count = 2*len(reports)

        while current_reports_count < previous_reports_count:
            config.debug(f"Creating a new layer from {len(reports)} reports: layer = {layers}")
            new_reports = []

            report_strings = [self.report_to_string(r, index=i) for i, r in enumerate(reports)]

            for batch_start in range(0, len(reports), batch_size):
                batch = report_strings[batch_start : min(len(reports), batch_start + batch_size)]

                new_report = self.model_isolator.process_prompts(
                    [query_header + "\n".join(batch)],
                    batch_size=20,
                    cfg={
                        "structure_header": "{\n    \"trends\": [",
                        "max_new_tokens": max_new_tokens,
                        "pre_prompt_name": "expt2-stage-3.txt"
                    }
                )

                new_report_json = model.extract_json(new_report, {"trends": [], "failed-output": True})

                for r in new_report_json:
                    new_reports.extend(r['trends'])
                    failed_output_counter += int(r.get("failed-output", False))

            reports = new_reports
            layers += 1

            previous_reports_count = current_reports_count
            current_reports_count = len(reports)

        try:
            final_reports = "\n".join([self.report_to_string(r, index=i) for i, r in enumerate(reports)])
        except:
            final_reports = "\n".join(reports)

        return final_reports, failed_output_counter


    # ===== Data Processing ===== #

    def row_to_string(self, row: pd.core.series.Series) -> str:
        """
        Converts a row into a string summarising the post.

        This function adds descriptions based on the data inside the row
        """
        s = f"- date posted: {row['date_posted'].strftime('%Y-%m-%d')}\n"

        for col in row.index:
            if col != "text":
                s += f"- {col}: {row[col]}\n"

        return s + f"- post contents:\n'{row['text']}'\n"


    def chunk_to_string(self, chunk: pd.DataFrame) -> str:
        """
        Turns a chunk of posts into a prompt string
        """
        start_date = chunk.iloc[0]["date_posted"]
        end_date = chunk.iloc[-1]["date_posted"]
        subreddit = chunk["subreddit"].iloc[0]

        prompt = f"All supplied posts will be from r/{subreddit}. " \
                 f"They were posted between {start_date} and {end_date}\n"

        post_prompts = chunk.reset_index(drop=True).apply(self.row_to_string, axis=1).tolist()

        return prompt + "\n".join(f"[post number {i + 1}]:\n{p}" for i, p in enumerate(post_prompts))


    def report_to_string(self, report: Dict[str, str], index: int | None = None) -> str:
        """
        Turns a report object into a string
        """
        query  = f"[REPORT NUMBER {index + 1}]\n" if index else ""
        query += f" - trend summary: {report['summary']}"
        
        if 'reasoning' in report:
            query += f" - reasoning: {report['reasoning']}"

        query += f" - evidence: {report['evidence']}"

        return query


    def dump_report(self, reports: dict, file_path: Path):
        """
        Formats a report dict for stringification and writes to a JSON file
        """
        reports_str_keys = { str(k) : v for k, v in reports.items() }

        with open(file_path, "w") as f:
            json.dump(reports_str_keys, f)


    # ===== Overall Experiment ===== #

    def run_experiment(
        self,
        experiment_sub_heading: str,
        which_subreddits: Literal["relevant", "irrelevant"],
        hierarchical_summarisation: bool,
        demographics_max_tokens: int,
        chunk_report_max_tokens: int,
        overall_report_max_tokens: int,
        model_name: str,
        model_id: str,
        num_samples: int | None = None
    ):
        """
        Runs the second experiment

        args:
        - experiment_sub_heading:
            the name of the parent directory to put results into
        - which_subreddits:
            whether to select the relevant or irrelevant posts
        - hierarchical_summarisation:
            whether to perform the hierarchical summarisation step
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
        self.model_isolator = model.BatchModelIsolator(model_id)

        subreddits = self.subreddit_selection[which_subreddits]
        age_ranges = np.array([f"{i}-{i + 5}" for i in np.arange(0, 100, 5)] + ["unknown"])
        gender_range = ["male", "female", "unknown"]

        output_path = config.RESULTS_DIR / experiment_sub_heading / model_name
        output_path.mkdir(parents=True, exist_ok=True)

        test_df = self.select_test_df(which_subreddits, num_samples=num_samples)

        # ===== Stage 1: generate demographic inferences ===== #
        test_df = self.generate_demographic_inferences(test_df, demographics_max_tokens, batch_size=100)

        # Dump metadata about demographic inference
        demographic_df = test_df[["text", "predicted_age", "predicted_gender"]]
        demographic_df.to_parquet(output_path / "demographic-inferences.parquet")

        # ===== Stage 2: generate mini reports ===== #
        stage_2_reports = {}

        # Generate reports for each (subreddit, age range, gender)
        for s, a, g in itertools.product(subreddits, age_ranges, gender_range):
            post_chunks = expt.create_balanced_post_selection(test_df, s, a, g, 25)

            if len(post_chunks) > 0:
                config.output(f"Generating short reports for sub = {s}, ages = {a}, gender = {g}")
                chunk_trends = expt.get_trends_from_chunk(post_chunks, chunk_report_max_tokens, 4)
                
                stage_2_reports[(s, a, g)] = {
                    "start_date": str(post_chunks[0].iloc[0]["date_posted"]),
                    "end_date": str(post_chunks[-1].iloc[-1]["date_posted"]),
                    "reports": chunk_trends['trends'],
                    "errors": chunk_trends['errors'],
                    "chunk_size": chunk_trends['chunk_size']
                }

        expt.dump_report(stage_2_reports, output_path / "experiment-2-stage-2-reports.json")

        if not hierarchical_summarisation:
            expt.model_isolator.kill_batch_worker()
            return

        # ===== Stage 3: hierarchical summarisation ===== #
        # In particular, this section produces a hierarchically-summarised
        # report for *each* age and gender pairing
        summarised_reports: dict = {}

        for a, g in itertools.product(age_ranges, gender_range):
            # accumulate all reports from all subreddits
            reports: List[dict] = []
            for s in subreddits:
                if (rs := stage_2_reports.get((s, a, g), None)) is not None:
                    reports.extend({ 'subreddit': s, **r } for r in rs['reports'])

            if len(reports) > 0:
                config.debug(f"summarising {len(reports)} reports for {(a, g)}")
                summarised_reports[(a, g)] = self.hierarchical_summarisation(
                    (a, g), reports, overall_report_max_tokens, 4
                )

        config.output("Done with experiment!")

        expt.dump_report(summarised_reports, output_path / "experiment-2-stage-3.json")
        expt.model_isolator.kill_batch_worker()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    expt = Experiment2(42)

    # List LLMs to sample
    model_ids = {
#        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
#        "llama-2": "meta-llama/Llama-2-7b-chat-hf",
        "deepseek": "deepseek-ai/deepseek-llm-7b-chat"
    }

    for model_name, model_id in model_ids.items():
        expt.run_experiment(
            experiment_sub_heading=f"expt2-relevant-subs-general-trends-all",
            which_subreddits="relevant",
            hierarchical_summarisation=True,
            demographics_max_tokens=30,
            chunk_report_max_tokens=200,
            overall_report_max_tokens=2000,
            model_name=model_name,
            model_id=model_id,
            num_samples=100
        )

    exit()

    model_name = "mistral"
    model_id = model_ids[model_name]

    for selection in "relevant", "irrelevant":
        expt.run_experiment(
            experiment_sub_heading=f"expt2-{selection}-subs-general-trends-all",
            which_subreddits=selection,
            hierarchical_summarisation=True,
            demographics_max_tokens=30,
            chunk_report_max_tokens=200,
            overall_report_max_tokens=2000,
            model_name=model_name,
            model_id=model_id
        )

    exit()
