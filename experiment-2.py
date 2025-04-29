import datetime
import itertools
import json
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


MAX_PROMPT_TOKENS = 4000


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
        prompts_with_idx: List[Tuple[int, str]] = [
            (i, self.row_to_string(row)) for i, row in test_df.reset_index(drop=True).iterrows()
        ]
        prompts_with_idx.sort(key=lambda p: len(p[1]))
        orig_indices: List[int] = [idx for idx, _ in prompts_with_idx]
        sorted_prompts: List[str] = [p for _, p in prompts_with_idx]



        outputs = self.batch_model.process_batch(
            sorted_prompts,
            structure_header="{",
            max_new_tokens=max_new_tokens
        )

        json_output = model.extract_json(outputs, {"age": None, "gender": None})
        inference_temp = pd.DataFrame.from_records(json_output)
        inference_temp["orig_idx"] = orig_indices  # map back to original rows

        # Re-order to original ordering
        inference_df = (
            inference_temp.sort_values("orig_idx")
            .drop(columns=["orig_idx"])
            .rename(columns={"age": "predicted_age", "gender": "predicted_gender"})
            .reset_index(drop=True)
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
            inference_df
        ], axis=1)


    # ===== Stage 2 ===== #

    def summarise_posts(
        self,
        df: pd.DataFrame,
        subreddits: List[str],
        age_ranges: np.ndarray,
        gender_range: list[str],
        max_new_tokens: int,
        prompts_per_batch: int = 16
    ) -> dict:
        stage_2_reports: dict = {}

        prompt_meta: List[Tuple[str, str, str]] = []   # (sub, age, gender) per prompt
        prompt_texts: List[str] = []

        # build prompts for vLLM
        config.debug("pre-computing truncated batchesg")
        for sub, age, gender in itertools.product(subreddits, age_ranges, gender_range):
            df_slice = df[
                (df["subreddit"] == sub) &
                (df["predicted_age"] == age) &
                (df["predicted_gender"] == gender)
            ].sort_values("date_posted")

            if df_slice.empty:
                continue

            for prompt in self.slice_to_truncated_prompts(df_slice):
                prompt_meta.append((sub, age, gender))
                prompt_texts.append(prompt)

        # send to vLLM in batches
        all_outputs: List[str] = []

        config.debug(f"now computing {(len(prompt_texts) // 16) + 1} batches")
        for batch_prompts in self.batch(prompt_texts, prompts_per_batch):
            outputs = self.batch_model.process_batch(
                batch_prompts,
                structure_header='{ "trends": [',
                max_new_tokens=max_new_tokens,
            )
            all_outputs.extend(outputs)

        # record outputs
        with open(config.RESULTS_DIR / "full-text-output.txt", "a") as f:
            for i, o in enumerate(outputs):
                f.write(f"[Output {i + 1}]:\n{o}\n\n")

        # parse and aggregate per (sub, age, gender)
        parsed = model.extract_json(all_outputs,
                                    {"trends": [], "format-error": True})

        for (sub, age, gender), obj in zip(prompt_meta, parsed):
            key = (sub, age, gender)
            bucket = stage_2_reports.setdefault(
                key,
                {"reports": [], "errors": 0,
                "start_date": str(df.date_posted.min()),
                "end_date":   str(df.date_posted.max()),
                "chunk_size": 0}
            )
            bucket["reports"].extend(obj.get("trends", []))
            bucket["errors"] += int(obj.get("format-error", False))
            bucket["chunk_size"] += 1

        return stage_2_reports


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
            prompt = query_header + "\n".join(report_strings)

            new_report = self.batch_model.process_batch(
                [prompt],
                structure_header="{\n    \"trends\": [",
                max_new_tokens=max_new_tokens
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
            final_reports = "\n".join([str(r) for r in reports])

        return final_reports, failed_output_counter


    # ===== Data Processing ===== #

    def batch(self, iterable, n):
        """
        Simple batching function
        """

        l = len(iterable)

        for ndx in range(0, l, n):
            yield iterable[ndx : min(ndx + n, l)]


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


    def posts_to_string(self, df: pd.DataFrame) -> str:
        """
        Turns a chunk of posts into a prompt string

        args:
        - df: a dataframe containing a selection of posts as well as inferred
            demographic information
        """
        start_date = df.iloc[0]["date_posted"]
        end_date = df.iloc[-1]["date_posted"]
        subreddit = df["subreddit"].iloc[0]

        prompt = f"All supplied posts will be from r/{subreddit}. " \
                 f"They were posted between {start_date} and {end_date}\n"

        post_prompts = df.reset_index(drop=True).apply(self.row_to_string, axis=1).tolist()

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


    def slice_to_truncated_prompts(self, df_slice: pd.DataFrame) -> List[str]:
        """
        Split one (subreddit, age, gender) slice into as many prompts as needed,
        each strictly ≤ MAX_PROMPT_TOKENS tokens, using O(n) tokenization.
        """
        tok = self.batch_model.tokenizer

        # Prepare header
        start = df_slice.iloc[0]["date_posted"]
        end   = df_slice.iloc[-1]["date_posted"]
        sub   = df_slice["subreddit"].iloc[0]
        header = (
            f"All supplied posts will be from r/{sub}. "
            f"They were posted between {start} and {end}\n"
        )
        header_ids = tok(header, add_special_tokens=False)["input_ids"]
        header_len = len(header_ids)

        # Pre-tokenize each post
        post_texts = [self.row_to_string(row) for _, row in df_slice.iterrows()]
        post_id_lens = [
            len(tok(text, add_special_tokens=False)["input_ids"])
            for text in post_texts
        ]
        sep_ids = tok("\n", add_special_tokens=False)["input_ids"]
        sep_len = len(sep_ids)

        prompts = []
        current_texts = []
        current_len = header_len

        for text, length in zip(post_texts, post_id_lens):
            add_len = length + (sep_len if current_texts else 0)
            if current_len + add_len > MAX_PROMPT_TOKENS:
                # flush
                prompts.append(header + "\n".join(current_texts))
                current_texts = [text]
                current_len = header_len + length
            else:
                if current_texts:
                    current_len += sep_len + length
                else:
                    current_len += length
                current_texts.append(text)

        if current_texts:
            prompts.append(header + "\n".join(current_texts))

        return prompts


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
        - experiment_sub_heading: the name of the parent directory to put results into
        - which_subreddits: whether to select the relevant or irrelevant posts
        - hierarchical_summarisation: whether to perform the hierarchical summarisation step
        - demographics_max_tokens: max tokens for demographic inference
        - chunk_report_max_tokens: max tokens for mini‐reports
        - overall_report_max_tokens: max tokens for final reports
        - model_name / model_id: identifier for naming output folder and loading the model
        - num_samples: optional subsample size
        """
        # Compute the largest possible prompt+generate footprint and add a small buffer
        max_input_plus_output = max(
            demographics_max_tokens,
            chunk_report_max_tokens,
            overall_report_max_tokens
        ) + MAX_PROMPT_TOKENS
        buffer = 512

        # 1) Load up vLLM with an increased max_seq_len
        self.batch_model = model.BatchModel(
            model_id,
            max_model_len=2*(max_input_plus_output + buffer),
        )

        # Prepare output directory
        output_path = config.RESULTS_DIR / experiment_sub_heading / model_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Pull the right slice of the data
        test_df = self.select_test_df(which_subreddits, num_samples=num_samples)

        # === Stage 1: demographic inference ===
        pre1 = config.CODE_DIR / "pre-prompts" / "expt1-zero-shot.txt"
        self.batch_model.load_pre_prompt(pre1)

        demographic_df = self.generate_demographic_inferences(
            test_df,
            max_new_tokens=demographics_max_tokens,
            batch_size=100
        )

        demographic_df.to_parquet(output_path / "demographic-inferences.parquet")

        # === Stage 2: mini reports ===
        pre2 = config.CODE_DIR / "pre-prompts" / "expt2-stage-2.txt"
        self.batch_model.load_pre_prompt(pre2)

        stage_2_reports = self.summarise_posts(
            demographic_df,
            subreddits=self.subreddit_selection[which_subreddits],
            age_ranges=np.array([f"{i}-{i+5}" for i in range(0,100,5)] + ["unknown"]),
            gender_range=["male","female","unknown"],
            max_new_tokens=chunk_report_max_tokens,
            prompts_per_batch=16
        )

        self.dump_report(stage_2_reports, output_path / "experiment-2-stage-2-reports.json")

        if not hierarchical_summarisation:
            return

        # === Stage 3: hierarchical summarisation ===
        pre3 = config.CODE_DIR / "pre-prompts" / "expt2-stage-3.txt"
        self.batch_model.load_pre_prompt(pre3)

        summarised_reports: dict = {}
        subs = self.subreddit_selection[which_subreddits]
        ages = np.array([f"{i}-{i+5}" for i in range(0,100,5)] + ["unknown"])
        genders = ["male","female","unknown"]

        for age, gender in itertools.product(ages, genders):
            reports: List[dict] = []
            for sub in subs:
                for r in stage_2_reports.get((sub, age, gender), {}).get("reports", []):
                    if isinstance(r, str):
                        r = {"summary": r, "evidence": [], "reasoning": ""}
                    reports.append({"subreddit": sub, **r})

            if not reports:
                continue

            config.debug(f"Summarising {len(reports)} mini‐reports for {(age, gender)}")
            summarised_reports[(age, gender)] = self.hierarchical_summarisation(
                (age, gender),
                reports,
                max_new_tokens=overall_report_max_tokens,
                batch_size=4,
            )

        config.output("Done with experiment!")
        self.dump_report(summarised_reports, output_path / "experiment-2-stage-3.json")

        del self.batch_model


if __name__ == "__main__":
    hf_token_path = config.CODE_DIR / "hf-access-token.txt"
    if hf_token_path.exists():
        with open(hf_token_path) as f:
            os.environ["HF_TOKEN"] = f.read().strip()

    expt = Experiment2(42)

    # List LLMs to sample
    model_ids = {
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "llama-2": "meta-llama/Llama-2-7b-chat-hf",
        "deepseek": "deepseek-ai/deepseek-llm-7b-chat"
    }

    model_name = "mistral"
    model_id = model_ids[model_name]

    for selection in "relevant", "irrelevant":
        expt.run_experiment(
            experiment_sub_heading=f"expt2-{selection}-subs-general-trends-full-dataset",
            which_subreddits=selection,
            hierarchical_summarisation=True,
            demographics_max_tokens=30,
            chunk_report_max_tokens=100,
            overall_report_max_tokens=500,
            model_name=model_name,
            model_id=model_id,
            num_samples=1000
        )

    exit()
