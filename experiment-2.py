import datetime
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


class BatchModelIsolator:
    def __init__(self, which_model: Literal["small", "large"]):
        if which_model == "small":
            self.model_id = config.SMALL_MODEL
        else:
            self.model_id = config.LARGE_MODEL

        self.p: Optional[mp.Process]  = None
        self.in_queue: Optional[mp.Queue] = None
        self.out_queue: Optional[mp.Queue] = None
        self.cfg: Dict[str, Any] = {}


    @no_type_check
    def process_prompts(
        self, prompts: List[str], batch_size: Optional[int] = None, cfg: Optional[dict] = None,
    ) -> List[str]:
        """
        Processes a list of prompts in a separate process

        args:
        - prompts: a list of prompts to process
        - batch_size: the number of batches to process at a time
        - cfg: a dict containing overrides for:
            1. max_new_tokens
            2. pre_prompt_path
            3. enforce_json
        """
        cfg_modified = False

        # Set configurations
        if mnt := cfg.get("max_new_tokens", None):
            self.cfg["max_new_tokens"] = mnt
            cfg_modified = True

        if ppn := cfg.get("pre_prompt_name", None):
            self.cfg["pre_prompt_path"] = config.CODE_DIR / "pre-prompts" / ppn
            cfg_modified = True

        if ej := cfg.get("enforce_json", None):
            self.cfg["enforce_json"] = ej
            cfg_modified = True

        if cfg_modified:
            self.p, self.in_queue, self.out_queue = self.create_batch_process_worker()

        # Process output
        outputs = []
        batch_start = 0

        while batch_start < len(prompts):
            assert self.p is not None
            assert self.in_queue is not None
            assert self.out_queue is not None

            prompt_batch = prompts[batch_start : min(len(prompts), batch_start + batch_size)]

            config.debug(f"Processing batch {batch_start}..{batch_start + batch_size} of {len(prompts)}")
            self.in_queue.put(prompt_batch)

            results = self.out_queue.get()
            if results["successful"]:
                outputs.extend(results["output"])
                batch_start += batch_size
            else:
                if self.p.is_alive():
                    self.kill_batch_worker()

                torch.cuda.empty_cache()

                self.p, self.in_queue, self.out_queue = self.create_batch_process_worker(enforce_json)

                new_batch_size = max(1, int(0.8 * batch_size))
                if new_batch_size == batch_size > 1:
                    new_batch_size -= 1

                batch_size = new_batch_size

        return outputs


    @config.debug_function
    def create_batch_process_worker(self) -> Tuple[mp.Process, mp.Queue, mp.Queue]:
        """
        Creates a new batch process worker

        returns:
        A tuple containing:
            1. The process
            2. The input queue
            3. The output queue
        """
        in_queue: mp.Queue = mp.Queue()
        out_queue: mp.Queue = mp.Queue()

        p = mp.Process(
            target=self.batch_process_worker,
            args=(in_queue, out_queue, self.model_id, self.cfg)
        )
        p.start()

        return p, in_queue, out_queue


    def batch_process_worker(
        cls, in_queue: mp.Queue, out_queue: mp.Queue, model_id: str, cfg: Dict[str, Any]
    ):
        """
        Creates a batch processing worker

        args:
        - input_queue: the queue this process take batches from
        - output_queue: the queue this process writes output to, as dicts:
            1. "successful": whether the processing was successful (or OOM)
            2. "output": the list of text output from the model
        - model_id: the model ID
        - cfg: the config dict
        """
        m = model.BatchModel(model_id)
        m.load_pre_prompt(cfg["pre_prompt_path"])

        with torch.no_grad():
            while True:
                if (batch := in_queue.get()) is None:
                    break

                try:
                    output = m.process_batch(batch, enforce_json=cfg['enforce_json'],
                                             max_new_tokens=cfg['max_new_tokens'])
                    out_queue.put({ "successful": True, "output": output })
                except RuntimeError as e:
                    if str(e).startswith('CUDA out of memory'):
                        out_queue.put({ "successful": False, "output": None })
                    else:
                        raise e

        torch.cuda.empty_cache()


    @config.debug_function
    def kill_batch_worker(self):
        """
        Kills the current batch worker and deletes the input and output queues
        """
        if self.p is not None and self.p.is_alive():
            self.p.terminate()
            self.p.join()
            self.p = None

        if self.in_queue is not None:
            self.in_queue.close()
            self.in_queue.join_thread()
            self.in_queue = None

        if self.out_queue is not None:
            self.out_queue.close()
            self.out_queue.join_thread()
            self.out_queue = None


class Experiment2:
    def __init__(self, small_model_max_tokens: int, large_model_max_tokens: int):
        config.debug("Loading dataset")

        self.data_dir = config.CODE_DIR / "data"
        self.reddit_df = pd.read_parquet(self.data_dir / "combined-filtered-reddit-data.parquet")
        self.reddit_df["date_posted"] = pd.to_datetime(self.reddit_df["created_utc"], unit="s")

        with open(self.data_dir / "subreddit-selection.json") as f:
            self.subreddit_selection = json.load(f)

        self.small_max_tokens = small_model_max_tokens
        self.large_max_tokens = large_model_max_tokens

        self.small_model_isolator = BatchModelIsolator("small")
        self.large_model_isolator = BatchModelIsolator("large")


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
            sample_idxs = np.random.choice(np.arange(0, n_posts), size=num_samples, replace=False)
            test_df = test_df.iloc[sample_idxs]

        return test_df


    @config.debug_function
    def generate_demographic_inferences(self, test_df: pd.DataFrame, batch_size: int = 20) -> pd.DataFrame:
        """
        Returns the df with demographic groups inferred

        args:
        - test_df: the dataframe to add demographic inferences to
        - batch_size: the size of each batch

        Post-processing:
        1. age is split into 5-year groupings
        2. if gender isn't strictly male or female, it is None
        """
        prompts = test_df.reset_index(drop=True).apply(self.row_to_prompt, axis=1).tolist()
        outputs = self.small_model_isolator.process_prompts(
            prompts,
            batch_size=batch_size,
            cfg={
                "enforce_json": True,
                "max_new_tokens": 20,
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
            try:
                if 0 <= (age_int := int(float(age))) <= len(bins):
                    return bins[age_int]
                else:
                    return "unknown"
            except (ValueError, TypeError):
                return "unknown"


        def format_gender(gender: Any) -> str:
            if isinstance(gender, str):
                gender = gender.lower()
                if gender in ['male', 'female']:
                    return gender
            return "unknown"


        bins = np.array([f"{5*(i // 5)}-{5*((i // 5) + 1)}" for i in range(100)])

        inference_df["predicted_age"] = inference_df["predicted_age"].apply(format_age)
        inference_df["predicted_gender"] = inference_df["predicted_gender"].apply(format_gender)

        return pd.concat([test_df, inference_df], axis=1)


    @config.debug_function
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

    @config.debug_function
    def get_trends_from_chunk(self, chunks: List[pd.DataFrame], batch_size = None) -> List[str]:
        """
        Obtain an enumeration of consumer trends indicated by a chunk of posts

        args:
        - chunks: a list of dataframes of posts from a specific subreddit and timeframe
        - batch_size: the number of chunks to process at once
    
        returns:
            A string containing a bullet-pointed list of trends indicated
        """
        prompts = [self.chunk_to_prompt(c) for c in chunks]
        outputs = self.small_model_isolator.process_prompts(
            prompts,
            batch_size=batch_size,
            cfg={
                "enforce_json": False,
                "max_new_tokens": self.small_max_tokens,
                "pre_prompt_name": "expt2-identify-trends-sector.txt"
            }
        )

        return outputs


    @config.debug_function
    def get_trends_from_reports(
        self, reports: List[str], subreddit: str | None, age_range: str, gender: str, batch_size: int
    ) -> str:
        """
        Uses the large model to produce a final report summarising consumer trends
        identified in the reports

        args:
        - reports: a list of reports made by the LLM
        - subreddit: the subreddit the report came from (nullable)
        - age_range: the inferred age range of the people who posted
        - gender: the inferred gender of the posters
        - batch_size: the number of reports to combine at each iteration
        """
        current_reports = reports[:]
        layers = 1

        query_context = (
            "metadata:\n"
            f" - all posts are from r/{subreddit}\n" if subreddit else ""
            f" - the inferred age range of the posters is {age_range}\n"
            f" - the inferred gender of the posters is {gender}"
        )

        while len(current_reports) > 1:
            config.debug(f"Creating a new layer of reports: layer = {layers}")

            reports_string = "\n".join(f"[report {i + 1}]:\n{r}" for i, r in enumerate(reports))
            reports = self.large_model_isolator.process_prompts(
                [query_context + reports_string],
                batch_size=batch_size,
                cfg={
                    "enforce_json": False,
                    "max_new_tokens": self.large_max_tokens,
                    "pre_prompt_name": "expt2-identify-trends-from-reports.txt"
                }
            )
            layers += 1

        return current_reports[0]


if __name__ == "__main__":
    mp.set_start_method("spawn")

    # Setup age and gender range
    age_range = np.array([f"{5*(i // 5)}-{5*((i // 5) + 1)}" for i in range(100)] + ["unknown"])
    gender_range = ["male", "female", "unknown"]

    expt = Experiment2(small_model_max_tokens=200, large_model_max_tokens=2000)

    subreddit_selection = expt.subreddit_selection

    relevant_df = expt.select_test_df("relevant", num_samples=5_000)
    irrelevant_df = expt.select_test_df("irrelevant")

    config.debug(f"len(relevant_df) = {len(relevant_df)}")
    relevant_df = expt.generate_demographic_inferences(relevant_df, batch_size=40)

    # We will focus on relevant subreddits
    trend_reports = {}

    for sub in subreddit_selection["relevant"]:
        for ages in age_range:
            for gender in gender_range:
                post_chunks = expt.create_balanced_post_selection(relevant_df, sub, ages, gender, 25)

                if len(post_chunks) == 0:
                    continue

                trend_reports[(sub, ages, gender)] = {
                    "reports": expt.get_trends_from_chunk(post_chunks, 4),
                    "start_date": str(post_chunks[0].iloc[0]["date_posted"]),
                    "end_date": str(post_chunks[-1].iloc[-1]["date_posted"])
                }

    expt.small_model_isolator.kill_batch_worker()

    with open(config.RESULTS_DIR / "experiment-2-trend-reports.json", "w") as f:
        json.dump(trend_reports, f)

    # Produce larger report
    larger_reports = {}

    for sub, ages, gender in trend_reports:
        config.output(f"Reports for subreddit r/{sub} with ages = {ages} and gender = {gender}:")
        config.output(f" - date range: {trend_reports[sub]['start_date']} to {trend_reports[sub]['end_date']}")
        config.output(f" - number of reports: {len(trend_reports[sub]['reports'])}")
        config.output(f" - total text: {len(' '.join(trend_reports[sub]['reports']))}")

        larger_reports[(sub, ages, gender)] = expt.get_trends_from_reports(trend_reports[sub]['reports'],
                                                           sub, ages, gender, 4)

        config.output(f" - overall report:\n{larger_reports[(sub, ages, gender)]}")

    with open(config.RESULTS_DIR / "experiment-2-overall-reports.json", "w") as f:
        json.dump(larger_reports, f)

    expt.large_model_isolator.kill_batch_worker()
