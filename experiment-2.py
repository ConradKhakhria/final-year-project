import datetime
import json
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys
import torch
from typing import Iterator, List, Literal, Tuple 

import config
import dataset
import model

NUM_SAMPLES = None


class BatchModelIsolator:
    def __init__(self, pre_prompt_name: str):
        self.pre_prompt_path = config.CODE_DIR / "pre-prompts" / pre_prompt_name


    def process_prompts(self, prompts: List[str], batch_size = None) -> List[str]:
        """
        Processes a list of prompts in a separate process

        args:
        - prompts: the list of prompts to process
        - batch_size: the batch size (default to sequential)
        """ 
        outputs = []
        batch_start = 0

        if batch_size is None:
            batch_size = 1

        p, in_queue, out_queue = self.create_batch_process_worker()

        while batch_start < len(prompts):
            prompt_batch = prompts[batch_start : min(len(prompts), batch_start + batch_size)]

            config.debug(f"Processing batch {batch_start}..{batch_start + batch_size} of {len(prompts)}")
            in_queue.put(prompt_batch)

            results = out_queue.get()
            if results["successful"]:
                outputs.extend(results["output"])
                batch_start += batch_size
            else:
                if p.is_alive():
                    p.terminate()
                    p.join()

                torch.cuda.empty_stack()

                p, in_queue, out_queue = self.create_batch_process_worker()

                new_batch_size = max(1, int(0.8 * batch_size))
                if new_batch_size == batch_size > 1:
                    new_batch_size -= 1

                batch_size = new_batch_size

        in_queue.put(None)
        p.join()

        return outputs


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
            args=(in_queue, out_queue, self.pre_prompt_path)
        )
        p.start()

        return p, in_queue, out_queue


    @classmethod
    def batch_process_worker(cls, in_queue: mp.Queue, out_queue: mp.Queue, pp_path: Path):
        """
        Creates a batch processing worker

        args:
        - input_queue: the queue this process take batches from
        - output_queue: the queue this process writes output to, as dicts:
            - "successful": whether the processing was successful (or OOM)
            - "output": the list of text output from the model
        - pp_path: pre-prompt path
        """
        m = model.BatchModel(config.SMALL_MODEL)
        m.load_pre_prompt(pp_path)

        with torch.no_grad():
            while True:
                if (batch := in_queue.get()) is None:
                    break

                try:
                    output = m.process_batch(batch, enforce_json=True)
                    out_queue.put({ "successful": True, "output": output })
                except RuntimeError as e:
                    if str(e).startswith('CUDA out of memory'):
                        out_queue.put({ "successful": False, "output": None })
                    else:
                        raise e

        torch.cuda.empty_cache()



class Experiment2:
    def __init__(self):
        config.debug("Loading dataset")

        self.data_dir = config.CODE_DIR / "data"
        self.reddit_df = pd.read_parquet(self.data_dir / "combined-filtered-reddit-data.parquet")
        self.reddit_df["date_posted"] = pd.to_datetime(self.reddit_df["created_utc"], unit="s")

        with open(self.data_dir / "subreddit-selection.json") as f:
            self.subreddit_selection = json.load(f)

        self.model_isolator = BatchModelIsolator("expt2-identify-trends-sector.txt")


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
    def create_balanced_post_selection(
        self, df_test: pd.DataFrame, subreddit: str, n_posts: int
    ) -> Iterator[pd.DataFrame]:
        """
        Yields dataframes each containing close to n_posts

        args:
        - df_test: the dataframe to select from
        - subreddit: the subreddit to select from
        - n_posts: the number of posts for each df

        All posts will be in time-order
        """
        time_sorted = df_test[df_test["subreddit"] == subreddit].sort_values(by="date_posted")
        df_len = len(time_sorted)

        for i_start in range(0, df_len, n_posts):
            yield time_sorted.iloc[i_start : min(df_len, i_start + n_posts)]


    def chunk_to_prompt(self, chunk: pd.DataFrame) -> str:
        """
        Turns a chunk of posts into a prompt string
        """
        start_date = chunk.iloc[0]["date_posted"]
        end_date = chunk.iloc[-1]["date_posted"]
        subreddit = chunk["subreddit"].iloc[0]

        prompt = f"All supplied posts will be from r/{subreddit}. " \
                 f"They were posted between {start_date} and {end_date}"

        for i in range(len(chunk)):
            post = chunk.iloc[i]

            prompt += f"[post number {i + 1}]:\n"
            prompt += json.dumps({
                "date": post["date_posted"].strftime('%Y-%m-%d'),
                "karma": str(post["score"]),
                "type": "text-post" if post["type"][0] == "s" else "comment",
                "content": repr(post["text"])
            }, indent=4)
            prompt += "\n"

        return prompt


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
        outputs = self.model_isolator.process_prompts(prompts, batch_size=batch_size)

        return outputs


if __name__ == "__main__":
    mp.set_start_method("spawn")

    expt = Experiment2()

    subreddit_selection = expt.subreddit_selection

    relevant_df = expt.select_test_df("relevant", NUM_SAMPLES)
    irrelevant_df = expt.select_test_df("irrelevant", NUM_SAMPLES)

    # We will focus on relevant subreddits
    subreddit_trend_repots = {}

    for sub in subreddit_selection["relevant"][:5]:
        post_chunks = expt.create_balanced_post_selection(relevant_df, sub, 25)
        post_repots = expt.get_trends_from_chunk(list(post_chunks), 4)
        subreddit_trend_repots[sub] = post_repots

    for sub in subreddit_trend_repots:
        print(f"Here are repots from subreddit {sub}")
        for i, report in enumerate(subreddit_trend_repots[sub]):
            print(f"[report {i + 1}]:\n{report}")


    """
    Approach:
    1. Load (a subset of) reddit posts (submissions & comments)
    3. Split up by time (4 days) and subreddit
    4. For each time and subreddit, produce a report of all consumer
       trends (relating to Coca-Cola) indicated by the post
    5. 
    """

