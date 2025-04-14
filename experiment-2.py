import datetime
import json
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys
from typing import Iterator, Tuple

import config
import dataset
import model

LOCAL_TESTING = False
NUM_SAMPLES = None


class Experiment2:
    def __init__(self):
        config.debug("Loading dataset")

        data_path = config.CODE_DIR / "data" / "combined-filtered-reddit-data.parquet"

        self.reddit_df = pd.read_parquet(data_path)
        self.reddit_df["date_posted"] = pd.to_datetime(self.reddit_df["created_utc"], unit="s")

        self.m = model.BatchModel(config.MODEL)


    def sample_reddit_df(self, num_samples: int):
        """
        Selects a subset of size num_samples for reddit_df
        """
        n_posts = len(self.reddit_df)
        idxs = np.random.choice(np.arange(0, n_posts), size=num_samples, replace=False)
        self.reddit_df = self.reddit_df[idxs]


    @config.debug_function
    def chunk_by_date_and_subreddit(self, chunk_size: int) -> Iterator[pd.DataFrame]:
        """
        Iterates over dates and subreddits

        args:
        - df_test: the dataframe to divide into chunks
        - chunk_size: the size *in days* of each chunk
        """
        start_date = self.reddit_df["date_posted"].min()
        end_date = self.reddit_df["date_posted"].max()

        time_sorted_df = self.reddit_df.sort_values(by="date_posted")
        subreddits = self.reddit_df["subreddit"].unique()

        current_start = start_date

        while current_start < end_date:
            current_end = current_start + datetime.timedelta(days=chunk_size)
            time_mask = (current_start <= time_sorted_df["date_posted"]) & \
                        (time_sorted_df["date_posted"] < current_end)

            for sub in subreddits:
                sub_mask = time_sorted_df["subreddit"] == sub
                chunk = time_sorted_df[time_mask & sub_mask]

                if not chunk.empty:
                    yield chunk

            current_start = current_end


    @config.debug_function
    def get_trends_from_chunk(self, chunk: pd.DataFrame) -> str:
        """
        Obtain an enumeration of consumer trends indicated by a chunk of posts

        args:
        - chunk: a dataframe of posts from a specific subreddit and timeframe
    
        returns:
            A string containing a bullet-pointed list of trends indicated
        """
        self.m.set_pre_prompt(config.CODE_DIR / "pre-prompts" / "expt2-identify-trends-sector.txt")

        subreddit = chunk["subreddit"].iloc[0]

        prompt = f"all posts are from r/{subreddit}\n--- BEGIN INPUTS ---"

        for i in range(len(chunk)):
            post = chunk.iloc[i]

            prompt += json.dumps({
                "date": post["date_posted"].strftime('%Y-%m-%d'),
                "karma": str(post["score"]),
                "type": "text-post" if post["type"][0] == "s" else "comment",
                "content": repr(post["text"])
            }, indent=4)

        prompt += "\n--- END INPUTS ---"

        output = self.m.process_batch([prompt], enforce_json=False, max_new_tokens=200)

        config.debug(f"The prompt has length {len(prompt)}")

        return output[0]


if __name__ == "__main__":
    mp.set_start_method("spawn")

    expt = Experiment2()

    if NUM_SAMPLES is not None:
        expt.sample_reddit_df(NUM_SAMPLES)

    post_chunks = expt.chunk_by_date_and_subreddit(4)

    for c in post_chunks:
        output = expt.get_trends_from_chunk(c)

        print("=====================\n" + output + "\n=================")


    """
    Approach:
    1. Load (a subset of) reddit posts (submissions & comments)
    2. filter subreddits with LLM
    3. Split up by time (4 days) and subreddit
    4. For each time and subreddit, produce a report of all 
    """

