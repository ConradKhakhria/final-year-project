import datetime
import json
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys
from typing import Iterator, Literal, Tuple 

import config
import dataset
import model

LOCAL_TESTING = False
NUM_SAMPLES = None


class Experiment2:
    def __init__(self):
        config.debug("Loading dataset")

        self.data_dir = config.CODE_DIR / "data"
        self.reddit_df = pd.read_parquet(self.data_dir / "combined-filtered-reddit-data.parquet")
        self.reddit_df["date_posted"] = pd.to_datetime(self.reddit_df["created_utc"], unit="s")

        self.m = model.BatchModel(config.MODEL)


    def select_test_df(
        self, relevance: Literal["relevant", "irrelevant"], num_samples: int | None = None
    ) -> pd.DataFrame:
        """
        Selects the testing df

        args:
        - relevance: whether to select the relevant or irrelevant subreddits
        - num_samples: optional - whether to take a subset
        """
        with open(self.data_dir / "subreddit-selection.json") as f:
            subreddit_selection = json.load(f)

        test_df = self.reddit_df[self.reddit_df["subreddit"].isin(subreddit_selection[relevance])]

        if num_samples is not None:
            n_posts = len(test_df)
            sample_idxs = np.random.choice(np.arange(0, n_posts), size=num_samples, replace=False)
            test_df = test_df.iloc[sample_idxs]

        return test_df


    @config.debug_function
    def chunk_by_date_and_subreddit(self, df_test: pd.DataFrame, chunk_size: int) -> Iterator[pd.DataFrame]:
        """
        Iterates over dates and subreddits

        args:
        - df_test: the dataframe to divide into chunks
        - chunk_size: the size *in days* of each chunk
        """
        start_date = df_test["date_posted"].min()
        end_date = df_test["date_posted"].max()

        time_sorted_df = df_test.sort_values(by="date_posted")
        subreddits = df_test["subreddit"].unique()

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

    relevant_df = expt.select_test_df("relevant", NUM_SAMPLES)
    irrelevant_df = expt.select_test_df("irrelevant", NUM_SAMPLES)

    post_chunks = expt.chunk_by_date_and_subreddit(4)

    for c in post_chunks:
        output = expt.get_trends_from_chunk(c)

        print("=====================\n" + output + "\n=================")


    """
    Approach:
    1. Load (a subset of) reddit posts (submissions & comments)
    3. Split up by time (4 days) and subreddit
    4. For each time and subreddit, produce a report of all consumer
       trends (relating to Coca-Cola) indicated by the post
    5. 
    """

