import datetime
import json
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys
from typing import Iterator, List, Literal, Tuple 

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
        with open(self.data_dir / "subreddit-selection.json") as f:
            self.subreddit_selection = json.load(f)

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
        self.m.set_pre_prompt(config.CODE_DIR / "pre-prompts" / "expt2-identify-trends-sector.txt")

        prompts = [self.chunk_to_prompt(chunks.iloc[i]) for i in range(len(chunks))]
        outputs = []

        if batch_size is None:
            batch_size = len(chunks)

        for batch_start in range(0, len(chunks), batch_size):
            prompt_batch = prompts[batch_start : min(len(chunks), batch_start + batch_size)]
            out = self.m.process_batch(prompt_batch, enforce_json=False, max_new_tokens=200)
            outputs.extend(out)

        return outputs


if __name__ == "__main__":
    mp.set_start_method("spawn")

    expt = Experiment2()

    relevant_df = expt.select_test_df("relevant", NUM_SAMPLES)
    irrelevant_df = expt.select_test_df("irrelevant", NUM_SAMPLES)

    post_chunks = expt.create_balanced_post_selection(relevant_df, "food", 25)

    relevant_post_outputs: List[str] = []

    for c in post_chunks:
        relevant_post_outputs.extend(expt.get_trends_from_chunk(c, 4))

    for i, out in enumerate(relevant_post_outputs):
        print(f"Output for chunk {i + 1}:\n{out}\n\n")



    """
    Approach:
    1. Load (a subset of) reddit posts (submissions & comments)
    3. Split up by time (4 days) and subreddit
    4. For each time and subreddit, produce a report of all consumer
       trends (relating to Coca-Cola) indicated by the post
    5. 
    """

