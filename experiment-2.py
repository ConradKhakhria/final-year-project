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
NUM_SUBREDDITS = 5000

# Local config
SUBREDDIT_SELECTOR_PRE_PROMPT = "expt2-select-subreddits.txt"


class Experiment2:
    def __init__(self):
        config.debug("Loading dataset")

        if LOCAL_TESTING:
            reddit_submissions_path = "../data/reddit/sampled_reddit_submissions.jsonl"
            reddit_comments_path = "../data/reddit/sampled_reddit_comments.jsonl"
        else:
            reddit_submissions_path = config.DATASET_DIR / "reddit" / "sampled_reddit_submissions.jsonl"
            reddit_comments_path = config.DATASET_DIR / "reddit" / "sampled_reddit_comments.jsonl"

        self.reddit_df = dataset.load_reddit_submissions_comments(reddit_submissions_path, reddit_comments_path)
        self.reddit_df["date_posted"] = pd.to_datetime(self.reddit_df["created_utc"], unit="s")


    @config.debug_function
    def filter_relevant_subreddits(self, batch_size: int = 20) -> pd.DataFrame:
        """
        Filters test_df to contain only relevant subreddits, as chosen by the LLM

        args:
        - batch_size: the number of subreddit names to process at once

        returns:
            The entire dataset but only containing relevant subreddits
        """
        global SUBREDDIT_SELECTOR_PRE_PROMPT

        subreddit_selection_path = config.CACHE_DIR / "selected-subreddits.json"

        if subreddit_selection_path.exists():
            with open(subreddit_selection_path) as f:
                selected_subreddit_names = json.load(f)
        else:
            m = model.BatchModel(config.MODEL)
            m.load_pre_prompt(config.CODE_DIR / "pre-prompts" / SUBREDDIT_SELECTOR_PRE_PROMPT)

            subreddits = self.reddit_df["subreddit"].unique()

            # For testing we will take only a few of these
            if NUM_SUBREDDITS is not None:
                subreddits = subreddits[:NUM_SUBREDDITS]

            n_subs = len(subreddits)
            batch_start = 0

            selected_subreddit_names = []
            unsuccessful_output_count = 0

            # Obtain all useful subreddits
            while batch_start < n_subs:
                batch = subreddits[batch_start : batch_start + batch_size]

                config.debug(f"Processing batch {batch_start}..{batch_start + batch_size} of {n_subs}")
                output = m.process_batch(batch, enforce_json=True)
                json_output = model.extract_json(output, { "name": None, "useful": None })

                for i, o in enumerate(json_output):
                    if o["useful"] is not None:
                        if o["useful"]:
                            selected_subreddit_names.append(batch[i])
                    else:
                        unsuccessful_output_count += 1

                batch_start += batch_size

            config.debug(f"There were {unsuccessful_output_count} unsuccessful batches")

            # record this list of subreddits
            with open(subreddit_selection_path, "w") as f:
                json.dump(selected_subreddit_names, f)

        return self.reddit_df[self.reddit_df["subreddit"].isin(selected_subreddit_names)]


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
        m = model.BatchModel(config.MODEL)
        m.set_pre_prompt(config.CODE_DIR / "pre-prompts" / "expt2-identify-trends-sector.txt")

        subreddit = chunk["subreddit"].iloc[0]

        prompt = f"all posts are from r/{subreddit}"

        for i in range(len(chunk)):
            post = chunk.iloc[i]

            prompt += json.dumps({
                "date": post["date_posted"],
                "karma": post["score"],
                "type": "text-post" if post["type"][0] == "s" else "comment",
                "content": repr(post["text"])
            }, indent=4)

        output = m.process_batch([prompt], enforce_json=False, max_new_tokens=50)

        return output[0]



if __name__ == "__main__":
    mp.set_start_method("spawn")

    expt = Experiment2()

    filtered_subreddits_df = expt.filter_relevant_subreddits(batch_size=100)
    N = len(filtered_subreddits_df)

    if NUM_SAMPLES is None:
        selected_idxs = np.arange(0, N)
    else:
        selected_idxs = np.random.choice(np.arange(0, N), size=NUM_SAMPLES, replace=False)

    filtered_subreddits_df = filtered_subreddits_df.iloc[selected_idxs]
    post_chunks = expt.chunk_by_date_and_subreddit(filtered_subreddits_df, 4)

    for c in post_chunks:
        output = expt.get_trends_from_chunk(c)

        print(output)

        exit()


    """
    Approach:
    1. Load (a subset of) reddit posts (submissions & comments)
    2. filter subreddits with LLM
    3. Split up by time (4 days) and subreddit
    4. For each time and subreddit, produce a report of all 
    """

