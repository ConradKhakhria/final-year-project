# Introduction

This directory contains all of the code that is needed to run all three stages of the dissertation.

# Note
This code was tested on two different platforms, both using Ubuntu 22:
- All code using vLLM was run on a Lambda Labs instance, with an attached A100 GPU.
- All code using Groq was run on an x86_64 machine with no GPU.
- All shallow learning code was run on a Paperspace instance, with no GPU.

# Setup
For each version of the experiments, a setup script is provided:
- groq-setup.sh: This installs all required libraries for the Groq experiments
- vllm-setup.sh: This installs all required libraries for the vLLM experiments
- shallow-setup.sh: This installs all required libraries for the shallow learning experiments

# Running Experiments
The Huggingface token should be stored in ./hf-access-token.txt
The Groq token should be stored in ./groq-access-token.txt

Experiment 1 uses the Blog Authorship Corpus corpus from Huggingface, and therefore requires a valid
Huggingface token. The Groq version of experiment 1 requires a valid Groq API key as well.

Experiments 2 and 3 vLLM require a valid Huggingface token. The Groq version only requires a valid Groq API key.

The second and third experiments are identical, but use different data. The dataset has not been included
for privacy reasons, but its schema is as follows:

1. *File Format*: Must be in .parquet format.
2. *Essential Columns*:
- text: Raw post content.
- created_utc: Unix timestamp of post creation; used for chronological batching.
- subreddit: Name of subreddit from which post was taken.
- predicted_age: Age group label (string) predicted for the post author.
- predicted_gender: Gender label (string) predicted for the post author.
(note: the predicted_age and predicted_gender only need to be included for the Groq version)
3. *File Location*: each file should be located in:
    config.RESULTS_DIR / "combined-filtered-reddit-data.parquet"
(note: the 2025 dataset is located in"2025-combined-filtered-reddit-data.parquet")
