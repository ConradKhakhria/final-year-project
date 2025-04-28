# mypy: ignore-errors
import zstandard
import os
import json
import random
import pandas as pd

def read_lines_zst(file_path):
    with open(file_path, 'rb') as fh:
        dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
        reader = dctx.stream_reader(fh)
        buffer = b''
        while True:
            chunk = reader.read(2**27)  # 128 MB
            if not chunk:
                break
            buffer += chunk
            lines = buffer.split(b'\n')
            for line in lines[:-1]:
                yield line
            buffer = lines[-1]
        reader.close()

def sample_from_files(file_paths, sample_size):
    reservoir = []
    total = 0
    for file_path in file_paths:
        print(f"Reading from {file_path}")
        for raw_line in read_lines_zst(file_path):
            total += 1
            try:
                line = raw_line.decode('utf-8')
                obj = json.loads(line)
            except Exception:
                continue
            if len(reservoir) < sample_size:
                reservoir.append(obj)
            else:
                idx = random.randint(0, total)
                if idx < sample_size:
                    reservoir[idx] = obj
            if total % 100000 == 0:
                print(f"Processed: {total:,} lines")
    print(f"Sampled {len(reservoir)} items.")
    return reservoir

if __name__ == "__main__":
    paths = [
        "reddit_data/RC_2016-01.zst",
        "reddit_data/RC_2016-02.zst",
        "reddit_data/RC_2016-03.zst"
    ]

    sample_size = 100000
    output_path = "sampled_reddit.parquet"

    sampled_data = sample_from_files(paths, sample_size)

    print("Converting to DataFrame...")
    df = pd.DataFrame(sampled_data)

    print(f"Saving to Parquet: {output_path}")
    df.to_parquet(output_path, compression='brotli')
    print("Done.")
