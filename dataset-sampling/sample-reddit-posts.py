#!/usr/bin/env python3
import os
import json
import random
import heapq
import multiprocessing as mp

import zstandard as zstd
import pandas as pd

def read_lines_zst(path):
    with open(path, 'rb') as f:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(f) as reader:
            buf = b''
            while True:
                chunk = reader.read(2**27)
                if not chunk:
                    break
                buf += chunk
                lines = buf.split(b'\n')
                for line in lines[:-1]:
                    yield line
                buf = lines[-1]

def sample_worker(args):
    import time
    path, k = args
    heap = []
    count = 0
    last_report = time.time()
    print(f"[{os.getpid()}] Sampling from {path}")

    for raw in read_lines_zst(path):
        count += 1
        try:
            obj = json.loads(raw.decode('utf-8'))
        except:
            continue
        r = random.random()
        if len(heap) < k:
            heapq.heappush(heap, (r, obj))
        else:
            if r > heap[0][0]:
                heapq.heapreplace(heap, (r, obj))
        if count % 100000 == 0:
            now = time.time()
            print(f"[{os.getpid()}] {count:,} lines processed ({int(now - last_report)}s since last report)")
            last_report = now

    print(f"[{os.getpid()}] Finished {path} with {count:,} total lines, {len(heap)} samples.")
    return heap

def merge_heaps(heaps, k):
    merged = []
    for heap in heaps:
        merged.extend(heap)
    merged.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in merged[:k]]

if __name__ == "__main__":
    input_paths = [
#        "reddit_data/reddit/comments/RC_2025-03.zst",
        "reddit_data/reddit/submissions/RS_2025-03.zst"
    ]
    total_sample = 100_000

    pool = mp.Pool(processes=len(input_paths))
    heaps = pool.map(sample_worker, [(p, total_sample) for p in input_paths])
    pool.close()
    pool.join()

    sampled = merge_heaps(heaps, total_sample)
    df = pd.DataFrame(sampled)
    df.to_parquet("sampled_reddit.parquet", compression='brotli')
