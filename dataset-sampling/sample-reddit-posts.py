#!/usr/bin/env python3
import os
import random
import heapq
import orjson
import pandas as pd
import multiprocessing as mp

SAMPLE_SIZE = 100_000
NUM_WORKERS = mp.cpu_count()
FILE_PATH = "RS_2025-03.json"  # Uncompressed JSONL file
CHUNK_SIZE = 1_000_000  # lines per chunk per worker

def sample_lines(lines, k):
    heap = []
    for line in lines:
        try:
            obj = orjson.loads(line)
        except:
            continue
        r = random.random()
        if len(heap) < k:
            heapq.heappush(heap, (r, obj))
        else:
            if r > heap[0][0]:
                heapq.heapreplace(heap, (r, obj))
    return heap

def worker(path, start_line, num_lines, k, return_dict, wid):
    heap = []
    with open(path, 'rb') as f:
        for _ in range(start_line):
            f.readline()
        lines = []
        for _ in range(num_lines):
            line = f.readline()
            if not line:
                break
            lines.append(line)
        heap = sample_lines(lines, k)
    return_dict[wid] = heap

def merge_heaps(heaps, k):
    merged = []
    for heap in heaps.values():
        merged.extend(heap)
    merged.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in merged[:k]]

def count_lines(path):
    with open(path, 'rb') as f:
        return sum(1 for _ in f)

if __name__ == "__main__":
    total_lines = count_lines(FILE_PATH)
    lines_per_worker = total_lines // NUM_WORKERS

    manager = mp.Manager()
    return_dict = manager.dict()
    jobs = []

    for i in range(NUM_WORKERS):
        start = i * lines_per_worker
        count = lines_per_worker if i < NUM_WORKERS - 1 else total_lines - start
        j = mp.Process(target=worker, args=(FILE_PATH, start, count, SAMPLE_SIZE, return_dict, i))
        jobs.append(j)
        j.start()

    for j in jobs:
        j.join()

    final_sample = merge_heaps(return_dict, SAMPLE_SIZE)
    df = pd.DataFrame(final_sample)
    df.to_parquet("sampled_reddit.parquet", compression='brotli')
