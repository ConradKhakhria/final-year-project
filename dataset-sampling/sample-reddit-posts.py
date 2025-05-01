#!/usr/bin/env python3
import os
import json
import random
import heapq
import multiprocessing as mp
from multiprocessing import Queue, Process
import zstandard as zstd
import pandas as pd

SAMPLE_SIZE = 100_000
NUM_WORKERS = mp.cpu_count()
ZST_PATH = "reddit_data/reddit/submissions/RS_2025-03.zst"
QUEUE_MAXSIZE = 10000

def read_lines_zst(path, out_queue, sentinel, verbose=True):
    with open(path, 'rb') as f:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(f) as reader:
            buf = b''
            count = 0
            while True:
                chunk = reader.read(2**27)
                if not chunk:
                    break
                buf += chunk
                lines = buf.split(b'\n')
                for line in lines[:-1]:
                    out_queue.put(line)
                    count += 1
                    if verbose and count % 100000 == 0:
                        print(f"[Producer] {count:,} lines queued")
                buf = lines[-1]
    for _ in range(NUM_WORKERS):
        out_queue.put(sentinel)

def worker(worker_id, in_queue, sentinel, k, return_dict):
    heap = []
    count = 0
    while True:
        line = in_queue.get()
        if line is sentinel:
            break
        try:
            obj = json.loads(line.decode('utf-8'))
        except:
            continue
        r = random.random()
        if len(heap) < k:
            heapq.heappush(heap, (r, obj))
        else:
            if r > heap[0][0]:
                heapq.heapreplace(heap, (r, obj))
        count += 1
        if count % 100000 == 0:
            print(f"[Worker {worker_id}] {count:,} processed")
    return_dict[worker_id] = heap

def merge_heaps(heaps, k):
    combined = []
    for h in heaps.values():
        combined.extend(h)
    combined.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in combined[:k]]

if __name__ == "__main__":
    manager = mp.Manager()
    q = mp.Queue(maxsize=QUEUE_MAXSIZE)
    sentinel = b"__SENTINEL__"

    return_dict = manager.dict()

    producer = Process(target=read_lines_zst, args=(ZST_PATH, q, sentinel))
    workers = [
        Process(target=worker, args=(i, q, sentinel, SAMPLE_SIZE, return_dict))
        for i in range(NUM_WORKERS)
    ]

    producer.start()
    for w in workers:
        w.start()

    producer.join()
    for w in workers:
        w.join()

    final_sample = merge_heaps(return_dict, SAMPLE_SIZE)
    df = pd.DataFrame(final_sample)
    df.to_parquet("sampled_reddit.parquet", compression="brotli")
