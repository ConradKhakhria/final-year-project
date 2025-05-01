import json
from pathlib import Path
import sys

if len(sys.argv) != 4:
    print("Usage: py display-stage-3-results.py <relevant|irrelevant> <historical|current> <2|3>")
    exit()

section, period, stage = sys.argv[1:]

path = Path.home() / "UCL" / "FYP" / "code" / "groq-results" / "experiment-2"

with open(path / f"{section}-{period}-overall-stage-{stage}-results.json") as f:
    results = json.load(f)


for group in results:

    if stage == "2":
        rs = results[group]

        for r in rs:
            if r.get("label", "no_trends") == "no_trends":
                continue
            for k in r:
                print(f"    {k}: {r[k]}")
    else:
        r = json.loads(results[group])

        if r["evidence"] == []:
            continue

        print(group)

        for k in r:
            print(f"    {k}: {r[k]}")

