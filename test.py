import json

with open("stage-3-fucking.json") as f:
    res = json.load(f)

for d in res:
    print(d)
    for r in res[d]:
        print(r)