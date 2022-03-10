import json

import pandas as pd

results = []
with open("t5results.txt") as results_file:
    for line in results_file:
        try:
            jll = json.loads(line)
            results.append(jll)
        except:
            print("Skipping",line)
df = pd.DataFrame()
for result in results:
    result["hint"] = not "no_hint" in result["file"]
    result["epoch"] = result["file"].split("_")[-1].replace("e","").replace(".jsonl","")
    result["model"] = result["file"].split("beam_outputs_base_")[1].split("_")[0]
    result["variation"] = result["file"].split("_")[-2].replace("hint","")
    if result["variation"] == "":
        result["variation"] = "0"
    df = df.append(result,ignore_index=True)
print(df)
df.to_csv("t5_results.csv",index=False)

