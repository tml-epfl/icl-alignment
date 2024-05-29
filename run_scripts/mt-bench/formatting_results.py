"""
The code is taken from the URIAL repository.
https://github.com/Re-Align/URIAL/blob/main/run_scripts/mt-bench/formatting_results.py
"""

import json
import os
import random
import string
import sys

# replace the answer_dir using your own dir
answer_dir = "/tmlscratch/hzhao/FastChat/fastchat/llm_judge/data/mt_bench/model_answer" 

model_name = sys.argv[1]
# suffix = sys.argv[2]
suffix = "0210v1"
output_folder = f"result_dirs/mt-bench/urial_bench/{model_name}"
model_id = f"{model_name}-URIAL-{suffix}"
turn1_path = f"result_dirs/mt-bench/urial_bench/{model_name}/{model_name}.turn1.json"
turn2_path = f"result_dirs/mt-bench/urial_bench/{model_name}/{model_name}.turn2.json"

turn1_results = json.load(open(turn1_path))
turn2_results = json.load(open(turn2_path))

results = []
for item1, item2 in zip(turn1_results, turn2_results):
    assert item1["question_id"] == item2["question_id"]
    res_item = {}
    res_item["question_id"] = item1["question_id"]
    # generate a random string
    res_item["answer_id"] = ''.join(random.choices(string.ascii_uppercase + string.digits, k=22))
    res_item["model_id"] = model_id
    res_item["choices"] = [
        {
            "index": 0,
            "turns": [
                item1["turn1_output"].replace("<|endoftext|>", "").strip(),
                item2["turn2_output"].replace("<|endoftext|>", "").strip()
            ]
        }
    ]
    results.append(res_item)
    
with open(f"{output_folder}/{model_id}.jsonl", "w") as f:
    for item in results:
        f.write(json.dumps(item) + "\n")

# copy the file to `fastchat/llm_judge/data/mt_bench/model_answer/`

os.system(f"cp {output_folder}/{model_id}.jsonl {answer_dir}")
