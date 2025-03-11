from copy import deepcopy
import json
from pathlib import Path
from generate_experiment import hash_experiment_config
import re

def find_completed_experiments(dataset: str) -> dict:
    results_path = Path(f"results/{dataset}/")
    for path in results_path.glob("**/results.json"):
        with open(path) as f:
            seed = re.search(r"seed_(\d+)", str(path))  # Get part after "seed_" using regex
            if not seed:
                print(f"No seed found for {path}")
            else:
                seed = seed.group(1)
            yield json.load(f), seed, path

def existing_log(dataset: str) -> dict:
    return json.loads(Path(f"results/{dataset}/experiment_log.json").read_text())

datasets = ["jc_penney_products", "online_boat_listings"]
dataset = datasets[0]
log = existing_log(dataset)
shared = {}
only_in_results = {}
deletable_results = []
only_in_log = deepcopy(log)
for experiment, seed, path in find_completed_experiments(dataset):
    config = {
        "preprocesser_order": experiment["preprocessers"],
        "tokeniser": experiment["tokeniser"],
        "featuriser": experiment["featuriser"],
        "model": experiment["model"],
        "finetune": experiment["finetune"],
        "seed": int(seed),
    }
    hash = hash_experiment_config(config)
    config["status"] = "success"
    config["error"] = None
    if hash in log:
        config["status"] = log[hash]["status"]
        config["error"] = log[hash]["error"]
        shared[hash] = config
        del only_in_log[hash]
    else:
        only_in_results[hash] = config
        deletable_results.append(path)

print(f"Shared: {len(shared)}")
print(f"Only in results: {len(only_in_results)}")
print(f"Only in log: {len(only_in_log)}")

if only_in_log:
    with open(f"{dataset}_log_only.json", "w") as f:
        json.dump(only_in_log, f, indent=0)
if only_in_results:
    with open(f"{dataset}_results_only.json", "w") as f:
        json.dump(only_in_results, f, indent=0)
if shared:
    with open(f"{dataset}_shared.json", "w") as f:
        json.dump(shared, f, indent=0)

if deletable_results:
    print(f"Found {len(deletable_results)} results not in experiment log.")
    with open("to_delete.txt", "w") as f:
        for path in deletable_results:
            f.write(f"{path}\n")
    delete_yn = input("Delete these results - check in to_delete.txt? (y/n): ")
    if delete_yn == "y":
        for path in deletable_results:
            path.unlink()
    else:
        print("Results not deleted")
