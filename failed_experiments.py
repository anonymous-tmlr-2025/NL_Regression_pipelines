import json
from pathlib import Path

def delete_experiments_that_failed(dataset: str) -> tuple[dict, list[Path]]:
    log_path = Path(f"results/{dataset}/experiment_log.json")
    with open(log_path) as f:
        log: dict = json.load(f)
    failed_experiments = [v for v in log.values() if v["status"] == "failed"]
    log_without_failed = {k: v for k, v in log.items() if v["status"] != "failed"}
    num_removed = len(log) - len(log_without_failed)
    assert num_removed == len(failed_experiments)
    return log_without_failed, failed_experiments

datasets = ["jc_penney_products", "online_boat_listings"]
dataset = datasets[1]
log_without_failed, failed_experiments = delete_experiments_that_failed(dataset)
print(f"Removed {len(failed_experiments)} experiments from {dataset}")
with open(f"failed_experiments_{dataset}.json", "w") as f:
    json.dump(failed_experiments, f, indent=0)
with open(Path(f"results/{dataset}/experiment_log.json"), "w") as f:
    json.dump(log_without_failed, f, indent=0)