import json

def count_success(look_in: str, dataset: str) -> int:
    with open(f"{look_in}/{dataset}/experiment_log.json", "r") as f:
        results = json.load(f)
    return len([r for r in results.values() if r["status"] == "success"])

def count_per_seed(look_in: str, dataset: str, status: str) -> dict[int, int]:
    with open(f"{look_in}/{dataset}/experiment_log.json", "r") as f:
        results = json.load(f)
    all_seeds = set([r["seed"] for r in results.values()])
    return {seed: len([r for r in results.values() if r["seed"] == seed and r["status"] == status]) for seed in all_seeds}

datasets = [
    "jc_penney_products",
    "online_boat_listings",
    "california_house_prices",
]
look_ins = [
    "results",
    # "results/downsample_0.2",
    # "no_preprocessers",
    # "results/normalised_response",
]
for dataset in datasets:
    for look_in in look_ins:
        print(f"{dataset} - {look_in}: {count_success(look_in, dataset)}")
        print(f"{dataset} - {look_in} - unique seeds - success: {count_per_seed(look_in, dataset, 'success')}")
        print(f"{dataset} - {look_in} - unique seeds - failure: {count_per_seed(look_in, dataset, 'failure')}")
