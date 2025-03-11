import json
from collections import defaultdict

import argparse

parser = argparse.ArgumentParser(description='Check experiment log for duplicate configurations')
parser.add_argument('-v', '--verbose', action='store_true',
                   help='Print detailed information about each duplicate found')
parser.add_argument('path', type=str, help='Path to experiment log file')

args = parser.parse_args()


# Load experiment log
with open(args.path) as f:
    experiment_log = json.load(f)

# Group experiments by config excluding seed
experiment_groups = defaultdict(list)
for exp_id, exp_config in experiment_log.items():
    # Create key from config excluding seed
    config_key = (
        tuple(exp_config["preprocesser_order"]),
        exp_config["tokeniser"],
        exp_config["featuriser"], 
        exp_config["model"],
        exp_config["finetune"]
    )
    experiment_groups[config_key].append(exp_id)

# Check for duplicates
duplicates_found = False
num_duplicates = 0
sets_of_duplicates = set()
for config, exp_ids in experiment_groups.items():
    dup_seeds = set()
    if len(exp_ids) > 1:
        duplicates_found = True
        num_duplicates += 1
        if args.verbose:
            print(f"\nFound {len(exp_ids)} duplicate experiments with config:")
            print(f"Preprocessers: {config[0]}")
            print(f"Tokeniser: {config[1]}")
            print(f"Featuriser: {config[2]}")
            print(f"Model: {config[3]}")
            print(f"Finetune: {config[4]}")
            print("Experiment IDs:")
        for exp_id in exp_ids:
            seed = experiment_log[exp_id]['seed']
            dup_seeds.add(seed)
            if args.verbose:
                print(f"- {exp_id} (seed: {seed})")
        sets_of_duplicates.add(frozenset(dup_seeds))

print(f"Total duplicate experiments found: {num_duplicates}")
print(f"Total sets of duplicates found: {len(sets_of_duplicates)}:")
print(f"Sets of duplicates: {sets_of_duplicates}")

if not duplicates_found:
    print("No duplicate experiments found (excluding seed)")

delete = True
while delete:
    new_log = experiment_log.copy()
    delete = input("Delete duplicates? (y/n): ")
    delete = delete == "y"
    if delete:
        which_to_delete = input("Which duplicates to delete? (csv set of seeds):")
        seeds_to_delete = set(int(s.strip()) for s in which_to_delete.split(","))
        print(f"Deleting seeds: {seeds_to_delete}")
        # Only delete if seeds match exactly
        for config, exp_ids in experiment_groups.items():
            if len(exp_ids) > 1:
                # Get seeds for this group
                group_seeds = set(experiment_log[exp_id]['seed'] for exp_id in exp_ids)
                
                # Only delete if seeds match exactly what was requested
                if group_seeds == seeds_to_delete:
                    # Delete all experiment with these seeds
                    for exp_id in exp_ids:
                        del new_log[exp_id]
                    print(f"Deleted {len(exp_ids)} experiments with seeds {seeds_to_delete}")
        with open("new_log.json", "w") as f:
            json.dump(new_log, f, indent=0)
    else:
        print("No duplicates deleted")
        break
