from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path", type=Path, help="Path to directory to clean")
args = parser.parse_args()

to_delete = []
all_folder_with_results = args.path.glob("**/**/results.json")
for result_path in all_folder_with_results:
    for path in result_path.parent.glob("*"):
        if path.suffix not in [".json", ".lock"] and path.is_file():
            if ("preds" not in path.name) and ("experiment_log" not in path.name):
                to_delete.append(path)

with open("to_delete.txt", "w") as f:
    for path in to_delete:
        f.write(f"{path}\n")

delete = input("Are you sure you want to delete these files - look at to_delete.txt first? (y/n)")
if delete == "y":
    print(f"Deleting {len(to_delete)} files")
    for path in to_delete:
        path.unlink()
else:
    print("Not deleting anything")
