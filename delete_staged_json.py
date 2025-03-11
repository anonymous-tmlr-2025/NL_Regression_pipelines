import json
import os
from pathlib import Path

def get_staged_files():
    """Get list of files staged in git"""
    # Run git command to get staged files
    git_command = "git diff --cached --name-only"
    staged_files = os.popen(git_command).read().splitlines()
    return staged_files

if __name__ == "__main__":
    staged_files = get_staged_files()
    to_delete = []
    for file in staged_files:
        if file.endswith(".json"):
            to_delete.append(file)
    Path("to_delete.txt").write_text("\n".join(to_delete))
    delete = input(f"Delete {len(to_delete)} staged json files? (y/n): ")
    if delete == "y":
        for file in to_delete:
            os.remove(file)
            print(f"Deleted {file}")
    else:
        print("Aborted")