import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Update experiment queue")
    parser.add_argument(
        "command",
        help="Command to perform on the queue"
    )
    parser.add_argument(
        "-n", "--num-repeats",
        type=int,
        help="Number of times to repeat the command (default: 1)"
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        required=True,
        help="Path to file containing queue to update"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    new_file = Path(args.file).stem
    old_file = args.file
    with open(old_file, "r") as f_old:
        old_lines = f_old.readlines()

    with open(new_file, "w") as f_new:
        f_new.writelines(old_lines)

    with open(new_file, "a") as f_new:
        for _ in range(args.num_repeats):
            f_new.write(args.command + "\n")
