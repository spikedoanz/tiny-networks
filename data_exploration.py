import json
from pathlib import Path
import sys
import numpy as np

COLORS = {
    0: "\033[48;5;16m  \033[0m",  # Black
    1: "\033[48;5;21m  \033[0m",  # Blue
    2: "\033[48;5;196m  \033[0m", # Red
    3: "\033[48;5;46m  \033[0m",  # Green
    4: "\033[48;5;226m  \033[0m", # Yellow
    5: "\033[48;5;244m  \033[0m", # Gray
    6: "\033[48;5;201m  \033[0m", # Magenta
    7: "\033[48;5;208m  \033[0m", # Orange
    8: "\033[48;5;51m  \033[0m",  # Light blue
    9: "\033[48;5;88m  \033[0m",  # Brown
}


def display_grid(grid: list[list[int]], title: str = "", raw: bool = False) -> None:
    """Display a grid with colored blocks or as raw array."""
    print(f"\n{title}")

    if raw:
        arr = np.array(grid)
        print(arr)
    else:
        print("-" * (len(grid[0]) * 2))
        for row in grid:
            print("".join(COLORS.get(cell, "  ") for cell in row))



# --------------------------------------------------------------------------------

split = "training"
n_samples = -1 
raw = False

assert split in ["training", "evaluation"]

data_dir = Path("ARC-AGI/data") / split
if not data_dir.exists():
    print(f"Error: Directory {data_dir} not found")
    sys.exit(1)
json_files = sorted(data_dir.glob("*.json"))
print(json_files)

if not json_files:
    print(f"Error: No JSON files found in {data_dir}")
    sys.exit(1)
json_files = json_files[:n_samples]

for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)

    print(f"Task: {json_file.stem}")

    # Display training examples
    for i, example in enumerate(data.get("train", []), 1):
        print(f"\nTraining Example {i}:")
        display_grid(example["input"], "Input:", raw=raw)
        display_grid(example["output"], "Output:", raw=raw)

    # Display test examples
    for i, example in enumerate(data.get("test", []), 1):
        print(f"\nTest Example {i}:")
        display_grid(example["input"], "Input:", raw=raw)
        if "output" in example:
            display_grid(example["output"], "Output:", raw=raw)
    print("=" * 80)
