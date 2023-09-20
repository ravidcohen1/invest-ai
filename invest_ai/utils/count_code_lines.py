import os
import re
from pathlib import Path
from typing import List


def list_methods(file_path: Path) -> List[str]:
    # Implement your method listing logic here.
    # Placeholder for demonstration.
    return []


def count_python_lines(root: Path) -> int:
    total_lines = 0
    for r, _, files in os.walk(root):
        if any(
            skip in r
            for skip in [
                "/.",
                "__pycache__",
                "playground",
                "data",
                "outputs",
                "tests",
                "wandb",
                "notebooks",
                "logs",
            ]
        ):
            continue
        for f in files:
            if f == ".DS_Store" or not f.endswith(".py"):
                continue
            p = Path(r) / f

            with open(p, "r") as file:
                in_docstring = False
                for line in file:
                    stripped_line = line.strip()
                    if stripped_line.startswith("'''") or stripped_line.startswith(
                        '"""'
                    ):
                        in_docstring = not in_docstring  # Toggle state
                        continue
                    if (
                        not in_docstring
                        and stripped_line
                        and not stripped_line.startswith("#")
                    ):
                        total_lines += 1

    return total_lines


if __name__ == "__main__":
    repo_path = Path("/Users/user/PycharmProjects/invest-ai")
    lines = count_python_lines(repo_path)
    print(f"Total lines of Python code: {lines}")
