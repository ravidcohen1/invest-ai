import ast
import os
from pathlib import Path


def list_methods(file: str):
    output = []
    with open(file, "r") as f:
        for node in ast.walk(ast.parse(f.read())):
            if isinstance(node, ast.FunctionDef):
                output.append(f"    |- {node.name}")
    return output


def list_files(root: Path):
    output = []
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
        output.append(f"Dir: {r}")
        for f in files:
            if f == ".DS_Store":
                continue
            p = Path(r) / f
            output.append(f"  |- {f}")
            if f.endswith(".py"):
                output.extend(list_methods(p))

    full_output = "\n".join(output)
    print(full_output)
    print(f"Total characters in output: {len(full_output)}")


if __name__ == "__main__":
    root = Path("/Users/user/PycharmProjects/invest-ai")
    list_files(root)
