from pathlib import Path

import yaml


def validate_experiment_names(directory: Path) -> None:
    """
    Validate that all YAML files in the given directory have a field `experiment_name`
    that matches the file name (without extension).

    :param directory: The directory containing the YAML files to validate.
    :raises ValueError: If any YAML file doesn't meet the conditions.
    """
    for filename in directory.iterdir():
        if filename.suffix in [".yaml", ".yml"]:
            with open(filename, "r") as f:
                config = yaml.safe_load(f)
                expected_name = (
                    filename.stem
                )  # stem gives the filename without the extension
                if "experiment_name" not in config:
                    raise ValueError(
                        f"File '{filename.name}' is missing the 'experiment_name' field."
                    )
                elif config["experiment_name"] != expected_name:
                    raise ValueError(
                        f"In file '{filename.name}', the 'experiment_name' field should be '{expected_name}', but is '{config['experiment_name']}'"
                    )
