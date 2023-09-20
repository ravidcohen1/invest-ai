from dotenv import load_dotenv

load_dotenv()

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 150)

from pathlib import Path

from invest_ai.utils.configs import validate_experiment_names

validate_experiment_names(Path(__file__).parent / "configs" / "experiment")
