from typing import List, Dict

import numpy as np
import pandas as pd


def create_evaluation_table(data: Dict, columns: List = None):
    res_df = pd.DataFrame(pd.Series(data))
    if columns:
        res_df.columns = columns
    return res_df.to_string()
