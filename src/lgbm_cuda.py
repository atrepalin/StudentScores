from functools import lru_cache
from typing import Literal

import lightgbm
import numpy as np


@lru_cache
def get_lgbm_device(device: Literal["cuda", "cpu"]):
    """
    Check if LightGBM is compiled with CUDA support

    Args:
        device (Literal["cuda", "cpu"]): "cuda" or "cpu"

    Returns:
        Literal["cuda", "cpu"]: supported device
    """
    if device == "cuda":
        try:
            data = np.random.rand(50, 2)
            label = np.random.randint(2, size=50)
            train_data = lightgbm.Dataset(data, label=label)

            params = {"num_iterations": 1, "device": "cuda", "verbosity": -1}

            gbm = lightgbm.train(params, train_set=train_data)
            del gbm

            return "cuda"
        except Exception as e:
            return "cpu"

    return "cpu"
