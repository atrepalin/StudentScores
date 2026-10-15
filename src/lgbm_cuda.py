from typing import Literal

import lightgbm
import numpy as np


def get_lgbm_device(device: Literal["cuda", "cpu"]):
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
