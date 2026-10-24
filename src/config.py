from dataclasses import dataclass, fields


class Dictable:
    """A dataclass that can be converted to a dictionary"""

    def keys(self):
        return [f.name for f in fields(self)]

    def __getitem__(self, key):
        return getattr(self, key)


@dataclass
class RidgeConfig(Dictable):
    alpha: float
    solver: str


@dataclass
class LassoConfig(Dictable):
    alpha: float
    max_iter: int
    selection: str


@dataclass
class ElasticNetConfig(Dictable):
    alpha: float
    l1_ratio: float
    max_iter: int
    selection: str


@dataclass
class DecisionTreeConfig(Dictable):
    criterion: str
    splitter: str
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int


@dataclass
class RandomForestConfig(Dictable):
    n_estimators: int
    max_depth: int
    max_features: float
    min_samples_leaf: int


@dataclass
class KNNConfig(Dictable):
    n_neighbors: int
    weights: str
    p: int
    algorithm: str


@dataclass
class XGBoostConfig(Dictable):
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_child_weight: float
    gamma: float
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float


@dataclass
class CatBoostConfig(Dictable):
    iterations: int
    learning_rate: float
    depth: int
    min_data_in_leaf: int
    bootstrap_type: str
    random_strength: float
    l2_leaf_reg: float
    leaf_estimation_iterations: int


@dataclass
class LightGBMConfig(Dictable):
    n_estimators: int
    learning_rate: float
    num_leaves: int
    max_depth: int
    min_child_samples: int
    max_bin: int
    subsample: float
    subsample_freq: int
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    min_split_gain: float


@dataclass
class TabMLPConfig(Dictable):
    hidden_dims: tuple[int, ...]
    dropout: float
    lr: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    scheduler_patience: int
    min_lr: float


@dataclass
class TabResNetConfig(Dictable):
    hidden_dim: int
    n_blocks: int
    dropout: float
    head_dims: tuple[int, ...]
    lr: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    scheduler_patience: int
    min_lr: float


@dataclass
class TabMConfig(Dictable):
    k: int
    n_blocks: int
    d_block: int
    dropout: float
    lr: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    scheduler_patience: int
    min_lr: float
