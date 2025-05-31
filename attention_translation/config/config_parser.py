from dataclasses import dataclass
import yaml


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    # teacher_forcing_ratio: float


@dataclass
class ModelConfig:
    hidden_dim: int
    num_layers: int


@dataclass
class DataConfig:
    max_length: int
    train_path: str
    # val_path: str


@dataclass
class MainConfig:
    training: TrainingConfig
    model: ModelConfig
    data: DataConfig


def _dict_to_dataclass(data: dict) -> MainConfig:
    return MainConfig(
        training=TrainingConfig(**data['training']),
        model=ModelConfig(**data['model']),
        data=DataConfig(**data['data'])
    )


def load_config(path: str) -> MainConfig:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return _dict_to_dataclass(data)

