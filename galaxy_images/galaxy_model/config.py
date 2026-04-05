from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DataConfig:
    mode: str = "precomputed"  # precomputed | neighbors
    precomputed_h5: str = "/data/vision/billf/scratch/pablomer/data/neighbor_batches/neighbours_vds.h5"
    neighbors_h5: str = "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5"
    max_neighbors: int = 5
    val_ratio: float = 0.05
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True
    drop_last_train: bool = True


@dataclass
class ModelConfig:
    in_channels: int = 4
    cond_channels: int = 4
    image_size: int = 48
    model_channels: int = 128
    channel_mult: List[int] = field(default_factory=lambda: [1, 2, 4, 4])
    layers_per_block: int = 2
    attention_head_dim: int = 8
    cross_attention_dim: int = 16
    pretrained_encoder: bool = False
    concat_conditioning: bool = False
    lr: float = 1e-4
    num_sample_images: int = 10
    num_mse_images: int = 32
    num_integration_steps: int = 250
    lambda_generative: float = 1.0
    lambda_geometric: float = 0.0
    num_umap_batches: int = 8
    mask_center: bool = False
    all_attention: bool = True
    figures_dir: Optional[str] = None


@dataclass
class TrainerConfig:
    seed: int = 42
    num_steps: int = 1_500_000
    devices: int = 4
    accelerator: str = "auto"
    strategy: str = "ddp_find_unused_parameters_true"
    precision: str = "16-mixed"
    h100_precision: str = "bf16-mixed"
    h100_batch_size: int = 64
    auto_adjust_for_h100: bool = True
    scale_steps_by_devices: bool = True
    val_check_interval: int = 1000
    log_every_n_steps: int = 10
    num_sanity_val_steps: int = 0
    monitor_metric: str = "val/loss"
    checkpoint_every_n_train_steps: int = 1000


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "galaxy-flow-matching-neighbours"
    name: str = "neighbours-48x48-zdim16-geom0.0-amd-5nbs"
    log_model: bool = False


@dataclass
class RunConfig:
    variant: str = "neighbors_all_attn"
    output_dir: str = "galaxy_images/galaxy_model/outputs"
    resume_from: Optional[str] = None  # path to checkpoint to resume from; None = fresh run


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        return cls(
            data=DataConfig(**data.get("data", {})),
            model=ModelConfig(**data.get("model", {})),
            trainer=TrainerConfig(**data.get("trainer", {})),
            wandb=WandbConfig(**data.get("wandb", {})),
            run=RunConfig(**data.get("run", {})),
        )

    def validate(self) -> None:
        if self.data.mode not in {"precomputed", "neighbors"}:
            raise ValueError(f"Unsupported data.mode={self.data.mode!r}. Use 'precomputed' or 'neighbors'.")
        if self.data.mode == "precomputed" and not self.data.precomputed_h5:
            raise ValueError("data.precomputed_h5 must be set for data.mode='precomputed'.")
        if self.data.mode == "neighbors" and not self.data.neighbors_h5:
            raise ValueError("data.neighbors_h5 must be set for data.mode='neighbors'.")


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def load_experiment_config(config_path: str | Path, overrides: Optional[List[str]] = None) -> ExperimentConfig:
    config_path = Path(config_path)
    user_config = _load_config_file(config_path)

    merged = _deep_update(ExperimentConfig().to_dict(), user_config)

    for raw_override in overrides or []:
        if "=" not in raw_override:
            raise ValueError(f"Invalid override {raw_override!r}. Expected key=value.")
        key, raw_value = raw_override.split("=", 1)
        _set_nested(merged, key, _parse_value(raw_value))

    config = ExperimentConfig.from_dict(merged)
    config.validate()
    return config


def _load_config_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        raw = f.read()

    if suffix == ".json":
        return json.loads(raw)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "YAML config requested but PyYAML is not installed. "
                "Use a .json config file or install pyyaml."
            ) from e
        return yaml.safe_load(raw) or {}

    raise ValueError(f"Unsupported config format {suffix!r}. Use .json, .yaml, or .yml.")


def _parse_value(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return raw_value
