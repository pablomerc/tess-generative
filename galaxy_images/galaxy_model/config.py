from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DataConfig:
    mode: str = "precomputed"  # precomputed | neighbors | efficient | ram48
    precomputed_h5: str = "/data/vision/billf/scratch/pablomer/data/neighbor_batches/neighbours_vds.h5"
    neighbors_h5: str = "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5"
    efficient_data_dir: Optional[str] = None
    max_neighbors: int = 5
    val_ratio: float = 0.05
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True
    drop_last_train: bool = True
    save_heldout_validation: bool = False
    heldout_validation_dir: Optional[str] = None
    heldout_num_batches: int = 4
    heldout_file_name: Optional[str] = None
    # If set, anchors whose object_id_legacy appears in this txt file are removed
    # from the train+val splits (kept out of training entirely so they remain
    # available as a downstream-evaluation holdout).
    downstream_holdout_ids_txt: Optional[str] = None
    # If True, the same-instrument neighbors are sampled uniformly at random
    # instead of using the precomputed nearest-neighbor list.
    random_neighbors: bool = False
    # Data-scale ablation: path to a JSON file listing the RAW anchor positions
    # (into the full dataset, before any holdout/lens exclusion) to train on.
    # All listed positions must fall inside the seeded train split.
    train_subset_json: Optional[str] = None
    # Cyclically tile the subset's index list up to this many items so that every
    # arm of a data-scale sweep has the SAME epoch length. This matters because the
    # LR scheduler steps per epoch, so epoch length sets the LR waveform period --
    # without tiling, a smaller subset would get different optimization dynamics
    # and the data-scale comparison would be confounded. Defaults to the full
    # train split size when train_subset_json is set.
    train_subset_tile_to: Optional[int] = None


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
    experiment_config: Any = "bn_36x16"
    instrument_zdim: Optional[int] = None
    instrument_pooling: str = "masked_mean"
    lr: float = 1e-4
    # "paper" (default) reproduces published runs exactly, including the epoch-boundary LR
    # alternation between lr and 0.0 documented in CLAUDE.md. Prefer "cosine" (anneal to 0
    # over max_steps), "linear", or "constant" for new runs.
    lr_schedule: str = "paper"
    num_sample_images: int = 10
    num_mse_images: int = 32
    num_integration_steps: int = 250
    lambda_generative: float = 1.0
    lambda_geometric: float = 0.0
    num_umap_batches: int = 8
    mask_center: bool = False
    all_attention: bool = True
    figures_dir: Optional[str] = None
    disable_global_physics: bool = False
    encoder_stride_overrides: Optional[Dict[str, int]] = None
    encoder_1_stride_overrides: Optional[Dict[str, int]] = None
    encoder_2_stride_overrides: Optional[Dict[str, int]] = None
    instrument_flatten_to_one_token: bool = False
    encoder_2_global_conv: bool = False
    instrument_as_class_conditioning: bool = False
    # Diffusion-ablation fields (ignored by FM modules via filter_supported_model_kwargs)
    prediction_type: str = "epsilon"
    num_train_timesteps: int = 1000
    beta_schedule: str = "squaredcos_cap_v2"


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
    # Grouping metadata for sweeps/ablations. `group` puts sibling runs in one
    # W&B group so they can be compared as a family; `tags` are filterable;
    # `job_type` distinguishes e.g. train vs eval runs within a group.
    group: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    job_type: Optional[str] = None
    # Extra scalars merged into the W&B run config. Use this for the independent
    # variable of a sweep (e.g. {"scale/n_anchors": 1000}) so W&B can plot metrics
    # against it -- a config path string is not plottable.
    extra_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    variant: str = "neighbors_all_attn"
    output_dir: str = "galaxy_images/galaxy_model/outputs"
    resume_from: Optional[str] = None  # path to checkpoint to resume from; None = fresh run
    shared_checkpoint_dir: Optional[str] = (
        "galaxy_images/galaxy_model/checkpoints"
    )  # if set, best checkpoint also goes to <shared>/<wandb.name or variant>/


@dataclass
class LensValConfig:
    enabled: bool = False
    lens_h5: Optional[str] = (
        "galaxy_images/galaxy_model/lense_reconstruction/lens_reconstruction_dataset.h5"
    )
    # 0-based indices into lens_h5 to exclude from training and use as a fixed val set.
    lens_indices_zero_based: List[int] = field(
        default_factory=lambda: [4, 7, 11, 17, 19, 28, 31, 32]
    )
    exclude_from_train: bool = True
    every_n_validations: int = 5
    num_integration_steps: int = 100
    num_samples_per_cond: int = 5


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    run: RunConfig = field(default_factory=RunConfig)
    lens_val: LensValConfig = field(default_factory=LensValConfig)

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
            lens_val=LensValConfig(**data.get("lens_val", {})),
        )

    def validate(self) -> None:
        if self.data.mode not in {"precomputed", "neighbors", "efficient", "ram48"}:
            raise ValueError(
                f"Unsupported data.mode={self.data.mode!r}. "
                "Use 'precomputed', 'neighbors', 'efficient', or 'ram48'."
            )
        if self.data.mode == "precomputed" and not self.data.precomputed_h5:
            raise ValueError("data.precomputed_h5 must be set for data.mode='precomputed'.")
        if self.data.mode == "neighbors" and not self.data.neighbors_h5:
            raise ValueError("data.neighbors_h5 must be set for data.mode='neighbors'.")
        if self.data.mode == "efficient" and not self.data.efficient_data_dir:
            raise ValueError("data.efficient_data_dir must be set for data.mode='efficient'.")
        if self.data.mode == "ram48" and not self.data.efficient_data_dir:
            raise ValueError("data.efficient_data_dir must be set for data.mode='ram48'.")
        if self.data.heldout_num_batches < 1:
            raise ValueError("data.heldout_num_batches must be >= 1.")


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
