Model architecture for galaxy images based on code shared by Carolina Cuesta-Lazaro.

## Unified training entrypoint

Use the single runner:

```bash
python -m galaxy_images.galaxy_model.train --config galaxy_images/galaxy_model/configs/neighbors_default.json
```

Override values without editing YAML:

```bash
python -m galaxy_images.galaxy_model.train \
  --config galaxy_images/galaxy_model/configs/neighbors_default.json \
  --set trainer.devices=1 \
  --set trainer.num_steps=5000 \
  --set wandb.enabled=false
```

## Variants

- `neighbors_all_attn`: cross-attention in all UNet blocks
- `neighbors_mixed_attn`: mixed attention/non-attention UNet blocks

Set variant in config:

```yaml
run:
  variant: neighbors_mixed_attn
```

## Backward compatibility

`neighbours_train.py` is still available and now delegates to the unified runner using `configs/neighbors_default.json`.
