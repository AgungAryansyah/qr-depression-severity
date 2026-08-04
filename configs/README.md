# Configuration layout

Every runnable configuration is a small YAML delta composed through its
`extends` field. The loader resolves parent files before validation and saves
the complete result as `config.resolved.yaml` in the run directory.

```text
base.yaml
  └── tracking/global.yaml
experiments/modern/
  ├── deberta_dora_e5_transformer.yaml
  ├── deberta_dora_e5_transformer_compact.yaml
  └── deberta_dora_e5_transformer_wandb.yaml
experiments/ablations/
  └── reference.yaml and one-component ablation deltas
ablations/core.yaml
  └── study manifest that selects and orchestrates ablation candidates
tracking/
  └── reusable W&B and disabled tracking profiles
```

Use `experiments/modern/deberta_dora_e5_transformer.yaml` for the reference
modern model. Use `experiments/modern/deberta_dora_e5_transformer_compact.yaml`
when the narrower QR-fusion MLP is wanted. Use `ablations/core.yaml` only with
`scripts/run_ablation.py`; individual files in `experiments/ablations/` are
component deltas consumed by that study.
