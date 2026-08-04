# Configuration layout

Every runnable configuration is a small YAML delta composed through its
`extends` field. The loader resolves parent files before validation and saves
the complete result as `config.resolved.yaml` in the run directory.

```text
base.yaml
  └── tracking/global.yaml
experiments/modern/
  ├── deberta_dora_e5_transformer.yaml
  ├── deberta_dora_e5_transformer_small.yaml
  └── deberta_dora_e5_transformer_compact.yaml
experiments/ablations/
  └── modern and small reference files with one-component ablation deltas
ablations/
  ├── core.yaml
  └── core_small.yaml
tracking/
  └── reusable W&B and disabled tracking profiles
```

Use `experiments/modern/deberta_dora_e5_transformer.yaml` for the reference
modern model; it inherits online W&B tracking from `base.yaml`. Use
`experiments/modern/deberta_dora_e5_transformer_compact.yaml` when the narrower
QR-fusion MLP is wanted. Disable W&B with
`--override tracking.backend=disabled --override tracking.mode=disabled`.
Use `experiments/modern/deberta_dora_e5_transformer_small.yaml` for the smaller
dual-branch profile: DeBERTa-v3-small, E5-small-v2, 128-dimensional QR vectors,
and a narrower interview Transformer. `ablations/core.yaml` and
`ablations/core_small.yaml` are study manifests for `scripts/run_ablation.py`;
their component deltas are consumed only by those studies.
