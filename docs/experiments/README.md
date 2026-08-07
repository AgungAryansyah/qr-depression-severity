# Experiment documentation

Choose the experiment type that matches the research question. All types use
the [shared protocol](protocol.md) unless their guide states an additional
constraint.

| Type | Configuration | Purpose | Guide |
| --- | --- | --- | --- |
| Baseline reproduction | No active configuration | Reproduce the published RoBERTa/MPNet study after resolving its documented discrepancies. | [Baseline reproduction](baseline-reproduction.md) |
| Modern model | `configs/experiments/modern/` | Train the DeBERTa + E5 dual-branch reference or its capacity variants. | [Modern models](modern-models.md) |
| Modern ablation | `configs/ablations/core.yaml`, `configs/ablations/core_small.yaml` | Compare modern-model components under one controlled selection protocol. | [Modern ablations](modern-ablations.md) |
| Simple control | `configs/ablations/simple_input.yaml`, `configs/ablations/simple_encoder.yaml` | Compare simple-model input and encoder controls. | [Simple controls](simple-controls.md) |

The files in `configs/experiments/ablations/` are candidate definitions. Run
their ablation manifest instead of an individual candidate so the selection
protocol remains intact.
