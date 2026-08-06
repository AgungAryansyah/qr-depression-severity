# Baseline paper reproduction

Use this track only to reproduce the published system. Follow the
[shared protocol](protocol.md) and keep its population and preprocessing fixed.

`baseline_paper.pdf` reports RoBERTa P-Tuning v2 plus
`all-mpnet-base-v2`, 128-dimensional average fusion, a one-layer 64-wide
BiLSTM with attention, MSE, dropout 0.5, learning rate 3e-4, 200 epochs, and
patience 20. The [published repository](https://github.com/clintonlau/dual-encoder-model)
YAML instead specifies batch size 1, whereas the paper specifies batch size 2.
Record the selected value and do not call a run a reproduction without
resolving this discrepancy.

The published preprocessing standardizes annotations, removes sync and routine
prompts, removes acronym underscores, lowercases text, and forms QR pairs. Its
public implementation has no handling for the three Ellie-less transcripts, so
it cannot establish a complete 189-participant reproduction by itself. A
reproduction must state its handling of those IDs and retain that policy across
train, development, and test.
