# Controlled Experiment Protocol

Use the committed 107/35/47 manifest for every comparison. Keep every official
participant. DAIC-WOZ transcripts 451, 458, and 480 lack Ellie turns; this
project represents each as one response-only pair and records that condition in
the resolved configuration and results.

## Paper reproduction track

`baseline_paper.pdf` reports RoBERTa P-Tuning v2 plus
`all-mpnet-base-v2`, 128-dimensional average fusion, a one-layer 64-wide
BiLSTM with attention, MSE, dropout 0.5, learning rate 3e-4, 200 epochs, and
patience 20. The [published repository](https://github.com/clintonlau/dual-encoder-model)
YAML instead specifies batch size 1,
whereas the paper specifies batch size 2. Record the selected value and do not
call a run a reproduction without resolving this discrepancy.

The published preprocessing also standardizes annotations, removes sync and
routine prompts, removes acronym underscores, lowercases text, and forms QR
pairs. Its public implementation has no handling for the three Ellie-less
transcripts, so it cannot itself establish a complete 189-participant
reproduction. A reproduction must state its handling of those IDs and retain
the same handling across train, development, and test.

## Simplified-model track

Hold the split, preprocessing, missing-Ellie policy, optimizer schedule,
maximum epochs, early stopping, and metric fixed. Change one factor per
candidate: input (response-only or QR), encoder, QR projection, sequence
aggregator, or loss. Do not compare a candidate selected with development data
against a test result chosen by a different rule.

Run five fixed seeds on development. Select exactly one candidate using mean
development MAE, then evaluate it once on test. Report development mean and
standard deviation, test MAE and RMSE, train-development gap, trainable
parameter count, resolved configuration, and the input/preprocessing policy.

## Invalid comparisons

Do not compare results across changed split membership, dropped participants,
unrecorded transcript cleaning, differing missing-Ellie treatment, or
development-selected test reruns. A lower MAE under any of those changes is a
new study condition, not an improvement over the baseline.
