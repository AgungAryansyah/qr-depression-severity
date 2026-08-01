# Modern QR Depression-Severity Pipeline

This document describes the active, modern question-response (QR) research
pipeline. It predicts a PHQ-8 total score and an auxiliary ordered severity
level from DAIC-WOZ interview transcripts. It is research software only and
must not be used for clinical diagnosis or treatment decisions.

The implementation intentionally keeps configuration, data handling, model
code, training, evaluation, and tracking separate. The retired paper
reproduction architecture is not part of this pipeline.

## System flow

```text
DAIC-WOZ transcript + official split membership
        |
        v
Transcript cleaning and QR extraction
  Ellie turn(s) -> participant response
        |
        v
Separate question / response tokenization
        |
        +------------------------------+-----------------------------+
        |                              |                             |
        v                              v                             v
DeBERTa-v3-base + DoRA          E5-base-v2 (frozen)           QR validity mask
question and response           question and response          [batch, QR pairs]
        |                              |
        v                              v
Feature-interaction QR fusion   Feature-interaction QR fusion
        |                              |
        +--------------+---------------+
                       v
             Vector-gated branch fusion
                       |
                       v
       Position-aware turn-level Transformer
                       |
                       v
          Learned attention interview pooling
                       |
              +--------+---------+
              v                  v
        PHQ-8 regression      CORN ordinal head
```

The default experiment is defined in
`configs/experiments/modern/deberta_dora_e5_transformer.yaml`. All
experiment behaviour is loaded and validated from YAML before a run starts;
the resolved configuration is saved with the run artifacts.

## Data and QR representation

`data.splits` validates the configured DAIC-WOZ partition membership before
loading interviews. It does not create generated replacement splits.

`data.qr_pairing` preserves the conversation direction: a QR pair contains
one or more preceding Ellie interviewer turns as its question and the next
participant turn as its response. Participant turns are never used as a
question. Each pair retains its participant ID, QR index, question type, and
available timestamps.

The collator keeps question and response inputs separate for both encoders.
For each branch its tensors have this layout:

```text
[batch, maximum QR pairs in this batch, token length]
```

`qr_mask` marks real QR pairs. Token attention masks mark real tokens within
each question or response. Empty cleaned turns and interviews without valid
pairs are skipped with explicit runtime warnings so a local data problem is
visible rather than silently changing the population. Processed QR pairs can
be cached locally; the cache is keyed by transcript metadata and preprocessing
settings, and never belongs in version control.

The current configuration supports at most `data.max_qr_pairs` pairs per
interview (128 by default). Exceeding that limit raises an error instead of
silently truncating the interview.

## QR encoder branches

Each branch independently encodes the question and the response using
attention-mask-aware mean pooling:

```text
q = pool(question token states)
r = pool(response token states)
```

The pooled embeddings are transformed using the default feature-interaction
QR representation:

```text
x = [q; r; q * r; |q - r|]
e_qr = Linear(4d, 2d) -> GELU -> LayerNorm -> Dropout -> Linear(2d, 256)
```

The two branches are deliberately complementary:

| Branch | Default backbone | Adaptation and state | Input formatting |
| --- | --- | --- | --- |
| Adapted QR branch | `microsoft/deberta-v3-base` | DoRA on attention projections; base encoder frozen | Raw question and response fields |
| Semantic QR branch | `intfloat/e5-base-v2` | Fully frozen and held in evaluation mode | `query: <question>` and `passage: <response>` |

DoRA targets DeBERTa's query, key, value, and attention-output projection
modules. The target module names are discovered and validated when the model
is built. DeBERTa can also be configured as frozen or with LoRA for controlled
experiments. E5 output is L2-normalized before QR projection in the default
configuration.

## Branch fusion

Both QR branches produce 256-dimensional vectors for every valid QR pair.
The default vector gate is calculated per feature:

```text
g = sigmoid(MLP([e_adapted; e_semantic]))
e_fused = g * e_adapted + (1 - g) * e_semantic
```

Layer normalization and dropout follow fusion. During training, configurable
branch dropout can replace either branch with the other branch's embedding;
it never removes both branches. The model returns gate values so evaluation
can report their mean and variance. Average, concatenation-plus-projection,
and scalar-gate alternatives are available as ablations.

## Interview encoder and heads

The fused QR sequence is encoded by a pre-layer-normalized Transformer. The
default is two layers with hidden size 256, four attention heads, a
512-dimensional feed-forward layer, GELU activation, and dropout 0.2.
Learned absolute QR-position embeddings are added before self-attention.

`qr_mask` is passed as the Transformer padding mask, so padded QR positions
cannot influence attention or pooling. An interview with no real QR pairs is
rejected.

The default learned attention pooler calculates an interview vector and a QR
attention distribution. The Transformer also supports CLS-token and masked
mean pooling. The pooled interview embedding feeds two heads:

- Regression head: LayerNorm, dropout, `256 -> 128`, GELU, dropout, then one
  scalar PHQ-8 prediction.
- Ordinal head: a CORN classifier with four ordered boundaries for the five
  PHQ-8 severity bands: 0--4, 5--9, 10--14, 15--19, and 20--24.

## Training and runtime execution

The default objective combines Huber regression loss (`delta=2.0`) with CORN
ordinal loss:

```text
total_loss = regression_loss + 0.5 * ordinal_loss
```

The trainer uses AdamW, a configurable scheduler, deterministic seed setup,
gradient clipping, automatic mixed precision when supported, gradient
accumulation, early stopping on development RMSE, and best-checkpoint restore.
Regression metrics include RMSE, MAE, MSE, signed mean error, and maximum
absolute error. Ordinal evaluation includes severity accuracy, macro-F1,
quadratic weighted kappa, and mean absolute severity error.

An interview can contain many QR pairs, so QR encodings are flattened only
over valid pairs and processed in configurable micro-batches. This reduces
peak activation memory without changing the model's numerical computation.
The default runtime placement uses the adapted encoder and interview model on
`cuda:0`, and frozen E5 on `cuda:1`; the semantic QR vectors are moved to the
adapted device immediately before fusion. DeBERTa gradient checkpointing is
enabled by default to trade additional compute for lower activation memory.

## Evaluation, provenance, and tracking

Training evaluates development data at the end of each epoch. The test split
is an explicit, separate evaluation command and must not be used to select
architecture, hyperparameters, checkpoints, or seeds.

Every run directory is the local source of truth. It contains the resolved
configuration, split IDs, environment and metadata records, training history,
best checkpoint, metrics, and prediction files. Raw transcript text and
tokenized text are not saved as run artifacts.

Tracking is an adapter, not a model or data dependency. The default local
tracker writes the artifacts and prints epoch-level progress. The optional
Weights & Biases adapter is configured in YAML and loads `WANDB_API_KEY` from
the process environment or an ignored `.env` file. It creates one run per
seed, logs epoch-level metrics and artifacts when enabled, and resumes that
same run for evaluation metrics and prediction artifacts.

## Package boundaries

```text
scripts/                     CLI argument parsing only
orchestration/               builds a configured experiment
configuration/               YAML composition, validation, serialization
data/                        splits, transcripts, QR pairs, collation, cache
models/                      pure tensor modules and model construction
training/                    losses, optimizer, scheduler, loop, checkpoints
evaluation/                  saved-checkpoint predictions and metrics
tracking/                    local and W&B adapters
```

Lower-level packages do not parse command-line arguments or YAML. Data and
model code do not import W&B, and the trainer emits metric dictionaries rather
than owning an external tracking implementation.

## Relevant entry points

```bash
uv run python scripts/train.py \
  --config configs/experiments/modern/deberta_dora_e5_transformer.yaml

uv run python scripts/evaluate.py \
  --config configs/experiments/modern/deberta_dora_e5_transformer.yaml \
  --checkpoint <path-to-best-checkpoint> --split dev
```

Use the W&B configuration variant only after placing the API key in the local
ignored `.env` file or the process environment. See `README.md` for complete
environment setup and run commands.
