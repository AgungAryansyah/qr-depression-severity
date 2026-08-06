# Shared experiment protocol

Use the committed 107/35/47 manifest for every comparison. Keep every official
participant. DAIC-WOZ transcripts 451, 458, and 480 lack Ellie turns; this
project represents each as one response-only pair and records that condition in
the resolved configuration and results.

## Valid comparisons

Hold the split, preprocessing, missing-Ellie policy, optimizer schedule,
maximum epochs, early stopping, and metric fixed unless the experiment
explicitly studies one of them. Do not compare a candidate selected with
development data against a test result chosen by a different rule.

Do not compare results across changed split membership, dropped participants,
unrecorded transcript cleaning, differing missing-Ellie treatment, or
development-selected test reruns. A lower MAE under any of those changes is a
new study condition, not an improvement over the baseline.
