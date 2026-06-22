# Checkpoints

`model.pt` is intentionally not committed. A randomly initialized classifier would
produce misleading forensic scores and is not a valid detector.

Create a checkpoint with `python -m ml.train --manifest data/manifest.csv`, evaluate
it on a held-out dataset, then copy the reviewed artifact here as `model.pt`. The API
will return `503 model_not_ready` until this validated artifact exists.

