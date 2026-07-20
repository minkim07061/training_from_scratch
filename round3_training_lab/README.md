# Round 3: training lab (hand-implementation study module)

After you finish the transformer itself, this module has you hand-implement the
pieces that turn a model into a real training/experimentation setup.

These are the components that matter for running small pretraining experiments
and, later, scaling-law studies.

## Rules

1. Implement the function/method bodies yourself.
2. Do not copy from an existing framework's source while implementing.
3. Each file gives you interfaces, shapes, invariants, and checkpoints only.
4. Use the tests as your feedback loop.
5. Commit after each module passes.

## Modules (suggested order)

1. `data.py`       - train/val split and next-token batch sampling
2. `schedule.py`   - cosine learning-rate schedule with warmup
3. `optim.py`      - AdamW optimizer from scratch
4. `sampling.py`   - temperature, top-k, and top-p (nucleus) filtering
5. `metrics.py`    - perplexity, running averages, throughput
6. `checkpoint.py` - save/load model + optimizer + step

Each module is intentionally standalone: it depends only on PyTorch, not on your
transformer implementation. That keeps this study module isolated and fast.

## Running the tests

From the repository root:

```bash
python3 -m pytest round3_training_lab/tests
```

Run a single module's tests:

```bash
python3 -m pytest round3_training_lab/tests/test_optim.py
```

The `conftest.py` in this folder puts `training_lab` on the import path, so you
do not need to install anything extra.

## What to do after this module

Once these pass, you have the tools to:

- prepare a small dataset (e.g. Tiny Shakespeare),
- train your transformer with a real LR schedule and AdamW,
- checkpoint and resume,
- sample text with temperature/top-k/top-p,
- log loss, perplexity, and tokens/sec,
- and run small model-size / data-size / compute sweeps for scaling curves.

Keep a bug diary as you go: which shape or dtype mistake did you make, and which
test caught it?
