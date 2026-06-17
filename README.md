# training_from_scratch

A learning-oriented PyTorch scaffold for hand-implementing a decoder-only
transformer from scratch.

The core logic is intentionally left as TODOs. The repository gives you:

- a clean module layout,
- concrete function/class signatures,
- comments describing the next implementation steps,
- sanity tests that pass before implementation and become behavioral checks as
  each TODO is completed.

## Repository layout

```text
src/scratch_transformer/
  tokenizer.py    # BPE tokenizer train/encode/decode/save/load TODOs
  rope.py         # RoPE cache construction and application TODOs
  norm.py         # RMSNorm TODO
  attention.py    # causal mask, attention, and KV cache TODOs
  mlp.py          # SwiGLU TODO
  model.py        # Transformer block and LM forward TODOs
  generate.py     # autoregressive generation TODO
  train.py        # training step and epoch loop TODOs
tests/
  test_project_structure.py  # always-on scaffold sanity tests
  test_todo_contracts.py     # xfail implementation-contract tests
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run tests

```bash
pytest
```

The TODO contract tests are marked with `pytest.mark.xfail(raises=NotImplementedError)`.
That means:

- before you implement a component, its test is reported as expected-to-fail;
- after you implement it correctly, the same test can pass without changing the
  test marker;
- if your implementation returns the wrong result, the test fails normally.

## Suggested implementation order

1. `tokenizer.py`: train a byte-level BPE tokenizer, then verify round trips.
2. `rope.py`: build cos/sin caches and rotate query/key pairs.
3. `norm.py`: implement RMSNorm and compare to the formula in the test.
4. `attention.py`: implement the causal mask, KV cache append, and attention.
5. `mlp.py`: implement SwiGLU.
6. `model.py`: wire blocks into a full decoder-only language model.
7. `generate.py`: add autoregressive generation with KV cache support.
8. `train.py`: implement next-token training steps and epoch logging.

Run `pytest` after each component. The `XPASS` entries tell you which learning
TODOs now satisfy their sanity checks.
