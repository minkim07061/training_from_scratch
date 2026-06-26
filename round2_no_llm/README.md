# Round 2: no-LLM transformer rebuild

This folder is a clean-room practice scaffold for re-implementing the project
without looking at the completed code.

Rules for the exercise:

1. Do not copy from `src/scratch_transformer/`.
2. Do not ask an LLM for implementation help.
3. Use these files only for interfaces, shape hints, invariants, and reminders.
4. Run the existing test suite from the repository root as your feedback loop.
5. Commit after each module passes its focused tests.

The files in this folder intentionally do **not** contain working solutions.
Each function body should be replaced by your own implementation.

Suggested order:

1. `tokenizer.py`
2. `rope.py`
3. `norm.py`
4. `mlp.py`
5. `attention.py`
6. `model.py`
7. `generate.py`
8. `train.py`

Recommended workflow:

```bash
# From repository root
python3 -m pytest tests/test_tokenizer_train.py
python3 -m pytest tests/test_tokenizer_encode.py
python3 -m pytest tests/test_tokenizer_decode.py
python3 -m pytest tests/test_tokenizer_save_load.py
python3 -m pytest tests/test_rope.py
python3 -m pytest tests/test_norm.py
python3 -m pytest tests/test_mlp.py
python3 -m pytest tests/test_kv_cache.py
python3 -m pytest tests/test_attention.py
python3 -m pytest tests/test_transformer_block.py
python3 -m pytest tests/test_transformer_lm.py
python3 -m pytest tests/test_generate.py
python3 -m pytest tests/test_train.py
```

When speed-running, keep a bug diary:

- What shape bug did you hit?
- Which tensor dimension did you confuse?
- Which invariant would have prevented the bug?
- Which test caught it?

