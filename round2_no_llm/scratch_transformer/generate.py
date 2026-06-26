"""Round 2 autoregressive generation scaffold.

Goal:
    Rebuild cached next-token generation.

Suggested test:
    python3 -m pytest tests/test_generate.py
"""

import torch

from scratch_transformer.model import TransformerLM


@torch.no_grad()
def generate(
    model: TransformerLM,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Generate token ids from a prompt.

    Implementation checkpoints:
        1. If max_new_tokens is zero, return a clone of the prompt.
        2. Save the model's current training/eval mode.
        3. Switch model to eval mode.
        4. Initialize per-layer caches.
        5. Feed the full prompt once with caches.
        6. For subsequent steps, feed only the most recent token.
        7. Extract final-step logits.
        8. Apply temperature.
        9. Optionally top-k filter logits.
        10. Convert logits to probabilities.
        11. Select/sample next token.
        12. Append next token to running sequence.
        13. Restore original training/eval mode before returning.

    Common bugs:
        - Forgetting to pass batch_size to init_caches.
        - Feeding the whole growing sequence every step despite having caches.
        - Not restoring train mode.
        - Treating top_k=0 as a request for top-0 filtering.
    """
    raise NotImplementedError

