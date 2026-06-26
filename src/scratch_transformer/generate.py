"""Autoregressive text generation scaffold."""

import torch

from scratch_transformer.model import TransformerLM

def logits_to_probs(logits, temperature=1.0, top_k=0):
    logits = logits / temperature
    if top_k >=0 and top_k < logits.shape[-1]:
        top_values, _ = torch.top_k(logits, top_k, dim=-1)
        logits = logits.masked_fill()


@torch.no_grad()
def generate(
    model: TransformerLM,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Generate continuations from a prompt.

    Args:
        model: Transformer language model.
        input_ids: Prompt token ids with shape ``(batch, prompt_len)``.
        max_new_tokens: Number of new tokens to sample.
        temperature: Softmax temperature. Use 1.0 as the neutral value.
        top_k: Optional top-k sampling cutoff.

    TODO:
    - Switch model to eval mode while preserving the original training state.
    - Initialize per-layer KV caches.
    - Feed the prompt, then one generated token at a time.
    - Convert final-step logits to probabilities using temperature/top-k.
    - Sample or choose next tokens and append them to the running sequence.
    """
    model.eval()
    model.init_caches()
    for i in range(max_new_tokens):
        logits, caches = model(input_ids)
        probs = logits_to_probs(logits)
        # choose the next token given the probs
