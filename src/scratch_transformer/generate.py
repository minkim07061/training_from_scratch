"""Autoregressive text generation scaffold."""

import torch

from scratch_transformer.model import TransformerLM

def logits_to_probs(next_token_logits, temperature=1.0, top_k=0):
    next_token_logits = next_token_logits / temperature
    if top_k is not None and top_k > 0 and top_k < next_token_logits.shape[-1]:
        top_values, indices = torch.topk(next_token_logits, top_k, dim=-1)
        thresholds = top_values[:,-1].unsqueeze(-1)
        mask = next_token_logits < thresholds
        next_token_logits = next_token_logits.masked_fill(mask, float("-inf"))
    return torch.softmax(next_token_logits, dim=-1)

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
    batch = input_ids.shape[0]
    if max_new_tokens == 0:
        return input_ids.clone()
    was_training = model.training
    model.eval()
    caches = model.init_caches(batch)
    running_sequence = input_ids.clone()
    for i in range(max_new_tokens):
        if i == 0:
            logits, caches = model(running_sequence, caches=caches)
        if i != 0:
            logits, caches = model(running_sequence[:,-1].unsqueeze(-1), caches=caches)
        next_token_logits = logits[:, -1, :]
        probs = logits_to_probs(next_token_logits)
        # choose the next token given the probs
        next_tokens = torch.argmax(probs, dim=-1).unsqueeze(-1)
        running_sequence = torch.cat([running_sequence, next_tokens], dim=-1)
    model.train(was_training)
    return running_sequence