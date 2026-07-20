import torch
from torch import nn

from training_lab.optim import AdamW


def _make_models():
    torch.manual_seed(0)
    model_a = nn.Linear(4, 4)
    model_b = nn.Linear(4, 4)
    model_b.load_state_dict(model_a.state_dict())
    return model_a, model_b


def test_adamw_matches_torch_reference_after_several_steps() -> None:
    model_a, model_b = _make_models()
    opt_a = AdamW(model_a.parameters(), lr=0.01, weight_decay=0.1)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=0.01, weight_decay=0.1)

    torch.manual_seed(1)
    x = torch.randn(8, 4)
    y = torch.randn(8, 4)

    for _ in range(25):
        opt_a.zero_grad()
        loss_a = ((model_a(x) - y) ** 2).mean()
        loss_a.backward()
        opt_a.step()

        opt_b.zero_grad()
        loss_b = ((model_b(x) - y) ** 2).mean()
        loss_b.backward()
        opt_b.step()

    for param_a, param_b in zip(model_a.parameters(), model_b.parameters(), strict=True):
        torch.testing.assert_close(param_a, param_b, rtol=1e-4, atol=1e-6)


def test_adamw_actually_updates_parameters() -> None:
    model = nn.Linear(4, 4)
    optimizer = AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
    before = [p.detach().clone() for p in model.parameters()]

    x = torch.randn(8, 4)
    y = torch.randn(8, 4)
    optimizer.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    optimizer.step()

    changed = any(
        not torch.equal(p.detach(), old)
        for p, old in zip(model.parameters(), before, strict=True)
    )
    assert changed
