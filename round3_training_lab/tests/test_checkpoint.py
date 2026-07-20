import torch
from torch import nn

from training_lab.checkpoint import load_checkpoint, save_checkpoint


def _build():
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, optimizer


def test_save_then_load_restores_model_and_metadata(tmp_path) -> None:
    torch.manual_seed(0)
    model, optimizer = _build()

    # take one step so optimizer/model state is non-trivial
    x = torch.randn(5, 4)
    y = torch.randn(5, 3)
    optimizer.zero_grad()
    ((model(x) - y) ** 2).mean().backward()
    optimizer.step()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, step=42, extra={"val_loss": 1.5})

    fresh_model, fresh_optimizer = _build()
    # make sure fresh model differs before loading
    with torch.no_grad():
        for param in fresh_model.parameters():
            param.add_(1.0)

    meta = load_checkpoint(path, model=fresh_model, optimizer=fresh_optimizer)

    assert meta["step"] == 42
    assert meta["extra"]["val_loss"] == 1.5
    for saved, loaded in zip(model.parameters(), fresh_model.parameters(), strict=True):
        torch.testing.assert_close(saved, loaded)


def test_load_without_optimizer_still_restores_model(tmp_path) -> None:
    model, optimizer = _build()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, step=7)

    fresh_model, _ = _build()
    meta = load_checkpoint(path, model=fresh_model)

    assert meta["step"] == 7
    for saved, loaded in zip(model.parameters(), fresh_model.parameters(), strict=True):
        torch.testing.assert_close(saved, loaded)
