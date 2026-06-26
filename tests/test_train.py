import torch
from torch import nn

import scratch_transformer.train as train_module
from scratch_transformer.config import TransformerConfig
from scratch_transformer.train import next_token_loss, train_epoch, train_step


class ToyTrainModel(nn.Module):
    """Small model with TransformerLM-compatible return shape."""

    def __init__(self, vocab_size: int = 5) -> None:
        super().__init__()
        self.config = TransformerConfig(vocab_size=vocab_size, d_model=8, n_heads=2, d_ff=16)
        self.embedding = nn.Embedding(vocab_size, 4)
        self.proj = nn.Linear(4, vocab_size)
        self.forward_training_flags: list[bool] = []

    def forward(self, input_ids: torch.Tensor):
        self.forward_training_flags.append(self.training)
        return self.proj(self.embedding(input_ids)), None


def test_next_token_loss_matches_cross_entropy_reference() -> None:
    logits = torch.tensor(
        [
            [[2.0, 0.0, -1.0], [0.0, 3.0, -1.0]],
            [[-1.0, 0.0, 2.0], [4.0, 0.0, -1.0]],
        ]
    )
    targets = torch.tensor([[0, 1], [2, 0]])

    actual = next_token_loss(logits, targets)

    expected = nn.functional.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1))
    torch.testing.assert_close(actual, expected)


def test_train_step_sets_train_mode_updates_parameters_and_returns_detached_loss() -> None:
    torch.manual_seed(0)
    model = ToyTrainModel(vocab_size=5)
    model.eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs = torch.tensor([[0, 1, 2], [2, 3, 4]])
    targets = torch.tensor([[1, 2, 3], [3, 4, 0]])
    before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = train_step(model, optimizer, inputs, targets)

    assert loss.ndim == 0
    assert not loss.requires_grad
    assert model.training is True
    assert model.forward_training_flags == [True]
    assert any(
        not torch.equal(parameter.detach(), old)
        for parameter, old in zip(model.parameters(), before, strict=True)
    )


def test_train_step_zeros_gradients_after_optimizer_step() -> None:
    torch.manual_seed(0)
    model = ToyTrainModel(vocab_size=5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs = torch.tensor([[0, 1, 2]])
    targets = torch.tensor([[1, 2, 3]])

    train_step(model, optimizer, inputs, targets)

    for parameter in model.parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0


def test_train_step_accepts_gradient_clipping() -> None:
    torch.manual_seed(0)
    model = ToyTrainModel(vocab_size=5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs = torch.tensor([[0, 1, 2]])
    targets = torch.tensor([[1, 2, 3]])

    loss = train_step(model, optimizer, inputs, targets, grad_clip=0.5)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_train_epoch_returns_mean_loss_and_calls_train_step_for_each_batch(monkeypatch) -> None:
    model = ToyTrainModel(vocab_size=5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = [
        (torch.tensor([[0, 1]]), torch.tensor([[1, 2]])),
        (torch.tensor([[2, 3]]), torch.tensor([[3, 4]])),
        (torch.tensor([[4, 0]]), torch.tensor([[0, 1]])),
    ]
    calls = []

    def fake_train_step(model_arg, optimizer_arg, input_ids, target_ids, *, grad_clip=None):
        calls.append((model_arg, optimizer_arg, input_ids, target_ids, grad_clip))
        return torch.tensor(float(len(calls)))

    monkeypatch.setattr(train_module, "train_step", fake_train_step)

    mean_loss = train_epoch(model, optimizer, batches, grad_clip=1.25)

    assert mean_loss == 2.0
    assert len(calls) == 3
    assert all(call[0] is model for call in calls)
    assert all(call[1] is optimizer for call in calls)
    assert all(call[4] == 1.25 for call in calls)
