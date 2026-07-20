import torch

from training_lab.data import TokenDataset, train_val_split


def test_train_val_split_holds_out_tail() -> None:
    tokens = list(range(100))

    train, val = train_val_split(tokens, val_fraction=0.1)

    assert isinstance(train, torch.Tensor)
    assert isinstance(val, torch.Tensor)
    assert train.dtype == torch.long
    assert val.dtype == torch.long
    assert len(train) == 90
    assert len(val) == 10
    assert train[0].item() == 0
    assert train[-1].item() == 89
    assert val[0].item() == 90
    assert val[-1].item() == 99


def test_token_dataset_length_counts_valid_windows() -> None:
    dataset = TokenDataset(list(range(100)), context_length=8)

    assert len(dataset) == 92


def test_get_batch_returns_shifted_next_token_targets() -> None:
    dataset = TokenDataset(list(range(100)), context_length=8)
    generator = torch.Generator().manual_seed(0)

    inputs, targets = dataset.get_batch(4, generator=generator)

    assert inputs.shape == (4, 8)
    assert targets.shape == (4, 8)
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long
    # next-token invariant
    torch.testing.assert_close(targets[:, :-1], inputs[:, 1:])
    # with a range() stream, each target is exactly input + 1
    torch.testing.assert_close(targets, inputs + 1)


def test_get_batch_is_reproducible_with_generator_seed() -> None:
    dataset = TokenDataset(list(range(200)), context_length=16)

    gen_a = torch.Generator().manual_seed(123)
    gen_b = torch.Generator().manual_seed(123)

    inputs_a, targets_a = dataset.get_batch(8, generator=gen_a)
    inputs_b, targets_b = dataset.get_batch(8, generator=gen_b)

    torch.testing.assert_close(inputs_a, inputs_b)
    torch.testing.assert_close(targets_a, targets_b)
