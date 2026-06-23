import torch

from scratch_transformer.attention import KVCache


def test_kv_cache_starts_empty() -> None:
    cache = KVCache()

    assert cache.keys is None
    assert cache.values is None
    assert cache.length == 0


def test_kv_cache_first_append_stores_and_returns_tensors() -> None:
    cache = KVCache()
    keys = torch.randn(2, 4, 3, 8)
    values = torch.randn(2, 4, 3, 8)

    cached_keys, cached_values = cache.append(keys, values)

    assert cache.length == 3
    assert cached_keys is cache.keys
    assert cached_values is cache.values
    torch.testing.assert_close(cached_keys, keys)
    torch.testing.assert_close(cached_values, values)


def test_kv_cache_appends_along_sequence_dimension() -> None:
    cache = KVCache()
    first_keys = torch.randn(2, 4, 1, 8)
    first_values = torch.randn(2, 4, 1, 8)
    second_keys = torch.randn(2, 4, 2, 8)
    second_values = torch.randn(2, 4, 2, 8)

    cache.append(first_keys, first_values)
    cached_keys, cached_values = cache.append(second_keys, second_values)

    assert cache.length == 3
    torch.testing.assert_close(cached_keys, torch.cat([first_keys, second_keys], dim=-2))
    torch.testing.assert_close(cached_values, torch.cat([first_values, second_values], dim=-2))


def test_kv_cache_preserves_dtype_and_device() -> None:
    cache = KVCache()
    keys = torch.randn(1, 2, 3, 4, dtype=torch.float64)
    values = torch.randn(1, 2, 3, 4, dtype=torch.float64)

    cached_keys, cached_values = cache.append(keys, values)

    assert cached_keys.dtype == keys.dtype
    assert cached_values.dtype == values.dtype
    assert cached_keys.device == keys.device
    assert cached_values.device == values.device
