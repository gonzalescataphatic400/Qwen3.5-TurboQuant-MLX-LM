import numpy as np

from turbomlx.core_ref.codebooks import CodebookKey, CodebookStore, generate_lloyd_max_codebook


def test_generate_codebook_shapes_and_ordering():
    entry = generate_lloyd_max_codebook(32, 2, grid_size=4097)
    assert entry.centroids.shape == (4,)
    assert entry.boundaries.shape == (3,)
    assert np.all(np.diff(entry.centroids) > 0)
    assert np.all(np.diff(entry.boundaries) > 0)
    assert entry.mse > 0


def test_codebook_store_persists_dimension_key(tmp_path):
    store = CodebookStore(cache_dir=tmp_path)
    key = CodebookKey(head_dim=16, bits=2)
    stored = store.get_or_create(key)
    loaded = store.get_or_create(key)
    assert stored.key.identifier == loaded.key.identifier
    assert np.allclose(stored.centroids, loaded.centroids)
