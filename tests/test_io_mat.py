import numpy as np
import pytest
import scipy.io

from pol.io_mat import MatReader


def test_matreader_context_manager_reads_old_mat(tmp_path):
    path = tmp_path / "data.mat"
    scipy.io.savemat(path, {"a": np.arange(6, dtype=np.float32).reshape(2, 3)})
    with MatReader(path) as reader:
        got = reader.read_field("a")
        assert got.shape == (2, 3)
    assert reader.data is None


def test_matreader_close_closes_hdf5_handle(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "data_h5.mat"
    with h5py.File(path, "w") as f:
        f.create_dataset("a", data=np.arange(6, dtype=np.float32).reshape(3, 2))
    reader = MatReader(path)
    handle = reader.data
    assert bool(handle.id.valid)
    assert reader.read_field("a").shape == (2, 3)
    reader.close()
    assert reader.data is None
    assert not bool(handle.id.valid)


def test_matreader_existing_direct_usage_still_works(tmp_path):
    path = tmp_path / "data.mat"
    scipy.io.savemat(path, {"a": np.ones((2, 2), dtype=np.float32)})
    reader = MatReader(path)
    assert reader.read_field("a").shape == (2, 2)
    reader.close()
