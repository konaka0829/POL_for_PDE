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


def test_matreader_scalar_metadata_preserves_strings_and_bools(tmp_path):
    path = tmp_path / "meta.mat"
    scipy.io.savemat(
        path,
        {
            "a": np.ones((2, 2), dtype=np.float32),
            "ic_type": "grf",
            "solver": "split_step",
            "dealias": np.array([[True]]),
            "nu": np.array([[0.01]]),
        },
    )
    with MatReader(path) as reader:
        assert reader.read_scalar_meta("ic_type") == "grf"
        assert reader.read_scalar_meta("solver") == "split_step"
        assert bool(reader.read_scalar_meta("dealias")) is True
        assert reader.read_scalar_meta("nu") == pytest.approx(0.01)


def test_matreader_read_field_rejects_string_metadata(tmp_path):
    path = tmp_path / "meta.mat"
    scipy.io.savemat(path, {"ic_type": "grf"})
    with MatReader(path) as reader:
        with pytest.raises(TypeError, match="read_meta"):
            reader.read_field("ic_type")
