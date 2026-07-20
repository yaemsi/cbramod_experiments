from __future__ import annotations

from pathlib import Path

import numpy as np

from cbramod_experiments.data_harmonization.readers import SHUEdfReader, SHUMatReader


RESOURCE_ROOT = Path("resources/shu-mi_dataset")


def test_edf_and_mat_reconstruct_the_same_shu_trials() -> None:
    mat_windows = list(
        SHUMatReader().iter_windows(
            RESOURCE_ROOT / "mat_files", target_sampling_rate_hz=250.0
        )
    )
    edf_windows = list(
        SHUEdfReader().iter_windows(
            RESOURCE_ROOT / "edf_files",
            events_root=RESOURCE_ROOT / "events",
            target_sampling_rate_hz=250.0,
        )
    )

    assert len(mat_windows) == len(edf_windows) == 100
    assert [window.label for window in mat_windows] == [
        window.label for window in edf_windows
    ]
    mat_signals = np.stack([window.signal for window in mat_windows])
    edf_signals = np.stack([window.signal for window in edf_windows])
    assert mat_signals.shape == edf_signals.shape == (100, 32, 1000)
    correlation = float(
        np.corrcoef(mat_signals.reshape(-1), edf_signals.reshape(-1))[0, 1]
    )
    assert correlation > 0.999999
    np.testing.assert_allclose(mat_signals, edf_signals, rtol=3e-5, atol=6e-4)
