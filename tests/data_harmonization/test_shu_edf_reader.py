from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cbramod_experiments.data_harmonization.readers import SHUEdfReader, SHUMatReader


@pytest.mark.integration
def test_edf_and_mat_reconstruct_the_same_shu_trials(
    shu_single_session_root: Path,
) -> None:
    mat_windows = list(
        SHUMatReader().iter_windows(
            shu_single_session_root / "mat_files", target_sampling_rate_hz=250.0
        )
    )
    edf_windows = list(
        SHUEdfReader().iter_windows(
            shu_single_session_root / "edf_files",
            events_root=shu_single_session_root / "events",
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
