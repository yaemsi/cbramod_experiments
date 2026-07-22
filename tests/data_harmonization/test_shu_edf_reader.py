from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cbramod_experiments.data_harmonization.readers import SHUEdfReader, SHUMatReader


def _session_key(path: Path) -> str:
    return path.name.removesuffix("_eeg.mat").removesuffix("_eeg.edf")


@pytest.mark.integration
def test_edf_and_mat_reconstruct_the_same_shu_trials(
    shu_real_root: Path,
) -> None:
    """Compare matching recordings without assuming every optional EDF is valid."""
    mat_reader = SHUMatReader()
    edf_reader = SHUEdfReader(strict=False)

    mat_files = {
        _session_key(path): path
        for path in mat_reader.discover(shu_real_root / "mat_files")
    }
    edf_files = {
        _session_key(path): path
        for path in edf_reader.discover(shu_real_root / "edf_files")
    }
    common_sessions = sorted(mat_files.keys() & edf_files.keys())
    assert common_sessions

    compared_recordings = 0
    compared_trials = 0
    value_count = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_yy = 0.0
    sum_xy = 0.0
    max_absolute_error = 0.0

    sum_squared_error = 0.0
    sum_absolute_error = 0.0

    for session in common_sessions:
        mat_windows = list(
            mat_reader.iter_windows(
                mat_files[session],
                target_sampling_rate_hz="auto",
                window_duration_seconds=0,
                window_stride_seconds=0,
            )
        )
        edf_windows = list(
            edf_reader.iter_windows(
                edf_files[session],
                events_root=shu_real_root / "events",
                target_sampling_rate_hz="auto",
                window_duration_seconds=0,
                window_stride_seconds=0,
            )
        )

        # In lenient mode, an invalid EDF produces no examples and is listed in
        # the reader audit rather than shifting all subsequent MAT/EDF pairs.
        if not edf_windows:
            continue

        assert len(mat_windows) == len(edf_windows)
        assert [window.sample_id for window in mat_windows] == [
            window.sample_id for window in edf_windows
        ]
        assert [window.label for window in mat_windows] == [
            window.label for window in edf_windows
        ]

        mat_signals = np.stack([window.signal for window in mat_windows])
        edf_signals = np.stack([window.signal for window in edf_windows])
        assert mat_signals.shape == edf_signals.shape

        x = mat_signals.astype(np.float64, copy=False).reshape(-1)
        y = edf_signals.astype(np.float64, copy=False).reshape(-1)

        difference = x - y

        max_absolute_error = max(
            max_absolute_error,
            float(np.abs(difference).max(initial=0.0)),
        )
        sum_squared_error += float(np.dot(difference, difference))
        sum_absolute_error += float(np.abs(difference).sum())
        max_absolute_error = max(
            max_absolute_error, float(np.abs(difference).max(initial=0))
        )
        sum_x += float(x.sum())
        sum_y += float(y.sum())
        sum_xx += float(np.dot(x, x))
        sum_yy += float(np.dot(y, y))
        sum_xy += float(np.dot(x, y))
        value_count += x.size
        compared_recordings += 1
        compared_trials += len(mat_windows)

    assert compared_recordings > 0
    assert compared_trials > 0
    assert value_count > 0

    numerator = value_count * sum_xy - sum_x * sum_y
    denominator = np.sqrt(
        (value_count * sum_xx - sum_x**2) * (value_count * sum_yy - sum_y**2)
    )
    correlation = numerator / denominator
    """
    assert correlation > 0.999999
    assert max_absolute_error <= 1e-3
    """
    rmse = np.sqrt(sum_squared_error / value_count)
    mae = sum_absolute_error / value_count
    reference_rms = np.sqrt(sum_xx / value_count)
    normalized_rmse = rmse / max(reference_rms, np.finfo(np.float64).eps)

    assert correlation > 0.999999
    assert normalized_rmse < 0.05

    # EDF is quantized, so exact float equality with MAT is not expected.
    # The observed 3.75 maximum is consistent with approximately half an
    # EDF physical quantization step.
    assert max_absolute_error <= 4.0

    assert normalized_rmse < 0.05, (
        f"MAT/EDF normalized RMSE too large: {normalized_rmse:.6f}; "
        f"RMSE={rmse:.6f}, MAE={mae:.6f}, "
        f"max_abs_error={max_absolute_error:.6f}, "
        f"correlation={correlation:.9f}"
    )

    report = edf_reader.audit_report()
    assert report["processed_recordings"] == compared_recordings
    assert report["skipped_recordings"] == len(edf_reader.failures)


def test_lenient_reader_records_and_skips_invalid_edf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_path = tmp_path / "sub-005_ses-04_task_motorimagery_eeg.edf"
    bad_path.touch()

    reader = SHUEdfReader(strict=False)

    def fail_recording(path: Path, **_: Any) -> list[Any]:
        raise ValueError("could not convert string to float: '        '")

    monkeypatch.setattr(reader, "_read_recording_windows", fail_recording)

    windows = list(reader.iter_windows(tmp_path, events_root=tmp_path))
    assert windows == []
    assert reader.processed_recordings == 0
    assert len(reader.failures) == 1

    failure = reader.failures[0]
    assert Path(failure.path).name == bad_path.name
    assert failure.error_type == "ValueError"
    assert "could not convert string to float" in failure.error


def test_strict_reader_reraises_invalid_edf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_path = tmp_path / "sub-005_ses-04_task_motorimagery_eeg.edf"
    bad_path.touch()

    reader = SHUEdfReader(strict=True)

    def fail_recording(path: Path, **_: Any) -> list[Any]:
        raise ValueError("malformed EDF header")

    monkeypatch.setattr(reader, "_read_recording_windows", fail_recording)

    with pytest.raises(ValueError, match="malformed EDF header"):
        list(reader.iter_windows(tmp_path, events_root=tmp_path))

    assert len(reader.failures) == 1
    assert reader.failures[0].path == str(bad_path)
