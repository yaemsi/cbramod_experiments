from __future__ import annotations

import numpy as np
import pytest

from cbramod_experiments.data_harmonization.schema import EEGWindow


def test_window_validation_rejects_mismatched_channel_mask() -> None:
    window = EEGWindow(
        signal=np.zeros((2, 8), dtype=np.float32),
        sampling_rate_hz=2.0,
        channel_names=("C3", "C4"),
        channel_mask=np.ones(1, dtype=np.bool_),
        dataset_id="test",
        subject_id="sub-1",
        session_id=None,
        task=None,
        start_seconds=0.0,
        duration_seconds=4.0,
        label=None,
        split=None,
        source_uri="test",
        source_format="synthetic",
        sample_id="sample-1",
    )
    with pytest.raises(ValueError, match="channel_mask"):
        window.validate()
