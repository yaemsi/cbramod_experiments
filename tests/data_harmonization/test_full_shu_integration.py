from __future__ import annotations

import os
from pathlib import Path

import pytest

from cbramod_experiments.data_harmonization import audit_arrow_shu


@pytest.mark.integration
def test_full_shu_manifest_matches_paper_protocol() -> None:
    """Validate the full corpus without rebuilding it during pytest."""
    manifest = Path(
        os.environ.get(
            "SHU_MI_MANIFEST",
            "resources/data/harmonized/shu_mi/manifest.parquet",
        )
    )
    if not manifest.exists():
        pytest.skip(
            "Set SHU_MI_MANIFEST to an existing full harmonized SHU-MI manifest"
        )

    audit = audit_arrow_shu(manifest, require_complete_protocol=True)
    assert audit.examples == 11_988
    assert audit.channels == 32
    assert audit.points == 800
    assert audit.complete_subject_protocol
    assert audit.paper_protocol_ready
