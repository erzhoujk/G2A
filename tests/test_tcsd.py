import pytest

from g2a.tcsd import compute_tcsd_evidence


def test_tcsd_unsafe_minus_safe_direction() -> None:
    evidence = compute_tcsd_evidence([-0.2, -0.3], [-0.8, -0.7])
    assert evidence.risk == pytest.approx(0.5)
    assert evidence.token_count == 2


def test_tcsd_rejects_different_tokenizations() -> None:
    with pytest.raises(ValueError, match="identical action tokens"):
        compute_tcsd_evidence([-0.2], [-0.8, -0.7])
