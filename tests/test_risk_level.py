from ml.inference import risk_level


def test_calibrated_risk_bands() -> None:
    assert risk_level(0.0) == "Low"
    assert risk_level(0.3999) == "Low"
    assert risk_level(0.4) == "Medium"
    assert risk_level(0.6999) == "Medium"
    assert risk_level(0.7) == "High"

