from PIL import Image

from ml.frequency import FREQUENCY_FEATURE_DIM, extract_frequency_features


def test_frequency_features_are_stable_and_finite() -> None:
    image = Image.new("RGB", (63, 79), color=(25, 80, 140))
    features = extract_frequency_features(image)
    assert features.shape == (FREQUENCY_FEATURE_DIM,)
    assert features.dtype.name == "float32"
    assert features.tolist() == features.tolist()

