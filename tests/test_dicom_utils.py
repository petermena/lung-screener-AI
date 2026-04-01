import numpy as np

from app.dicom_utils import normalize_for_display


def test_normalize_for_display_range():
    arr = np.array([[-1000, -600, 400]], dtype=np.float32)
    out = normalize_for_display(arr)
    assert out.dtype == np.uint8
    assert out.min() >= 0
    assert out.max() <= 255
