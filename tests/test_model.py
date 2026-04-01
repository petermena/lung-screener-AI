import numpy as np

from app.model import LungNoduleModel


def test_model_predict_contract():
    model = LungNoduleModel()
    vol = np.zeros((8, 16, 16), dtype=np.float32) - 700
    out = model.predict(vol)
    assert 0.0 <= out.risk_score <= 1.0
    assert isinstance(out.findings, list)
    assert "Model adapter target" in out.notes
