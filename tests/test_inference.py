"""Tests for the inference engine."""

from lung_screener.inference import NoduleFinding, ScanResult


class TestNoduleFinding:
    def test_lung_rads_small(self):
        finding = NoduleFinding(x=0, y=0, z=0, diameter_mm=4.0, confidence=0.9)
        assert finding.lung_rads == "2"

    def test_lung_rads_medium(self):
        finding = NoduleFinding(x=0, y=0, z=0, diameter_mm=7.0, confidence=0.9)
        assert finding.lung_rads == "3"

    def test_lung_rads_4a(self):
        finding = NoduleFinding(x=0, y=0, z=0, diameter_mm=10.0, confidence=0.9)
        assert finding.lung_rads == "4A"

    def test_lung_rads_4b(self):
        finding = NoduleFinding(x=0, y=0, z=0, diameter_mm=20.0, confidence=0.9)
        assert finding.lung_rads == "4B"

    def test_to_dict(self):
        finding = NoduleFinding(x=1.0, y=2.0, z=3.0, diameter_mm=5.0, confidence=0.85)
        d = finding.to_dict()
        assert d["location_mm"] == {"x": 1.0, "y": 2.0, "z": 3.0}
        assert d["diameter_mm"] == 5.0
        assert d["confidence"] == 0.85


class TestScanResult:
    def test_empty_result(self):
        result = ScanResult(series_uid="test")
        d = result.to_dict()
        assert d["num_findings"] == 0
        assert d["lung_rads_overall"] == "1"

    def test_overall_lung_rads(self):
        findings = [
            NoduleFinding(x=0, y=0, z=0, diameter_mm=4.0, confidence=0.9),
            NoduleFinding(x=10, y=10, z=10, diameter_mm=10.0, confidence=0.8),
        ]
        result = ScanResult(series_uid="test", findings=findings)
        assert result.lung_rads_overall == "4A"

    def test_summary_format(self):
        finding = NoduleFinding(x=100, y=200, z=50, diameter_mm=8.5, confidence=0.92)
        result = ScanResult(series_uid="test123", findings=[finding])
        summary = result.summary()
        assert "test123" in summary
        assert "8.5mm" in summary
        assert "1" in summary  # num findings
