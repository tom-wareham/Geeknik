"""Regression coverage for channel layout, saturation and file failures."""
import numpy as np
import pytest
import soundfile as sf

import mmm.preserving_sanitizer as preserving
from mmm.sanitization.fingerprint_remover import FingerprintRemover


@pytest.mark.parametrize("layout", ["flat_mono", "mono_column", "stereo"])
@pytest.mark.parametrize("paranoid", [False, True])
def test_fingerprint_removal_preserves_layout_and_computes_paired_metrics(layout, paranoid, monkeypatch):
    sr = 22050
    t = np.arange(4096) / sr
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    if layout == "mono_column":
        audio = audio[:, None]
    elif layout == "stereo":
        audio = np.column_stack([audio, 0.1 * np.sin(2 * np.pi * 660 * t)])
    original = audio.copy()
    remover = FingerprintRemover(paranoid_mode=paranoid)
    calculate = remover._calculate_quality_metrics

    def check_metric_shapes(before, after, sample_rate):
        # A (samples, 1) minus (samples,) broadcast creates a samples-squared
        # matrix: reject mismatched axes before calculating any metrics.
        assert before.shape == after.shape
        assert before.ndim == 2
        return calculate(before, after, sample_rate)

    monkeypatch.setattr(remover, "_calculate_quality_metrics", check_metric_shapes)
    result = remover.remove_fingerprints(audio, sr)
    cleaned = result["cleaned_audio"]
    assert cleaned.shape == original.shape
    assert np.isfinite(cleaned).all()
    np.testing.assert_array_equal(audio, original)
    expected_snr = 10 * np.log10(np.mean(original ** 2) / np.mean((original - cleaned) ** 2))
    assert result["quality_metrics"]["snr_db"] == pytest.approx(expected_snr)
    assert np.isfinite(list(result["quality_metrics"].values())).all()


@pytest.mark.parametrize("paranoid", [False, True])
@pytest.mark.parametrize("sr", [44100, 48000])
def test_warmth_bounds_overload_and_preserves_silence(paranoid, sr):
    t = np.arange(4096) / sr
    audio = np.column_stack([4 * np.sin(2 * np.pi * 440 * t), np.zeros(len(t))])
    original = audio.copy()
    result = preserving._apply_analog_warmth(audio, sr, paranoid)
    assert result.shape == audio.shape
    assert np.isfinite(result).all()
    assert np.max(np.abs(result)) < 1
    np.testing.assert_array_equal(result[:, 1], 0)
    np.testing.assert_array_equal(audio, original)


@pytest.mark.parametrize("paranoid", [False, True])
def test_warmth_saturation_has_unity_small_signal_gain(monkeypatch, paranoid):
    # Isolate saturation from the intentional high-pass filter.
    monkeypatch.setattr(preserving, "filtfilt", lambda b, a, audio, axis: audio.copy())
    audio = np.full((1024, 1), 1e-4)
    result = preserving._apply_analog_warmth(audio, 44100, paranoid)
    np.testing.assert_allclose(result, audio, rtol=1e-6, atol=0)


@pytest.mark.parametrize("failure", ["same_file", "missing_parent", "directory_input"])
def test_file_errors_return_failure_without_changing_source(tmp_path, failure):
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(2205), 44100)
    original_bytes = source.read_bytes()
    output = tmp_path / "output.wav"
    input_path = source
    if failure == "same_file":
        output = source
    elif failure == "missing_parent":
        output = tmp_path / "missing" / "output.wav"
    else:
        input_path = tmp_path
    result = preserving.preserving_sanitize(input_path, output)
    assert result["success"] is False
    assert result["error"]
    assert source.read_bytes() == original_bytes
    if output != source:
        assert not output.exists()
