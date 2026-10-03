#!/usr/bin/env python3
# pylint: disable=too-many-lines
"""
PRESERVING Sanitizer - Removes threats while keeping audio playable
"""

import os
import logging
import librosa
import numpy as np
from pathlib import Path
import time
import soundfile as sf
import shutil
from numpy.fft import fft, ifft, fftfreq
from scipy.signal import butter, filtfilt
import random

logger = logging.getLogger(__name__)

def _configure_thread_counts() -> None:
    """Set thread counts for numeric libraries. Call once before heavy computation."""
    cpu_count = str(os.cpu_count() or 1)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(var, cpu_count)


def _repair_non_finite_audio(audio: np.ndarray, label: str = "processing") -> np.ndarray:
    """Fail-closed repair for NaN/Inf propagation in DSP stages."""
    if np.all(np.isfinite(audio)):
        return audio

    logger.warning("Non-finite audio detected after %s; applying safety repair", label)
    repaired = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    repaired = np.clip(repaired, -1.0, 1.0)
    return repaired


def preserving_sanitize(
    input_file,
    output_file=None,
    paranoid_mode=False,
    threat_count=0,
    output_format=None,
    phase_dither=True,
    comb_mask=True,
    transient_shift=True,
    resample_nudge=True,
    gated_resample_nudge=False,
    phase_noise=True,
    phase_swirl=True,
    masked_hf_phase=False,
    micro_eq_flutter=False,
    hf_decorrelate=False,
    refined_transient=False,
    adaptive_transient=False,
    spectral_clean=True,
    fingerprint_remove=True,
    tempo_drift=True,
    onset_velocity=True,
    mfcc_perturb=True,
    seed=None,
):
    """
    Audio sanitization that PRESERVES audio quality while removing threats

    Args:
        seed: Optional int for reproducible random state. When set, all
              random operations use this seed, making output deterministic
              for the same input. Useful for debugging and regression tests.

    Returns:
        Dict containing sanitization results
    """
    _configure_thread_counts()

    input_file = Path(input_file)
    if not input_file.is_file():
        return {"success": False, "error": f"Input audio file not found: {input_file}"}
    if output_file is not None:
        output_file = Path(output_file)

    if seed is not None:
        np.random.seed(seed)

    print(f"🎵 PRESERVING SANITIZATION - Keeping audio alive!")
    print(f"   Input: {input_file}")
    # Decide output format and path up front so we don't dump WAV data into an MP3 filename
    normalized_format = (output_format or "").lower().lstrip(".")
    if normalized_format in ("", "preserve"):
        normalized_format = input_file.suffix.lstrip(".").lower()
    if not normalized_format:
        normalized_format = "wav"

    if output_file:
        output_file = output_file.with_suffix(f".{normalized_format}")
    else:
        output_file = input_file.with_suffix(f".clean.{normalized_format}")

    print(f"   Output: {output_file or 'auto-generated'}")
    print(f"   Paranoid mode: {paranoid_mode}")
    print(f"   Target threats: {threat_count}")
    print()

    start_time = time.time()

    # Phase 1: Complete metadata annihilation
    print("🔥 Phase 1: Complete metadata annihilation...")
    phase_start = time.time()

    # Copy file first
    try:
        shutil.copy2(input_file, output_file)
    except (OSError, shutil.Error) as e:
        logger.warning("Could not copy input audio to output: %s", e)
        return {"success": False, "error": "Could not copy input audio to output."}

    # Remove metadata using mutagen
    try:
        from mutagen import File as MutagenFile

        audio_file = MutagenFile(output_file)
        if audio_file is not None:
            audio_file.delete()
            audio_file.save()
        metadata_time = time.time() - phase_start
        print(f"   ✅ Metadata completely annihilated in {metadata_time:.2f}s")
    except Exception as e:
        print(f"   ⚠️ Metadata removal failed: {e}")
        metadata_time = time.time() - phase_start

    # Load the audio for PRESERVING processing
    print("🎵 Loading audio for PRESERVING threat removal...")
    load_start = time.time()
    try:
        # Load as float32 for better precision, keep channel layout
        audio, sr = librosa.load(str(input_file), sr=None, mono=False, dtype=np.float32)
        audio = _ensure_channel_layout(audio)
        duration = audio.shape[0] / sr
        load_time = time.time() - load_start
        print(f"   ✅ Loaded {duration:.1f}s in {load_time:.2f}s")
        print(
            f"   📊 Audio stats: Max={np.max(np.abs(audio)):.4f}, Mean={np.mean(np.abs(audio)):.6f}, Channels={audio.shape[1]}"
        )
    except Exception as e:
        print(f"   ❌ Failed to load audio: {e}")
        return {"success": False, "error": "Failed to load audio."}

    # Phase 2: PRESERVING threat neutralization
    print("🎵 Phase 2: PRESERVING threat neutralization...")
    phase_start = time.time()

    # CRITICAL: Make a copy and preserve the original amplitude
    sanitized_audio = audio.copy()
    original_rms = np.sqrt(np.mean(sanitized_audio**2))

    # Preserve per-channel RMS ratio to prevent stereo balance drift
    channel_rms = np.array(
        [np.sqrt(np.mean(sanitized_audio[:, ch] ** 2))
         for ch in range(sanitized_audio.shape[1])]
    )

    print(f"   🔍 Original RMS level: {original_rms:.6f}")

    # 1. VERY GENTLE spectral modification
    print("   🎯 Applying GENTLE spectral modification...")
    sanitized_audio = _gentle_spectral_phase_noise(sanitized_audio, sr, paranoid_mode)

    # 1b. Remove spectral watermarks (known frequency bands, periodic patterns)
    if spectral_clean:
        print("   🎯 Removing spectral watermarks...")
        sanitized_audio = _apply_spectral_watermark_cleaning(
            sanitized_audio, sr, paranoid_mode
        )

    # 1c. Remove statistical fingerprints (entropy, kurtosis, timing patterns)
    if fingerprint_remove:
        print("   🎯 Removing statistical fingerprints...")
        sanitized_audio = _apply_fingerprint_removal(
            sanitized_audio, sr, paranoid_mode
        )

    # Restore RMS after watermark/fingerprint removal
    mid_rms = np.sqrt(np.mean(sanitized_audio**2))
    if mid_rms > 0:
        sanitized_audio = sanitized_audio * (original_rms / mid_rms)

    # 2. Add very subtle noise only to high-frequency bands + low floor dither
    print("   🎯 Adding SUBTLE high-frequency noise and human dither...")
    sanitized_audio = _add_hf_noise_and_dither(sanitized_audio, sr, paranoid_mode)

    # 3. Add human-like micro-variations to break AI-regular timing
    print("   🎯 Adding micro timing/pitch humanization...")
    sanitized_audio = _apply_humanization(sanitized_audio, sr, paranoid_mode)

    # 4. Micro resample warp to disturb spectral/temporal regularity
    print("   🎯 Applying micro resample warp...")
    sanitized_audio = _apply_micro_resample_warp(sanitized_audio, sr, paranoid_mode)

    # 5. Subtle analog coloration + micro ambience + gentle bandlimit to trim watermark bands
    print("   🎯 Adding analog warmth, micro ambience, and gentle bandlimit...")
    sanitized_audio = _apply_analog_warmth(sanitized_audio, sr, paranoid_mode)
    sanitized_audio = _add_micro_ambience(sanitized_audio, sr, paranoid_mode)
    sanitized_audio = _apply_gentle_bandlimit(sanitized_audio, sr, paranoid_mode)

    # 6. Apply subtle resample nudge + phase swirl + phase noise + micro EQ motion (transparent)
    print("   🎯 Applying stealth resample nudge and phase swirl...")
    if resample_nudge:
        sanitized_audio = _apply_resample_nudge(sanitized_audio, sr, paranoid_mode)
    if gated_resample_nudge:
        sanitized_audio = _apply_rms_gated_resample_nudge(
            sanitized_audio, sr, paranoid_mode
        )
    if phase_swirl:
        sanitized_audio = _apply_phase_swirl(sanitized_audio, sr, paranoid_mode)
    if phase_noise:
        sanitized_audio = _apply_phase_noise_fft(sanitized_audio, paranoid_mode)
    if masked_hf_phase:
        sanitized_audio = _apply_masked_hf_phase_noise(
            sanitized_audio, sr, paranoid_mode
        )
    if hf_decorrelate:
        sanitized_audio = _apply_hf_decorrelate(sanitized_audio, sr, paranoid_mode)
    sanitized_audio = _apply_micro_eq_modulation(sanitized_audio, sr, paranoid_mode)

    # 7. Optional stealth extras
    if (
        phase_dither
        or comb_mask
        or transient_shift
        or micro_eq_flutter
        or refined_transient
        or adaptive_transient
    ):
        print("   🎯 Applying optional stealth steps...")
        if phase_dither:
            sanitized_audio = _apply_subblock_phase_dither(
                sanitized_audio, sr, paranoid_mode
            )
        if comb_mask:
            sanitized_audio = _apply_dynamic_comb_mask(
                sanitized_audio, sr, paranoid_mode
            )
        if transient_shift:
            sanitized_audio = _apply_transient_micro_shift(
                sanitized_audio, sr, paranoid_mode
            )
        if micro_eq_flutter:
            sanitized_audio = _apply_gated_micro_eq_flutter(
                sanitized_audio, sr, paranoid_mode
            )
        if refined_transient:
            sanitized_audio = _apply_refined_transient_shift(
                sanitized_audio, sr, paranoid_mode
            )
        if adaptive_transient:
            sanitized_audio = _apply_adaptive_transient_shift(
                sanitized_audio, sr, paranoid_mode
            )

    # 7b. Onset velocity variation (break AI-consistent attack energy)
    if onset_velocity:
        print("   🎯 Applying onset velocity variation...")
        sanitized_audio = _apply_onset_velocity_variation(
            sanitized_audio, sr, paranoid_mode
        )

    # 7c. Long-range tempo drift (break AI-perfect BPM)
    if tempo_drift:
        print("   🎯 Applying long-range tempo drift...")
        sanitized_audio = _apply_long_range_tempo_drift(
            sanitized_audio, sr, paranoid_mode
        )

    # 7d. MFCC-targeted perturbation (defeat MFCC-based classifiers)
    if mfcc_perturb:
        print("   🎯 Applying MFCC perturbation...")
        sanitized_audio = _apply_mfcc_perturbation(sanitized_audio, sr, paranoid_mode)
        sanitized_audio = _repair_non_finite_audio(
            sanitized_audio, label="MFCC perturbation"
        )

    # 8. Restore a touch of clarity lost to masking
    print("   🎯 Restoring clarity tilt...")
    sanitized_audio = _apply_clarity_tilt(sanitized_audio, sr, paranoid_mode)

    # Restore per-channel stereo balance (prevents L/R drift from independent processing)
    for ch in range(sanitized_audio.shape[1]):
        ch_rms = np.sqrt(np.mean(sanitized_audio[:, ch] ** 2))
        if ch_rms > 0 and channel_rms[ch] > 0:
            sanitized_audio[:, ch] *= channel_rms[ch] / ch_rms

    # 6. Restore original audio level (CRITICAL!)
    print("   🎯 Restoring original audio level...")
    new_rms = np.sqrt(np.mean(sanitized_audio**2))
    if new_rms > 0:
        sanitized_audio = sanitized_audio * (original_rms / new_rms)
        print(f"   ✅ Restored RMS level: {np.sqrt(np.mean(sanitized_audio ** 2)):.6f}")

    # 5. Gentle limiting to prevent clipping, then restore RMS
    sanitized_audio = np.tanh(sanitized_audio * 0.95)  # Soft limiting
    post_tanh_rms = np.sqrt(np.mean(sanitized_audio**2))
    if post_tanh_rms > 0:
        sanitized_audio = sanitized_audio * (original_rms / post_tanh_rms)

    # 6. Final quality check
    sanitized_audio = _repair_non_finite_audio(sanitized_audio, label="finalization")
    final_rms = np.sqrt(np.mean(sanitized_audio**2))
    peak = np.max(np.abs(sanitized_audio))
    print(f"   📊 Final audio stats: RMS={final_rms:.6f}, Peak={peak:.4f}")

    cleaning_time = time.time() - phase_start
    print(f"   ✅ PRESERVING sanitization completed in {cleaning_time:.2f}s")

    # Phase 3: Save the preserved audio
    print("💾 Saving preserved audio...")
    save_start = time.time()

    try:
        # Normalize and convert to int16 for exporting
        sanitized_audio = np.real(sanitized_audio)
        if sanitized_audio.ndim == 1:
            sanitized_audio = np.expand_dims(sanitized_audio, axis=1)

        peak_val = (
            float(np.max(np.abs(sanitized_audio))) if sanitized_audio.size else 1.0
        )
        if peak_val > 1.0:
            sanitized_audio = sanitized_audio / peak_val

        sanitized_audio_int16 = np.clip(sanitized_audio * 32767, -32768, 32767).astype(
            np.int16
        )

        if normalized_format == "mp3":
            try:
                from pydub import AudioSegment
            except Exception as e:
                return {
                    "success": False,
                    "error": f"MP3 export requires pydub/ffmpeg: {e}",
                }

            segment = AudioSegment(
                sanitized_audio_int16.tobytes(),
                frame_rate=sr,
                sample_width=2,
                channels=sanitized_audio_int16.shape[1],
            )
            segment.export(
                str(output_file),
                format="mp3",
                bitrate="320k",
                parameters=[
                    "-map_metadata",
                    "-1",
                    "-write_xing",
                    "0",
                    "-id3v2_version",
                    "0",
                    "-write_id3v1",
                    "0",
                ],
            )
        else:
            sf.write(
                str(output_file),
                sanitized_audio_int16,
                sr,
                format=normalized_format.upper(),
                subtype="PCM_16",
            )

        save_time = time.time() - save_start
        print(f"   ✅ Saved in {save_time:.2f}s")
    except Exception as e:
        print(f"   ❌ Failed to save: {e}")
        return {"success": False, "error": "Failed to save sanitized audio."}

    total_time = time.time() - start_time

    # Report honest stats.  We know metadata was deleted (Phase 1).
    # The audio-domain processing applies spectral perturbation; we
    # cannot count individual "watermarks removed" because we don't
    # know what the detector would have found.  Be transparent.
    stats = {
        "metadata_removed": 1,  # mutagen.delete() strips all tags
        "watermarks_removed": 0,  # spectral perturbation applied, not measured
        "watermarks_detected": 0,
        "processing_time": total_time,
        "processing_speed": f"{duration / total_time:.1f}x real-time",
    }

    print(f"\n🎉 PRESERVING SANITIZATION COMPLETE!")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Processing speed: {stats['processing_speed']}")
    print(f"   Output: {output_file}")

    return {"success": True, "output_file": str(output_file), "stats": stats}


# --- helper functions ---


def _ensure_channel_layout(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio shape to (samples, channels).
    """
    if audio is None:
        return audio

    if audio.ndim == 1:
        return audio.reshape(-1, 1)

    # librosa with mono=False returns (channels, samples)
    if audio.shape[0] < audio.shape[1]:
        return np.ascontiguousarray(audio.T)

    return audio


def _gentle_spectral_phase_noise(
    audio: np.ndarray, sr: int, paranoid_mode: bool
) -> np.ndarray:
    """
    Apply tiny phase perturbations, heavier above 8-10kHz, per channel.
    """
    audio = np.real(audio)
    for ch in range(audio.shape[1]):
        channel = audio[:, ch]
        spectrum = fft(channel)
        freqs = fftfreq(len(channel), 1 / sr)

        high_cut = 8000 if paranoid_mode else 10000
        mask = np.abs(freqs) > high_cut

        phase_jitter = np.random.uniform(-0.12, 0.12, len(spectrum))
        spectrum[mask] = np.abs(spectrum[mask]) * np.exp(
            1j * (np.angle(spectrum[mask]) + phase_jitter[mask])
        )

        # tiny broadband phase wobble
        light_mask = ~mask
        spectrum[light_mask] = np.abs(spectrum[light_mask]) * np.exp(
            1j * (np.angle(spectrum[light_mask]) + phase_jitter[light_mask] * 0.2)
        )

        audio[:, ch] = np.real(ifft(spectrum))

    return audio


def _add_hf_noise_and_dither(
    audio: np.ndarray, sr: int, paranoid_mode: bool
) -> np.ndarray:
    """
    Add shaped noise above 12-15kHz and a low noise floor dither to mask AI-regular spectra.
    """
    noise_level = 1.8e-7 if paranoid_mode else 9e-8  # lower to reduce hiss
    dither_level = 4e-6 if paranoid_mode else 2e-6
    nyquist = sr / 2
    hf_cut = 13500 if paranoid_mode else 17500

    for ch in range(audio.shape[1]):
        n = len(audio)
        t = np.arange(n) / sr
        white = np.random.normal(0, 1, n)

        # high-pass the hiss so it lives in upper band
        try:
            b, a = butter(4, hf_cut / nyquist, btype="high")
            hf_noise = filtfilt(b, a, white) * noise_level
        except Exception:
            hf_noise = white * noise_level

        # shaped low-level dither (brown-ish)
        brown = np.cumsum(np.random.normal(0, 1, n))
        brown = brown / (np.max(np.abs(brown)) + 1e-9) * dither_level

        audio[:, ch] = audio[:, ch] + hf_noise + brown

    return audio


def _apply_humanization(audio: np.ndarray, sr: int, paranoid_mode: bool) -> np.ndarray:
    """
    Introduce subtle wow/flutter, micro gain swings, and tiny channel decorrelation
    to nudge the signal away from AI-perfect regularity.
    """
    n, channels = audio.shape
    t = np.arange(n) / sr

    # Wow/flutter: slow LFO pitch drift via fractional delay
    rate_hz = 0.21 if paranoid_mode else 0.15
    depth_samples = 8 if paranoid_mode else 5  # up to ~0.2ms
    for ch in range(channels):
        lfo = depth_samples * np.sin(
            2 * np.pi * rate_hz * t + np.random.uniform(0, 2 * np.pi)
        )
        shifted_positions = np.clip(np.arange(n) - lfo, 0, n - 1)
        audio[:, ch] = np.interp(shifted_positions, np.arange(n), audio[:, ch])

    # Micro gain swings (simulate human dynamics)
    env_noise = np.random.normal(0, 1, n)
    # Smooth with a small window
    window = 400 if paranoid_mode else 600
    kernel = np.ones(window) / window
    smoothed = np.convolve(env_noise, kernel, mode="same")
    gain_depth = 0.012 if paranoid_mode else 0.008
    gain = 1.0 + gain_depth * smoothed
    gain = np.clip(gain, 0.98, 1.02)
    audio = audio * gain[:, None]

    # Stereo decorrelation via per-channel all-pass (no Haas effect)
    # A time-delay approach pans toward the earlier channel; all-pass
    # decorrelates phase without shifting the perceived stereo image.
    if channels >= 2:
        alpha = 0.02 if paranoid_mode else 0.012
        for ch in range(channels):
            coeff = alpha * (1 if ch == 0 else -1)
            b_ap = np.array([coeff, 1.0])
            a_ap = np.array([1.0, coeff])
            try:
                audio[:, ch] = filtfilt(b_ap, a_ap, audio[:, ch])
            except Exception as e:
                logger.warning("Stereo decorrelation filter failed on ch %d: %s", ch, e)

    return audio


def _apply_micro_resample_warp(
    audio: np.ndarray, sr: int, paranoid_mode: bool
) -> np.ndarray:
    """
    Apply tiny, random resample warp per channel to disturb perfectly uniform timing.
    """
    n, channels = audio.shape
    warped = np.zeros_like(audio)
    max_warp = 0.0022 if paranoid_mode else 0.0015  # up to ~0.22% stretch/compress

    for ch in range(channels):
        factor = 1.0 + np.random.uniform(-max_warp, max_warp)
        new_len = max(1, int(n * factor))

        src_idx = np.arange(n)
        dst_idx = np.linspace(0, n - 1, new_len)
        resampled = np.interp(dst_idx, src_idx, audio[:, ch])

        # Back to original length
        warped[:, ch] = np.interp(src_idx, np.linspace(0, n - 1, new_len), resampled)

    return warped


def _apply_resample_nudge(
    audio: np.ndarray, sr: int, paranoid_mode: bool
) -> np.ndarray:
    """
    Slight resample up/down and back to original sr to decorrelate spectral bins
    without audible pitch shift. Keeps duration identical.
    """
    eps = 0.0006 if paranoid_mode else 0.00035  # +/- ~0.06%
    factor = 1.0 + np.random.uniform(-eps, eps)
    target_len = audio.shape[0]
    nudged = np.zeros_like(audio)

    for ch in range(audio.shape[1]):
        # High-quality resample to nudged rate then back
        up = librosa.resample(
            audio[:, ch], orig_sr=sr, target_sr=sr * factor, res_type="soxr_hq"
        )
        back = librosa.resample(
            up, orig_sr=sr * factor, target_sr=sr, res_type="soxr_hq"
        )
        # Match original length
        if len(back) < target_len:
            back = np.pad(back, (0, target_len - len(back)), mode="edge")
        nudged[:, ch] = back[:target_len]

    return nudged


def _apply_rms_gated_resample_nudge(
    audio: np.ndarray, sr: int, paranoid_mode: bool
) -> np.ndarray:
    """
    Apply tiny resample nudge only on higher-energy segments to reduce audibility.
    """
    n, channels = audio.shape
    rms = np.sqrt(np.mean(audio**2, axis=1))
    thresh = np.percentile(rms, 60 if paranoid_mode else 70)

    eps = 0.0005 if paranoid_mode else 0.0003
    factor_high = 1.0 + eps
    factor_low = 1.0 - eps

    gated = audio.copy()
    # Process in chunks where RMS exceeds threshold
    win = int(sr * 0.1)  # 100 ms
    hop = win // 2
    for start in range(0, n, hop):
        end = min(n, start + win)
        if end - start < 8:
            continue
        if np.mean(rms[start:end]) < thresh:
            continue

        for ch in range(channels):
            seg = audio[start:end, ch]
            # alternate factors to avoid bias
            factor = factor_high if (start // hop) % 2 == 0 else factor_low
            up = librosa.resample(
                seg, orig_sr=sr, target_sr=sr * factor, res_type="soxr_hq"
            )
            back = librosa.resample(
                up, orig_sr=sr * factor, target_sr=sr, res_type="soxr_hq"
            )
            if len(back) < end - start:
                back = np.pad(back, (0, end - start - len(back)), mode="edge")
            gated[start:end, ch] = back[: end - start]

    return gated


def _apply_analog_warmth(audio: np.ndarray, sr: int, paranoid_mode: bool) -> np.ndarray:
    """
    Add light saturation and remove DC/ultrasonic content to mimic analog chain.
    """
    nyquist = sr / 2
    drive = 1.07 if paranoid_mode else 1.04  # back off to reduce perceived mud

    # DC/high-pass to keep low-end clean
    try:
        b, a = butter(2, 20 / nyquist, btype="high")
        audio = filtfilt(b, a, audio, axis=0)
    except Exception as e:
        logger.warning("Analog warmth high-pass filter failed: %s", e)

    # Unity gain for quiet signals and bounded peaks, including filter overshoot.
    # Dividing by tanh(drive) adds gain and allows peaks above full scale.
    audio = np.tanh(audio * drive) / drive

    return audio


def _apply_gentle_bandlimit(
    audio: np.ndarray, sr: int, paranoid_mode: bool
) -> np.ndarray:
    """
    Gently roll off ultrasonics where watermark energy often hides.
    """
    nyquist = sr / 2
    cutoff = 19000 if paranoid_mode else 20000
    cutoff = min(cutoff, nyquist - 500)

    if cutoff <= 1000:  # safety
        return audio

    try:
        b, a = butter(4, cutoff / nyquist, btype="low")
        return filtfilt(b, a, audio, axis=0)
    except Exception:
        return audio


def _add_micro_ambience(audio: np.ndarray, sr: int, paranoid_mode: bool) -> np.ndarray:
    """
    Add tiny crossfeed delay and micro ambience to reduce spectral/phase regularity.
    """
    n, channels = audio.shape
    delay_ms = 0.8 if paranoid_mode else 0.6
    delay_samples = max(1, int(sr * delay_ms / 1000))
    decay = 0.012 if paranoid_mode else 0.008

    # Crossfeed for stereo; for mono, use a subtle tapped delay
    if channels >= 2:
        left = audio[:, 0]
        right = audio[:, 1]

        l_delayed = np.pad(left, (delay_samples, 0), mode="edge")[:n]
        r_delayed = np.pad(right, (delay_samples, 0), mode="edge")[:n]

        audio[:, 0] = left + decay * r_delayed
        audio[:, 1] = right + decay * l_delayed
    else:
        delayed = np.pad(audio[:, 0], (delay_samples, 0), mode="edge")[:n]
        audio[:, 0] = audio[:, 0] + decay * delayed

    # Light all-pass-like tilt to break perfect linear phase
    alpha = 0.12 if paranoid_mode else 0.08
    b = [alpha, 1, -alpha]
    a = [1, 0, -alpha]
    try:
        audio = filtfilt(b, a, audio, axis=0)
    except Exception as e:
        logger.warning("Micro-ambience all-pass filter failed: %s", e)

    return audio


def _apply_clarity_tilt(audio: np.ndarray, sr: int, paranoid_mode: bool) -> np.ndarray:
    """
    Apply a gentle high-shelf tilt to restore clarity and reduce perceived mud.
    """
    nyquist = sr / 2
    shelf_start = 4800 if paranoid_mode else 5200
    gain_db = 1.5 if paranoid_mode else 1.2

    # Use a simple first-order shelf approximation
    w0 = shelf_start / nyquist
    if w0 >= 1.0:
        return audio

    # Biquad high shelf coefficients (RBJ cookbook)
    A = 10 ** (gain_db / 40)
    alpha = np.sin(np.pi * w0) / 2 * np.sqrt((A + 1 / A) * (1 / 0.707 - 1) + 2)
    cos_w0 = np.cos(np.pi * w0)

    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])

    try:
        return filtfilt(b, a, audio, axis=0)
    except Exception:
        return audio


def _apply_phase_swirl(audio: np.ndarray, sr: int, paranoid_mode: bool) -> np.ndarray:
    """
    Apply very light all-pass swirl to decorrelate phase without audible tone shift.
    """
    n, channels = audio.shape
    swirl = np.zeros_like(audio)
    # two short all-pass sections per channel
    alphas = [0.016, -0.014] if paranoid_mode else [0.012, -0.01]

    for ch in range(channels):
        sig = audio[:, ch]
        for a in alphas:
            b = np.array([a, 1, -a])
            a_den = np.array([1, 0, -a])
            try:
                sig = filtfilt(b, a_den, sig)
            except Exception as e:
                logger.warning("Phase swirl filter failed on ch %d: %s", ch, e)
        swirl[:, ch] = sig

    return swirl


def _apply_masked_hf_phase_noise(
    audio: np.ndarray, sr: int, paranoid_mode: bool
) -> np.ndarray:
    """
    Apply tiny, masked phase noise only in high frequencies where energy is present.
    """
    n, channels = audio.shape
    output = np.zeros_like(audio)

    # STFT parameters
    n_fft = 1024
    hop = n_fft // 4
    # HF band threshold
    hf_start = 14500 if paranoid_mode else 15500
    hf_mask_ratio = (
        0.2 if paranoid_mode else 0.15
    )  # portion of max HF magnitude to trigger noise
    max_jitter = 0.0015 if paranoid_mode else 0.0010  # radians (upper bound)
    min_jitter = 0.0005 if paranoid_mode else 0.0003

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    hf_bins = freqs >= hf_start

    for ch in range(channels):
        S = librosa.stft(
            audio[:, ch], n_fft=n_fft, hop_length=hop, window="hann", center=True
        )
        mag = np.abs(S)
        phase = np.angle(S)

        # Determine HF energy per frame
        hf_mag = mag[hf_bins, :]
        frame_max = np.max(hf_mag, axis=0, keepdims=True) + 1e-9
        bin_thresh = frame_max * hf_mask_ratio

        # Build jitter only on HF bins that exceed threshold, scaled by local energy
        jitter_hf = np.zeros_like(hf_mag)
        active = hf_mag > bin_thresh
        if np.any(active):
            energy_scale = np.sqrt(np.clip(hf_mag / (frame_max + 1e-9), 0, 1))
            jitter_vals = np.random.normal(0, max_jitter, jitter_hf.shape)
            jitter_hf = (
                (min_jitter + energy_scale * (max_jitter - min_jitter))
                * jitter_vals
                * active
            )

        # Apply to phase (HF only)
        phase[hf_bins, :] = phase[hf_bins, :] + jitter_hf

        modified = mag * np.exp(1j * phase)
        output[:, ch] = librosa.istft(modified, hop_length=hop, length=n)

    return output


def _apply_phase_noise_fft(audio: np.ndarray, paranoid_mode: bool) -> np.ndarray:
    """
    Add very small random phase noise across the spectrum to decorrelate patterns
    without changing magnitude response.
    """
    n, channels = audio.shape
    noisy = np.zeros_like(audio)

    for ch in range(channels):
        spectrum = np.fft.rfft(audio[:, ch])
        mags = np.abs(spectrum)
        phases = np.angle(spectrum)

        freqs = np.linspace(0, 1, len(phases))
        jitter = (0.0025 if paranoid_mode else 0.0015) * freqs + (
            0.0006 if paranoid_mode else 0.0004
        ) * (1 - freqs)
        phase_noise = np.random.normal(0, jitter)

        new_phase = phases + phase_noise
        new_spec = mags * np.exp(1j * new_phase)
        noisy[:, ch] = np.fft.irfft(new_spec, n=len(audio))

    return noisy


def _apply_hf_decorrelate(
    audio: np.ndarray, sr: int, paranoid_mode: bool
) -> np.ndarray:
    """
    Apply very light, time-varying phase offsets only in HF band (12-16 kHz).
    """
    n, channels = audio.shape
    output = np.zeros_like(audio)

    n_fft = 1024
    hop = n_fft // 4
    f_low, f_high = (12000, 16000) if paranoid_mode else (13000, 17000)
    max_jitter = 0.001 if paranoid_mode else 0.0007

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = (freqs >= f_low) & (freqs <= f_high)

    for ch in range(channels):
        S = librosa.stft(
            audio[:, ch], n_fft=n_fft, hop_length=hop, window="hann", center=True
        )
        mag = np.abs(S)
        phase = np.angle(S)

        if np.any(band):
            # Build a small, frame-dependent sinusoidal jitter for band bins
            frames = phase.shape[1]
            t = np.linspace(0, 1, frames)
            jitter_env = max_jitter * (
                0.5
                + 0.5 * np.sin(2 * np.pi * 0.37 * t + np.random.uniform(0, 2 * np.pi))
            )
            jitter = np.random.normal(0, 1, (np.sum(band), frames)) * jitter_env
            phase[band, :] += jitter

        modified = mag * np.exp(1j * phase)
        output[:, ch] = librosa.istft(modified, hop_length=hop, length=n)

    return output


def _apply_subblock_phase_dither(
    audio: np.ndarray, sr: int, paranoid_mode: bool
) -> np.ndarray:
    """
    Split into overlapping blocks and apply tiny phase dither per block to break stationarity.
    """
    block_ms = 220 if paranoid_mode else 260  # larger blocks to reduce audibility
    hop_ms = block_ms // 2
    block = int(sr * block_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if block < 256:
        block = 256
    if hop < 128:
        hop = 128

    n, channels = audio.shape
    output = np.zeros_like(audio)
    window = np.hanning(block)

    for ch in range(channels):
        accum = np.zeros(n)  # Reset accumulator per channel
        idx = 0
        while idx < n:
            end = min(idx + block, n)
            seg = audio[idx:end, ch]
            if len(seg) < block:
                seg = np.pad(seg, (0, block - len(seg)), mode="edge")

            spec = np.fft.rfft(seg * window)
            mags = np.abs(spec)
            phases = np.angle(spec)

            freqs = np.linspace(0, 1, len(phases))
            jitter = (0.0015 if paranoid_mode else 0.0010) * freqs + (
                0.0005 if paranoid_mode else 0.0003
            ) * (1 - freqs)
            phases = phases + np.random.normal(0, jitter)

            modified = np.fft.irfft(mags * np.exp(1j * phases), n=len(seg))
            output[idx:end, ch] += modified[: end - idx] * window[: end - idx]
            accum[idx:end] += window[: end - idx]

            idx += hop

    accum = np.where(accum == 0, 1, accum)
    return output / accum[:, None]


def _apply_dynamic_comb_mask(
    audio: np.ndarray, sr: int, paranoid_mode: bool
) -> np.ndarray:
    """
    Apply a very shallow moving comb notch to disturb fine spectral regularity.
    """
    n, channels = audio.shape
    t = np.arange(n) / sr
    # Move into upper band to avoid audible coloration
    base_freq = 15000 if paranoid_mode else 17000
    drift = 250 if paranoid_mode else 180
    depth = 0.008 if paranoid_mode else 0.006  # extremely shallow
    mix = 0.004 if paranoid_mode else 0.003

    delay = sr / (base_freq + drift * np.sin(2 * np.pi * 0.1 * t))  # samples
    out = np.zeros_like(audio)
    for ch in range(channels):
        delayed = np.zeros(n)
        # fractional delay via interpolation
        idxs = np.arange(n) - delay
        idxs = np.clip(idxs, 0, n - 2)
        idx0 = idxs.astype(int)
        frac = idxs - idx0
        delayed = (1 - frac) * audio[idx0, ch] + frac * audio[idx0 + 1, ch]
        out[:, ch] = audio[:, ch] * (1 - mix) + (audio[:, ch] - depth * delayed) * mix
    return out


def _apply_transient_micro_shift(
    audio: np.ndarray, sr: int, paranoid_mode: bool
) -> np.ndarray:
    """
    Detect transients and apply tiny sub-sample shifts to decorrelate onsets.
    """
    n, channels = audio.shape
    shifted = audio.copy()
    hop = 512
    win = 1024
    onset_env = librosa.onset.onset_strength(
        y=audio.mean(axis=1), sr=sr, hop_length=hop, n_fft=win
    )
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop, backtrack=True
    )
    onset_samples = librosa.frames_to_samples(onsets, hop_length=hop)

    max_shift = int(0.005 * sr) if paranoid_mode else int(0.003 * sr)  # ~5ms / ~3ms (human-scale)
    mix = 0.65 if paranoid_mode else 0.5
    for pos in onset_samples:
        start = max(0, pos - win)
        end = min(n, pos + win)
        region = np.arange(start, end)
        if len(region) < 4:
            continue
        for ch in range(channels):
            shift = np.random.randint(-max_shift, max_shift + 1)
            region_shifted = np.clip(region + shift, 0, n - 1)
            # crossfade blend to avoid clicks
            fade = np.linspace(0, 1, len(region))
            shifted[region, ch] = (1 - fade * mix) * shifted[region, ch] + (
                fade * mix
            ) * audio[region_shifted, ch]

    return shifted


def _apply_micro_eq_modulation(
    audio: np.ndarray, sr: int, paranoid_mode: bool
) -> np.ndarray:
    """
    Apply imperceptible, slow per-band gain flutter to break spectral stationarity.
    """
    n, channels = audio.shape
    t = np.linspace(0, 1, n)
    bands = [(80, 250), (250, 800), (800, 2500), (2500, 6500), (6500, 14000)]
    band_gains = np.zeros((len(bands), n))

    # Create tiny modulation envelopes
    for i, _ in enumerate(bands):
        depth = 0.0018 if paranoid_mode else 0.0012  # ~0.015 dB
        rate = np.random.uniform(0.1, 0.22)
        band_gains[i] = 1.0 + depth * np.sin(
            2 * np.pi * rate * t + np.random.uniform(0, 2 * np.pi)
        )

    # FFT-based filtering per channel (lightweight)
    win = 2048
    hop = win // 4
    for ch in range(channels):
        # STFT
        S = librosa.stft(
            audio[:, ch], n_fft=win, hop_length=hop, center=True, window="hann"
        )
        freqs = librosa.fft_frequencies(sr=sr, n_fft=win)
        for bi, (f_lo, f_hi) in enumerate(bands):
            mask = (freqs >= f_lo) & (freqs < f_hi)
            if not np.any(mask):
                continue
            # Broadcast envelope to time bins
            env = band_gains[bi, : S.shape[1]]
            S[mask, :] *= env
        # ISTFT back
        audio[:, ch] = librosa.istft(S, hop_length=hop, length=n)

    return audio


def _apply_gated_micro_eq_flutter(
    audio: np.ndarray, sr: int, paranoid_mode: bool
) -> np.ndarray:
    """
    Tiny band gain flutter gated by RMS to reduce audibility.
    """
    n, channels = audio.shape
    rms = np.sqrt(np.mean(audio**2, axis=1))
    thresh = np.percentile(rms, 60 if paranoid_mode else 70)

    bands = [(120, 300), (300, 900), (900, 3000), (3000, 7000), (7000, 12000)]
    depth = 0.0015 if paranoid_mode else 0.001  # ~0.013 dB
    n_fft = 2048
    hop = n_fft // 4
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    output = np.zeros_like(audio)
    for ch in range(channels):
        S = librosa.stft(
            audio[:, ch], n_fft=n_fft, hop_length=hop, center=True, window="hann"
        )
        mag = np.abs(S)
        phase = np.angle(S)

        frames = S.shape[1]
        frame_rms = (
            librosa.util.frame(rms, frame_length=hop, hop_length=hop).mean(axis=0)
            if len(rms) >= hop
            else np.array([np.mean(rms)])
        )
        frame_rms = np.pad(frame_rms, (0, max(0, frames - len(frame_rms))), mode="edge")

        t = np.linspace(0, 1, frames)
        for bi, (f_lo, f_hi) in enumerate(bands):
            mask = (freqs >= f_lo) & (freqs < f_hi)
            if not np.any(mask):
                continue
            rate = np.random.uniform(0.1, 0.25)
            env = 1.0 + depth * np.sin(
                2 * np.pi * rate * t + np.random.uniform(0, 2 * np.pi)
            )
            # Apply only where rms above threshold
            env = np.where(frame_rms > thresh, env, 1.0)
            mag[mask, :] *= env

        modified = mag * np.exp(1j * phase)
        output[:, ch] = librosa.istft(modified, hop_length=hop, length=n)

    return output


def _apply_refined_transient_shift(
    audio: np.ndarray, sr: int, paranoid_mode: bool
) -> np.ndarray:
    """
    Refined transient shift: ultra-small shifts gated by strong onsets.
    """
    n, channels = audio.shape
    shifted = audio.copy()

    hop = 512
    win = 1024
    onset_env = librosa.onset.onset_strength(
        y=audio.mean(axis=1), sr=sr, hop_length=hop, n_fft=win
    )
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop, backtrack=True, units="time"
    )
    onset_samples = (onsets * sr).astype(int)

    max_shift = int(0.004 * sr) if paranoid_mode else int(0.002 * sr)  # ~4ms / ~2ms (human-scale)
    mix = 0.55 if paranoid_mode else 0.4
    for pos in onset_samples:
        start = max(0, pos - win)
        end = min(n, pos + win)
        region = np.arange(start, end)
        if len(region) < 4:
            continue
        for ch in range(channels):
            shift = np.random.randint(-max_shift, max_shift + 1)
            region_shifted = np.clip(region + shift, 0, n - 1)
            fade = np.linspace(0, 1, len(region))
            shifted[region, ch] = (1 - fade * mix) * shifted[region, ch] + (
                fade * mix
            ) * audio[region_shifted, ch]

    return shifted


def _apply_adaptive_transient_shift(
    audio: np.ndarray, sr: int, paranoid_mode: bool
) -> np.ndarray:
    """
    Adaptive transient shift: ultra-small shifts gated by onset strength with adaptive depth.
    """
    n, channels = audio.shape
    shifted = audio.copy()

    hop = 512
    win = 1024
    onset_env = librosa.onset.onset_strength(
        y=audio.mean(axis=1), sr=sr, hop_length=hop, n_fft=win
    )
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop, backtrack=True, units="time"
    )
    onset_samples = (onsets * sr).astype(int)

    # Normalize onset strength to [0,1]
    if len(onset_env):
        env_range = np.ptp(onset_env)
        env_norm = (onset_env - onset_env.min()) / (env_range + 1e-9)
    else:
        env_norm = np.array([])

    base_shift = 0.002 if paranoid_mode else 0.0015  # ~2ms / ~1.5ms base (human-scale)
    max_shift = 0.008 if paranoid_mode else 0.005  # ~8ms / ~5ms max

    for idx, pos in enumerate(onset_samples):
        strength = env_norm[idx] if idx < len(env_norm) else 0.0
        shift_samples = int(sr * (base_shift + strength * (max_shift - base_shift)))
        if shift_samples < 1:
            continue

        start = max(0, pos - win)
        end = min(n, pos + win)
        region = np.arange(start, end)
        if len(region) < 4:
            continue

        # Alternate direction to avoid bias
        direction = -1 if (idx % 2 == 0) else 1
        actual_shift = direction * shift_samples

        for ch in range(channels):
            region_shifted = np.clip(region + actual_shift, 0, n - 1)
            fade = np.linspace(0, 1, len(region))
            mix = 0.55 if paranoid_mode else 0.4
            shifted[region, ch] = (1 - fade * mix) * shifted[region, ch] + (
                fade * mix
            ) * audio[region_shifted, ch]

    return shifted


def _apply_spectral_watermark_cleaning(
    audio: np.ndarray, sr: int, paranoid_mode: bool
) -> np.ndarray:
    """
    Delegate to SpectralCleaner to remove known watermark frequency bands,
    periodic patterns, and synchronization tones.
    """
    from mmm.sanitization.spectral_cleaner import SpectralCleaner

    n, channels = audio.shape
    if n == 0:
        return audio

    cleaner = SpectralCleaner(paranoid_mode=paranoid_mode)

    # SpectralCleaner handles stereo internally via channel loop
    result = cleaner.clean_watermarks(audio, sr)
    cleaned = result["cleaned_audio"]

    # Ensure shape consistency (SpectralCleaner may return 1D for mono)
    if cleaned.ndim == 1:
        cleaned = cleaned.reshape(-1, 1)
    # Trim or pad if length changed
    if cleaned.shape[0] > n:
        cleaned = cleaned[:n]
    elif cleaned.shape[0] < n:
        pad_len = n - cleaned.shape[0]
        cleaned = np.pad(cleaned, ((0, pad_len), (0, 0)), mode="edge")

    return cleaned


def _apply_fingerprint_removal(
    audio: np.ndarray, sr: int, paranoid_mode: bool
) -> np.ndarray:
    """
    Delegate to FingerprintRemover for statistical normalization,
    temporal randomization, phase randomization, micro-timing perturbation,
    and human imperfections.
    """
    from mmm.sanitization.fingerprint_remover import FingerprintRemover

    n, channels = audio.shape
    if n == 0:
        return audio

    remover = FingerprintRemover(paranoid_mode=paranoid_mode)

    result = remover.remove_fingerprints(audio, sr)
    cleaned = result["cleaned_audio"]

    if cleaned.ndim == 1:
        cleaned = cleaned.reshape(-1, 1)
    if cleaned.shape[0] > n:
        cleaned = cleaned[:n]
    elif cleaned.shape[0] < n:
        pad_len = n - cleaned.shape[0]
        cleaned = np.pad(cleaned, ((0, pad_len), (0, 0)), mode="edge")

    return cleaned


def _apply_onset_velocity_variation(
    audio: np.ndarray, sr: int, paranoid_mode: bool
) -> np.ndarray:
    """
    Vary onset amplitude to break AI-consistent attack energy.
    Human performance naturally varies onset velocity by +-10-25%.
    Uses AR(1) correlated noise for natural feel.
    """
    n, channels = audio.shape
    if n < sr:
        return audio

    output = audio.copy()

    hop = 512
    win = 1024
    onset_env = librosa.onset.onset_strength(
        y=audio.mean(axis=1), sr=sr, hop_length=hop, n_fft=win
    )
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop, backtrack=True
    )
    onset_samples = librosa.frames_to_samples(onsets, hop_length=hop)

    if len(onset_samples) == 0:
        return audio

    variation_range = 0.15 if paranoid_mode else 0.08
    attack_ms = 25 if paranoid_mode else 20
    attack_samples = int(attack_ms * sr / 1000)

    # AR(1) correlated gains for natural performance feel
    gains = np.zeros(len(onset_samples))
    gains[0] = np.random.uniform(-variation_range, variation_range)
    for i in range(1, len(gains)):
        gains[i] = 0.3 * gains[i - 1] + 0.7 * np.random.uniform(
            -variation_range, variation_range
        )

    for idx, pos in enumerate(onset_samples):
        if pos >= n:
            continue
        gain = 1.0 + gains[idx]

        region_end = min(n, pos + attack_samples)
        region_len = region_end - pos
        if region_len < 4:
            continue

        # Smooth envelope: ramp from 1.0 to gain, then back to 1.0
        env = np.ones(region_len)
        half = region_len // 2
        env[:half] = np.linspace(1.0, gain, half)
        env[half:] = np.linspace(gain, 1.0, region_len - half)

        for ch in range(channels):
            output[pos:region_end, ch] *= env

    return output


def _apply_long_range_tempo_drift(
    audio: np.ndarray, sr: int, paranoid_mode: bool
) -> np.ndarray:
    """
    Apply slow, smooth tempo drift to break AI-perfect BPM consistency.
    Human performances drift +-1-3 BPM over minutes. AI generators produce
    mathematically constant tempo -- the strongest temporal detection signal
    for long tracks.
    """
    n, channels = audio.shape
    if n < sr * 5:
        return audio

    duration = n / sr
    t = np.linspace(0, duration, n)

    max_drift_frac = 0.015 if paranoid_mode else 0.008

    # Sum of slow sinusoids with random phases for natural feel
    drift = np.zeros(n)
    for _ in range(4):
        period = (
            np.random.uniform(15, 35) if paranoid_mode else np.random.uniform(20, 45)
        )
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = max_drift_frac * np.random.uniform(0.3, 1.0)
        drift += amplitude * np.sin(2 * np.pi * t / period + phase)

    # Add low-frequency Brownian component
    brown = np.cumsum(np.random.normal(0, 1, n))
    cutoff = 0.05 / (sr / 2)
    if 0 < cutoff < 1.0:
        b, a = butter(2, cutoff, btype="low")
        brown = filtfilt(b, a, brown)
    max_brown = np.max(np.abs(brown))
    if max_brown > 0:
        brown = brown / max_brown * max_drift_frac * 0.3
    drift += brown

    # Normalize drift and apply fade-in/fade-out to avoid endpoint artifacts
    max_drift = np.max(np.abs(drift))
    if max_drift > 0:
        drift = drift / max_drift * max_drift_frac

    fade_len = min(int(sr * 2), n // 4)
    if fade_len > 0:
        drift[:fade_len] *= np.linspace(0, 1, fade_len)
        drift[-fade_len:] *= np.linspace(1, 0, fade_len)

    # Convert drift curve to monotonic warp map
    rate = 1.0 + drift
    cumulative = np.cumsum(rate)
    warp_map = cumulative / cumulative[-1] * (n - 1)

    # Interpolate audio along warp map
    output = np.zeros_like(audio)
    original_indices = np.arange(n).astype(np.float64)
    for ch in range(channels):
        output[:, ch] = np.interp(warp_map, original_indices, audio[:, ch])

    return output


def _apply_mfcc_perturbation(
    audio: np.ndarray, sr: int, paranoid_mode: bool
) -> np.ndarray:
    """
    Perturb MFCC coefficients to defeat MFCC-based AI classifiers.
    Modifies c1-c6 (the most classification-relevant coefficients),
    reconstructs via inverse mel, and blends with original at low ratio.
    """
    n, channels = audio.shape
    if n < sr:
        return audio

    blend = 0.25 if paranoid_mode else 0.15
    n_fft = 2048
    hop_length = 512
    n_mfcc = 13
    n_mels = 128

    # Per-coefficient perturbation strength (skip c0 = energy)
    coeff_std = np.zeros(n_mfcc)
    if paranoid_mode:
        coeff_std[1:5] = 1.0
        coeff_std[5:7] = 0.6
        coeff_std[7:] = 0.2
    else:
        coeff_std[1:5] = 0.5
        coeff_std[5:7] = 0.3
        coeff_std[7:] = 0.1

    output = audio.copy()

    for ch in range(channels):
        try:
            # Forward: compute mel spectrogram and MFCCs
            source = np.nan_to_num(audio[:, ch], nan=0.0, posinf=0.0, neginf=0.0)
            S = librosa.feature.melspectrogram(
                y=source, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
            )
            S = np.maximum(np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0), 1e-10)
            S_db = librosa.power_to_db(S)
            S_db = np.nan_to_num(S_db, nan=-80.0, posinf=0.0, neginf=-80.0)
            mfccs = librosa.feature.mfcc(S=S_db, n_mfcc=n_mfcc)

            n_frames = mfccs.shape[1]

            # Generate smooth time-varying perturbation per coefficient
            for c in range(1, n_mfcc):
                if coeff_std[c] == 0:
                    continue
                noise = np.random.normal(0, coeff_std[c], n_frames)
                smooth_win = max(3, int(0.5 * sr / hop_length))
                if smooth_win % 2 == 0:
                    smooth_win += 1
                kernel = np.ones(smooth_win) / smooth_win
                noise = np.convolve(noise, kernel, mode="same")
                mfccs[c, :] += noise

            # Inverse: MFCC -> mel spectrogram -> audio
            S_db_modified = librosa.feature.inverse.mfcc_to_mel(mfccs, n_mels=n_mels)
            S_db_modified = np.nan_to_num(
                S_db_modified, nan=-80.0, posinf=0.0, neginf=-80.0
            )
            # Bound dB to keep reconstruction numerically stable
            S_db_modified = np.clip(
                S_db_modified, -100.0, 18.0 if paranoid_mode else 12.0
            )

            S_modified = librosa.db_to_power(S_db_modified)
            S_modified = np.maximum(
                np.nan_to_num(S_modified, nan=0.0, posinf=0.0, neginf=0.0), 0.0
            )

            src_energy = float(np.mean(S))
            mod_energy = float(np.mean(S_modified))
            if src_energy > 0 and mod_energy > 0 and np.isfinite(src_energy + mod_energy):
                S_modified = S_modified * (src_energy / mod_energy)

            # Cap extreme mel bins to prevent unstable Griffin-Lim iterations
            cap = np.percentile(S_modified, 99.9)
            if np.isfinite(cap) and cap > 0:
                S_modified = np.clip(S_modified, 0.0, cap * 6.0)

            reconstructed = librosa.feature.inverse.mel_to_audio(
                S_modified,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                length=n,
                n_iter=16 if not paranoid_mode else 24,
            )
            reconstructed = np.nan_to_num(
                reconstructed, nan=0.0, posinf=0.0, neginf=0.0
            )
            if not np.all(np.isfinite(reconstructed)):
                raise ValueError("mel reconstruction produced non-finite output")

            src_rms = np.sqrt(np.mean(source**2))
            rec_rms = np.sqrt(np.mean(reconstructed**2))
            if src_rms > 0 and rec_rms > 0:
                reconstructed = reconstructed * (src_rms / rec_rms)

            blended = (1 - blend) * source + blend * reconstructed
            output[:, ch] = np.nan_to_num(blended, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            logger.warning(
                "MFCC perturbation failed on channel %d; keeping original channel (%s)",
                ch,
                e,
            )
            output[:, ch] = audio[:, ch]

    return output


def main():
    """Main function for testing"""
    input_file = Path("before.mp3")
    if not input_file.exists():
        input_file = Path("Schizo Shaman.mp3")

    if not input_file.exists():
        print("❌ Error: No test file found")
        return

    result = preserving_sanitize(input_file, paranoid_mode=False, threat_count=317)

    if result["success"]:
        print(f"\n✨ Preserving sanitization complete!")
        print(f"   Audio quality: PRESERVED")
        print(f"   Effectiveness: {result['stats']['effectiveness']:.1f}%")
    else:
        print(
            f"\n💥 Preserving sanitization failed: {result.get('error', 'Unknown error')}"
        )


if __name__ == "__main__":
    main()
