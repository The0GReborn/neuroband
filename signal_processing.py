"""
signal_processing.py — EEG Signal Preprocessing for NeuroBand
Place in: /home/pi/neuroband/signal_processing.py

Pipeline per window:
  1. DC offset removal  (high-pass effect via mean subtraction)
  2. Notch filter       (50 Hz mains noise)
  3. Bandpass filter    (0.5 – 40 Hz, 4th-order Butterworth)
  4. Artifact rejection (amplitude threshold)
  5. Z-score normalisation
"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, iirnotch, tf2sos
import logging

import config

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Filter design (computed once at import time)
# ══════════════════════════════════════════════════════════════════════════════
def _design_bandpass(low: float, high: float, fs: float, order: int = 4):
    """Return second-order sections for a Butterworth bandpass filter."""
    nyq = fs / 2.0
    return butter(order, [low / nyq, high / nyq], btype="band", output="sos")

def _design_notch(freq: float, fs: float, quality: float = 30.0):
    """Return second-order sections for an IIR notch filter."""
    b, a = iirnotch(freq / (fs / 2.0), quality)
    return tf2sos(b, a)

_sos_bandpass = _design_bandpass(
    config.BANDPASS_LOW, config.BANDPASS_HIGH, config.SAMPLE_RATE, config.FILTER_ORDER
)
_sos_notch = _design_notch(config.NOTCH_FREQ, config.SAMPLE_RATE)

# Initialise filter state vectors (one set per channel, maintained across calls)
# Shape: (n_sections, 2)  — carries memory between consecutive windows
_bp_zi:   list[np.ndarray] = [sosfilt_zi(_sos_bandpass) for _ in range(config.NUM_CHANNELS)]
_notch_zi: list[np.ndarray] = [sosfilt_zi(_sos_notch)   for _ in range(config.NUM_CHANNELS)]

# Amplitude threshold for artifact rejection (in volts; ±200 µV typical EEG limit)
_ARTIFACT_THRESHOLD = 200e-6


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════
class SignalProcessor:
    """
    Stateful processor — maintains IIR filter memories across successive
    windows so there is no edge discontinuity between consecutive buffers.

    Usage:
        proc = SignalProcessor()
        clean = proc.process(raw_window)   # raw_window: list[list[float]]
                                            # returns:    np.ndarray shape (C, N)
    """

    def __init__(self):
        # Per-channel IIR state (persist across windows → no boundary artefacts)
        self._bp_zi    = [sosfilt_zi(_sos_bandpass) for _ in range(config.NUM_CHANNELS)]
        self._notch_zi = [sosfilt_zi(_sos_notch)    for _ in range(config.NUM_CHANNELS)]
        log.info("SignalProcessor ready — bandpass %.1f–%.1f Hz, notch %.0f Hz",
                 config.BANDPASS_LOW, config.BANDPASS_HIGH, config.NOTCH_FREQ)

    # ── Core pipeline ──────────────────────────────────────────────────────────
    def process(self, raw_window: list[list[float]]) -> np.ndarray | None:
        """
        Process a multi-channel EEG window.

        Parameters
        ----------
        raw_window : list of lists, shape (NUM_CHANNELS, WINDOW_SAMPLES)

        Returns
        -------
        np.ndarray of shape (NUM_CHANNELS, WINDOW_SAMPLES), or None if
        the window is rejected due to motion/muscle artifact.
        """
        log.debug("SignalProcessor.process(): starting pipeline, window shape=(%d, %d)",
                  len(raw_window), len(raw_window[0]) if raw_window else 0)
        data = np.array(raw_window, dtype=np.float64)   # (C, N)

        log.debug("SignalProcessor: step 1 — removing DC offset")
        data = self._remove_dc(data)

        log.debug("SignalProcessor: step 2 — applying notch filter (%.0f Hz)", config.NOTCH_FREQ)
        data = self._apply_notch(data)

        log.debug("SignalProcessor: step 3 — applying bandpass filter (%.1f–%.1f Hz)",
                  config.BANDPASS_LOW, config.BANDPASS_HIGH)
        data = self._apply_bandpass(data)

        log.debug("SignalProcessor: step 4 — artifact rejection check")
        if self._is_artifact(data):
            log.debug("Window rejected: amplitude exceeds artifact threshold.")
            return None

        log.debug("SignalProcessor: step 5 — z-score normalising")
        data = self._normalise(data)
        log.debug("SignalProcessor.process(): pipeline complete, output shape=%s", data.shape)
        return data

    # ── Step 1 — DC removal ────────────────────────────────────────────────────
    @staticmethod
    def _remove_dc(data: np.ndarray) -> np.ndarray:
        """Subtract per-channel mean to eliminate DC offset."""
        return data - data.mean(axis=1, keepdims=True)

    # ── Step 2 — Notch filter (50 Hz) ─────────────────────────────────────────
    def _apply_notch(self, data: np.ndarray) -> np.ndarray:
        out = np.empty_like(data)
        for ch in range(config.NUM_CHANNELS):
            filtered, self._notch_zi[ch] = sosfilt(
                _sos_notch, data[ch], zi=self._notch_zi[ch]
            )
            out[ch] = filtered
        return out

    # ── Step 3 — Bandpass filter (0.5 – 40 Hz) ────────────────────────────────
    def _apply_bandpass(self, data: np.ndarray) -> np.ndarray:
        out = np.empty_like(data)
        for ch in range(config.NUM_CHANNELS):
            filtered, self._bp_zi[ch] = sosfilt(
                _sos_bandpass, data[ch], zi=self._bp_zi[ch]
            )
            out[ch] = filtered
        return out

    # ── Step 4 — Artifact rejection ────────────────────────────────────────────
    @staticmethod
    def _is_artifact(data: np.ndarray) -> bool:
        """Return True if any sample exceeds the amplitude threshold."""
        peak = float(np.max(np.abs(data)))
        result = peak > _ARTIFACT_THRESHOLD
        if result:
            log.debug("_is_artifact: REJECTED — peak=%.2fuV > threshold=%.0fuV",
                      peak * 1e6, _ARTIFACT_THRESHOLD * 1e6)
        else:
            log.debug("_is_artifact: OK — peak=%.2fuV", peak * 1e6)
        return result

    # ── Step 5 — Z-score normalisation ────────────────────────────────────────
    @staticmethod
    def _normalise(data: np.ndarray) -> np.ndarray:
        """
        Z-score per channel: (x - mean) / std
        Clips to ±5σ to reduce residual outlier influence.
        """
        mean = data.mean(axis=1, keepdims=True)
        std  = data.std(axis=1, keepdims=True) + 1e-12   # avoid div-by-zero
        log.debug("_normalise: ch0 mean=%.4f std=%.4f | ch1 mean=%.4f std=%.4f",
                  float(mean[0]), float(std[0]), float(mean[1]), float(std[1]))
        z    = (data - mean) / std
        return np.clip(z, -5.0, 5.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Utility: band power helper (also used by feature_extraction.py)
# ══════════════════════════════════════════════════════════════════════════════
def band_power(signal: np.ndarray, fs: float, low: float, high: float) -> float:
    """
    Compute average power in a frequency band using Welch's method.

    Parameters
    ----------
    signal : 1-D np.ndarray (single channel)
    fs     : sampling frequency in Hz
    low, high : band limits in Hz

    Returns
    -------
    float — mean power spectral density in the band (V²/Hz)
    """
    from scipy.signal import welch
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), 256))
    idx = np.logical_and(freqs >= low, freqs <= high)
    if not np.any(idx):
        return 0.0
    return float(np.trapz(psd[idx], freqs[idx]))