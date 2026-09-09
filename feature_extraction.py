"""
feature_extraction.py — EEG Feature Extraction for NeuroBand
Place in: /home/pi/neuroband/feature_extraction.py

Features extracted per window (per channel):
  1.  Peak-to-peak amplitude
  2.  RMS amplitude
  3.  Variance
  4.  Skewness
  5.  Kurtosis
  6.  Delta power   (0.5 – 4 Hz)
  7.  Theta power   (4 – 8 Hz)
  8.  Alpha power   (8 – 12 Hz)
  9.  Beta power    (12 – 30 Hz)
  10. Gamma power   (30 – 40 Hz)
  11. Alpha/Beta ratio        ← key EEG biomarker
  12. Spectral edge frequency (95% of power below X Hz)

For 2 channels → 12 × 2 = 24 features total.
Additional cross-channel feature:
  13. Alpha asymmetry (F4_alpha − F3_alpha) / (F4_alpha + F3_alpha)
      → lateralisation indicator for YES/NO discrimination

Total feature vector size: 25
"""

import numpy as np
from scipy.stats import skew, kurtosis as scipy_kurtosis
import logging

import config
from signal_processing import band_power

log = logging.getLogger(__name__)

# Feature names (for logging / model inspection)
_PER_CHANNEL_NAMES = [
    "p2p_amp", "rms", "variance", "skewness", "kurtosis",
    "delta_pow", "theta_pow", "alpha_pow", "beta_pow", "gamma_pow",
    "alpha_beta_ratio", "spectral_edge_95",
]
FEATURE_NAMES = (
    [f"ch0_{n}" for n in _PER_CHANNEL_NAMES]
    + [f"ch1_{n}" for n in _PER_CHANNEL_NAMES]
    + ["alpha_asymmetry"]
)
N_FEATURES = len(FEATURE_NAMES)   # 25


class FeatureExtractor:
    """
    Converts a preprocessed EEG window into a 1-D feature vector.

    Usage:
        extractor = FeatureExtractor()
        fv = extractor.extract(clean_window)   # clean_window: np.ndarray (C, N)
        # fv is np.ndarray of shape (N_FEATURES,) or None on failure
    """

    def extract(self, data: np.ndarray) -> np.ndarray | None:
        """
        Parameters
        ----------
        data : np.ndarray, shape (NUM_CHANNELS, WINDOW_SAMPLES)
               Pre-processed (filtered, normalised) EEG window.

        Returns
        -------
        np.ndarray of shape (N_FEATURES,), or None if extraction fails.
        """
        log.debug("FeatureExtractor.extract(): input shape=%s", data.shape if data is not None else "None")
        if data is None or data.shape[1] < 64:
            log.debug("Feature extraction skipped: window too short.")
            return None

        try:
            log.debug("FeatureExtractor: extracting per-channel features for %d channels", config.NUM_CHANNELS)
            ch_features = [self._channel_features(data[ch]) for ch in range(config.NUM_CHANNELS)]
            log.debug("FeatureExtractor: extracting cross-channel features (alpha asymmetry)")
            cross = self._cross_channel_features(data)
            fv = np.concatenate(ch_features + [cross])
            log.debug("FeatureExtractor.extract(): done — feature vector length=%d", len(fv))
            return fv.astype(np.float32)
        except Exception as exc:
            log.error("Feature extraction error: %s", exc)
            return None

    # ── Per-channel features ───────────────────────────────────────────────────
    def _channel_features(self, sig: np.ndarray) -> np.ndarray:
        log.debug("_channel_features(): computing time-domain stats, len=%d", len(sig))
        fs = config.SAMPLE_RATE

        # Time-domain
        p2p      = float(sig.max() - sig.min())
        rms      = float(np.sqrt(np.mean(sig ** 2)))
        var      = float(np.var(sig))
        sk       = float(skew(sig))
        kurt     = float(scipy_kurtosis(sig))
        log.debug("_channel_features(): p2p=%.4f rms=%.4f var=%.6f skew=%.4f kurt=%.4f",
                  p2p, rms, var, sk, kurt)

        # Frequency-domain (band powers)
        log.debug("_channel_features(): computing band powers (delta/theta/alpha/beta/gamma)")
        d_pow = band_power(sig, fs, *config.BAND_DELTA)
        t_pow = band_power(sig, fs, *config.BAND_THETA)
        a_pow = band_power(sig, fs, *config.BAND_ALPHA)
        b_pow = band_power(sig, fs, *config.BAND_BETA)
        g_pow = band_power(sig, fs, *config.BAND_GAMMA)
        log.debug("_channel_features(): delta=%.4e theta=%.4e alpha=%.4e beta=%.4e gamma=%.4e",
                  d_pow, t_pow, a_pow, b_pow, g_pow)

        # Alpha / Beta ratio (high ratio = relaxed, low = engaged)
        ab_ratio = a_pow / (b_pow + 1e-30)
        log.debug("_channel_features(): alpha/beta ratio=%.4f", ab_ratio)

        # Spectral edge frequency at 95% cumulative power
        log.debug("_channel_features(): computing spectral edge (95%%)")
        se95 = self._spectral_edge(sig, fs, edge=0.95)
        log.debug("_channel_features(): spectral_edge_95=%.2f Hz", se95)

        return np.array([p2p, rms, var, sk, kurt,
                         d_pow, t_pow, a_pow, b_pow, g_pow,
                         ab_ratio, se95], dtype=np.float64)

    # ── Cross-channel features ─────────────────────────────────────────────────
    def _cross_channel_features(self, data: np.ndarray) -> np.ndarray:
        log.debug("_cross_channel_features(): computing alpha asymmetry")
        fs = config.SAMPLE_RATE
        a0 = band_power(data[0], fs, *config.BAND_ALPHA)   # F3 alpha
        a1 = band_power(data[1], fs, *config.BAND_ALPHA)   # F4 alpha

        # Alpha asymmetry: positive → right-dominant (approach motivation)
        #                  negative → left-dominant  (withdrawal)
        asym = (a1 - a0) / (a1 + a0 + 1e-30)
        log.debug("_cross_channel_features(): F3_alpha=%.4e F4_alpha=%.4e asymmetry=%.4f",
                  a0, a1, asym)
        return np.array([asym], dtype=np.float64)

    # ── Helpers ────────────────────────────────────────────────────────────────
    @staticmethod
    def _spectral_edge(sig: np.ndarray, fs: float, edge: float = 0.95) -> float:
        """Return frequency below which `edge` fraction of total power resides."""
        from scipy.signal import welch
        freqs, psd = welch(sig, fs=fs, nperseg=min(len(sig), 256))
        cumpower   = np.cumsum(psd)
        threshold  = edge * cumpower[-1]
        idx        = np.searchsorted(cumpower, threshold)
        return float(freqs[min(idx, len(freqs) - 1)])