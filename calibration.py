"""
calibration.py -- Auto-Calibration for NeuroBand
Place in: /home/pi/neuroband/calibration.py

What it does:
  - Records 30 seconds of resting EEG on startup
  - Computes your personal baseline feature statistics (mean, std per feature)
  - Saves baseline to models/baseline.pkl
  - Provides a transform() method to normalise live features against baseline

Why this matters:
  - Your brain's alpha/beta levels shift daily (tired, stressed, caffeinated)
  - A model trained yesterday may misclassify today without recalibration
  - Baseline normalisation centres features around YOUR current brain state
  - This is standard practice in clinical BCI systems

Usage in main.py:
    cal = Calibrator()
    cal.run(reader, processor, extractor)   # blocking, 30 seconds
    # then during inference:
    fv_calibrated = cal.transform(fv)
    label, conf = classifier.predict(fv_calibrated)
"""

import os
import time
import pickle
import logging
import numpy as np

import config
from signal_processing  import SignalProcessor
from feature_extraction import FeatureExtractor

log = logging.getLogger(__name__)


class Calibrator:
    """
    Captures a resting baseline and normalises future feature vectors against it.

    Baseline stored as:
      baseline_mean : np.ndarray (N_FEATURES,)
      baseline_std  : np.ndarray (N_FEATURES,)

    Transform: fv_calibrated = (fv - baseline_mean) / (baseline_std + eps)
    """

    def __init__(self):
        self._mean: np.ndarray | None = None
        self._std:  np.ndarray | None = None
        self._loaded = False
        self._load()

    # -- Persistence ------------------------------------------------------------
    def _load(self):
        log.debug("Calibrator._load(): checking for baseline at '%s'", config.CALIBRATION_PATH)
        if os.path.exists(config.CALIBRATION_PATH):
            try:
                with open(config.CALIBRATION_PATH, "rb") as f:
                    data = pickle.load(f)
                self._mean = data["mean"]
                self._std  = data["std"]
                self._loaded = True
                log.info("Baseline loaded from %s", config.CALIBRATION_PATH)
                log.debug("Calibrator._load(): mean shape=%s std shape=%s",
                          self._mean.shape, self._std.shape)
            except Exception as exc:
                log.warning("Could not load baseline: %s", exc)
        else:
            log.debug("Calibrator._load(): no baseline file found — will need calibration run")

    def _save(self):
        log.debug("Calibrator._save(): saving baseline to '%s'", config.CALIBRATION_PATH)
        os.makedirs(os.path.dirname(config.CALIBRATION_PATH), exist_ok=True)
        with open(config.CALIBRATION_PATH, "wb") as f:
            pickle.dump({"mean": self._mean, "std": self._std}, f)
        log.info("Baseline saved to %s", config.CALIBRATION_PATH)

    # -- Calibration run --------------------------------------------------------
    def run(self, reader, processor: SignalProcessor, extractor: FeatureExtractor):
        """
        Collect resting EEG for CALIBRATION_DURATION seconds and compute baseline.
        reader must already be started before calling this.
        """
        duration = config.CALIBRATION_DURATION
        log.debug("Calibrator.run(): starting — duration=%ds", duration)
        print("\n" + "="*55)
        print("  AUTO-CALIBRATION")
        print("="*55)
        print(f"  Sit still, relax, and close your eyes.")
        print(f"  Recording {duration} seconds of baseline EEG...")
        print("="*55)

        # Countdown
        for i in range(3, 0, -1):
            print(f"  Starting in {i}...", flush=True)
            time.sleep(1)
        print("  GO -- stay relaxed!\n", flush=True)

        features_collected: list[np.ndarray] = []
        deadline = time.time() + duration
        last_sample = None
        cal_step = config.SAMPLE_RATE // 4   # 62 samples = ~0.25s stride
        log.debug("Calibrator.run(): collecting windows with step=%d samples", cal_step)

        while time.time() < deadline:
            avail = reader.samples_available()

            if avail < config.WINDOW_SAMPLES:
                time.sleep(0.01)
                continue

            if last_sample is None:
                last_sample = avail
            elif avail - last_sample < cal_step:
                time.sleep(0.01)
                continue
            else:
                last_sample = avail

            raw   = reader.get_latest(config.WINDOW_SAMPLES)
            log.debug("Calibrator.run(): processing raw window #%d", len(features_collected) + 1)
            clean = processor.process(raw)
            if clean is None:
                log.debug("Calibrator.run(): window rejected by SignalProcessor (artifact)")
                continue

            fv = extractor.extract(clean)
            if fv is not None:
                features_collected.append(fv)
                log.debug("Calibrator.run(): window accepted, total=%d", len(features_collected))

            remaining = int(deadline - time.time())
            print(f"\r  Calibrating... {remaining:>2}s remaining  "
                  f"({len(features_collected)} windows)", end="", flush=True)

        print("\n")
        log.debug("Calibrator.run(): collection done — %d windows total", len(features_collected))

        if len(features_collected) < 3:
            log.warning("Calibration collected only %d windows — too few, skipping.",
                        len(features_collected))
            print("  WARNING: Not enough clean windows. Calibration skipped.")
            print("  Check electrode contact and try again.\n")
            return

        X = np.array(features_collected)          # (n_windows, N_FEATURES)
        self._mean = X.mean(axis=0)
        self._std  = X.std(axis=0)
        self._loaded = True
        log.debug("Calibrator.run(): baseline computed — mean min=%.4f max=%.4f",
                  float(self._mean.min()), float(self._mean.max()))
        self._save()

        print(f"  Calibration complete! ({len(features_collected)} windows captured)")
        print(f"  Baseline saved to {config.CALIBRATION_PATH}")
        print("="*55 + "\n")
        log.info("Calibration complete: %d windows, %d features",
                 len(features_collected), X.shape[1])

    # -- Feature transform ------------------------------------------------------
    def transform(self, fv: np.ndarray) -> np.ndarray:
        """
        Normalise a feature vector against the captured baseline.
        If no baseline is loaded, returns fv unchanged.
        """
        if not self._loaded or self._mean is None:
            log.debug("Calibrator.transform(): no baseline loaded — returning raw features")
            return fv

        eps = 1e-12
        fv_cal = (fv - self._mean) / (self._std + eps)
        log.debug("Calibrator.transform(): applied baseline normalisation")
        return fv_cal

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def status_line(self) -> str:
        if self._loaded:
            return "Baseline: LOADED"
        return "Baseline: NOT CALIBRATED (using raw features)"