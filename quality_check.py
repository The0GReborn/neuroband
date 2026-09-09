"""
quality_check.py -- EEG Signal Quality Monitor for NeuroBand
Place in: /home/pi/neuroband/quality_check.py

Checks performed per channel:
  1. FLAT CHECK    -- amplitude too low = electrode not making contact
  2. NOISE CHECK   -- amplitude too high = muscle artifact / bad ground
  3. BAND CHECK    -- too much power outside EEG range = interference
  4. NOTCH CHECK   -- strong 50 Hz component = poor ground electrode

Rating per channel:
  GOOD  -- all checks pass, safe to classify
  WARN  -- minor issues, proceed with caution
  FAIL  -- serious problem, electrode check required

Usage:
    checker = QualityChecker()
    report  = checker.check(reader)
    report.print()
    if report.is_ok:
        run_inference()
"""

import time
import logging
import numpy as np
from dataclasses import dataclass

import config
from signal_processing import band_power

log = logging.getLogger(__name__)

# Rating constants
GOOD = "GOOD"
WARN = "WARN"
FAIL = "FAIL"


@dataclass
class ChannelReport:
    channel:    int
    name:       str         # "F3-Cz" or "F4-Cz"
    rating:     str         # GOOD / WARN / FAIL
    amplitude:  float       # RMS amplitude in uV
    noise_ratio: float      # out-of-band / in-band power ratio
    notch_power: float      # power at 50 Hz in uV^2
    issues:     list[str]   # human-readable issue descriptions


@dataclass
class QualityReport:
    channels: list[ChannelReport]
    overall:  str           # GOOD / WARN / FAIL

    @property
    def is_ok(self) -> bool:
        """Returns True if safe to proceed (GOOD or WARN)."""
        return self.overall != FAIL

    def print(self):
        """Print a formatted quality report to the terminal."""
        print("\n" + "="*55)
        print("  SIGNAL QUALITY REPORT")
        print("="*55)

        icons = {GOOD: "[GOOD]", WARN: "[WARN]", FAIL: "[FAIL]"}

        for ch in self.channels:
            icon = icons[ch.rating]
            print(f"  {icon}  CH{ch.channel} ({ch.name})")
            print(f"         Amplitude : {ch.amplitude*1e6:6.2f} uV RMS")
            print(f"         Noise     : {ch.noise_ratio*100:5.1f}% out-of-band")
            print(f"         50Hz power: {ch.notch_power*1e12:.2f} pV^2")
            if ch.issues:
                for issue in ch.issues:
                    print(f"         ! {issue}")
            else:
                print(f"         All checks passed.")

        print("-"*55)
        overall_icon = icons[self.overall]
        print(f"  OVERALL: {overall_icon}")

        if self.overall == GOOD:
            print("  Electrode contact is good. Ready to record.")
        elif self.overall == WARN:
            print("  Minor issues detected. Results may be less accurate.")
            print("  Tip: Press electrodes firmly, reduce movement.")
        else:
            print("  Poor signal quality. Do NOT proceed with classification.")
            print("  Action required:")
            print("    - Check electrode gel / contact")
            print("    - Verify ground electrode (forehead Fpz)")
            print("    - Verify reference electrode (earlobe)")
            print("    - Reduce muscle tension in face/jaw")

        print("="*55 + "\n")


class QualityChecker:
    """
    Samples EEG for QUALITY_CHECK_DURATION seconds and evaluates signal quality.

    Usage:
        checker = QualityChecker()
        report  = checker.check(reader)   # reader must be started
        report.print()
    """

    CHANNEL_NAMES = ["F3-Cz", "F4-Cz"]

    def check(self, reader) -> QualityReport:
        """
        Collect raw samples and evaluate quality.
        """
        log.debug("QualityChecker.check(): starting — duration=%ds", config.QUALITY_CHECK_DURATION)
        print("\n  Checking signal quality...", flush=True)
        duration = config.QUALITY_CHECK_DURATION

        needed = config.SAMPLE_RATE * duration
        deadline = time.time() + duration + 3.0
        log.debug("QualityChecker.check(): waiting for %d samples", needed)

        while reader.samples_available() < needed:
            if time.time() > deadline:
                log.warning("Quality check timeout — not enough samples.")
                return self._timeout_report()
            time.sleep(0.1)

        log.debug("QualityChecker.check(): samples ready, evaluating %d channels", config.NUM_CHANNELS)
        raw = reader.get_latest(needed)
        channel_reports = []

        for ch_idx in range(config.NUM_CHANNELS):
            log.debug("QualityChecker.check(): evaluating ch%d (%s)",
                      ch_idx, self.CHANNEL_NAMES[ch_idx])
            sig = np.array(raw[ch_idx], dtype=np.float64)
            report = self._evaluate_channel(ch_idx, sig)
            log.debug("QualityChecker.check(): ch%d rating=%s amplitude=%.2fuV",
                      ch_idx, report.rating, report.amplitude * 1e6)
            channel_reports.append(report)

        overall = self._overall_rating(channel_reports)
        log.info("QualityChecker.check(): overall rating=%s", overall)
        return QualityReport(channels=channel_reports, overall=overall)

    # -- Per-channel evaluation -------------------------------------------------
    def _evaluate_channel(self, ch_idx: int, sig: np.ndarray) -> ChannelReport:
        log.debug("QualityChecker._evaluate_channel(ch%d): signal length=%d", ch_idx, len(sig))
        issues  = []
        ratings = []
        fs      = config.SAMPLE_RATE

        # 1 -- Amplitude (RMS)
        rms = float(np.sqrt(np.mean(sig ** 2)))
        log.debug("QualityChecker ch%d: RMS=%.2fuV (min=%.0f max=%.0f threshold)",
                  ch_idx, rms * 1e6,
                  config.QUALITY_MIN_AMPLITUDE * 1e6, config.QUALITY_MAX_AMPLITUDE * 1e6)
        if rms < config.QUALITY_MIN_AMPLITUDE:
            issues.append(f"Signal too flat ({rms*1e6:.2f} uV) -- electrode not connected?")
            ratings.append(FAIL)
            log.debug("QualityChecker ch%d: FAIL — signal too flat", ch_idx)
        elif rms > config.QUALITY_MAX_AMPLITUDE:
            issues.append(f"Signal too noisy ({rms*1e6:.0f} uV) -- muscle artifact or bad ground")
            ratings.append(FAIL)
            log.debug("QualityChecker ch%d: FAIL — signal too noisy", ch_idx)
        elif rms > config.QUALITY_MAX_AMPLITUDE * 0.5:
            issues.append(f"Elevated amplitude ({rms*1e6:.0f} uV) -- try relaxing jaw/face")
            ratings.append(WARN)
            log.debug("QualityChecker ch%d: WARN — elevated amplitude", ch_idx)
        else:
            ratings.append(GOOD)

        # 2 -- Band noise ratio (out-of-band vs in-band power)
        in_band  = band_power(sig, fs, config.BANDPASS_LOW, config.BANDPASS_HIGH)
        out_band = band_power(sig, fs, 0.1, 0.5) + band_power(sig, fs, 40.0, 100.0)
        total    = in_band + out_band + 1e-30
        noise_ratio = out_band / total
        log.debug("QualityChecker ch%d: noise_ratio=%.1f%% (threshold=%.0f%%)",
                  ch_idx, noise_ratio * 100, config.QUALITY_NOISE_RATIO * 100)

        if noise_ratio > config.QUALITY_NOISE_RATIO:
            issues.append(f"High out-of-band noise ({noise_ratio*100:.0f}%) -- check ground")
            ratings.append(WARN)
            log.debug("QualityChecker ch%d: WARN — high out-of-band noise", ch_idx)
        else:
            ratings.append(GOOD)

        # 3 -- 50 Hz notch check (mains interference)
        notch_power = band_power(sig, fs, 48.0, 52.0)
        log.debug("QualityChecker ch%d: notch_power=%.2epV^2", ch_idx, notch_power * 1e12)
        if notch_power > 1e-11:
            issues.append(f"Strong 50 Hz interference -- improve ground electrode contact")
            ratings.append(WARN)
            log.debug("QualityChecker ch%d: WARN — 50Hz interference detected", ch_idx)
        else:
            ratings.append(GOOD)

        if FAIL in ratings:
            rating = FAIL
        elif ratings.count(WARN) >= 2:
            rating = WARN
        elif WARN in ratings:
            rating = WARN
        else:
            rating = GOOD

        log.debug("QualityChecker._evaluate_channel(ch%d): final rating=%s", ch_idx, rating)
        return ChannelReport(
            channel     = ch_idx,
            name        = self.CHANNEL_NAMES[ch_idx],
            rating      = rating,
            amplitude   = rms,
            noise_ratio = noise_ratio,
            notch_power = notch_power,
            issues      = issues,
        )

    # -- Helpers ----------------------------------------------------------------
    @staticmethod
    def _overall_rating(reports: list[ChannelReport]) -> str:
        ratings = [r.rating for r in reports]
        if FAIL in ratings:
            return FAIL
        if ratings.count(WARN) >= 2:
            return WARN
        if WARN in ratings:
            return WARN
        return GOOD

    @staticmethod
    def _timeout_report() -> QualityReport:
        dummy = ChannelReport(
            channel=0, name="UNKNOWN", rating=FAIL,
            amplitude=0, noise_ratio=1.0, notch_power=0,
            issues=["Could not collect enough samples -- ADC timeout"]
        )
        return QualityReport(channels=[dummy], overall=FAIL)