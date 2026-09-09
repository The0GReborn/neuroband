"""
pattern_trainer.py — Structured Training Session for NeuroBand v3
Place in: /home/pi/neuroband/pattern_trainer.py

Flow per trial:
  1. Beep (high tone) + "THINK: <word>" shown on app
  2. Record V3_THINK_SECONDS of EEG → extract features → save sample
  3. Instantly refit k-NN
  4. Emit progress over WebSocket
  5. Beep (low tone) + "REST" shown on app
  6. Wait V3_REST_SECONDS
  7. Repeat

Simulation mode:
  In --simulate, fake EEG is generated instead of reading hardware.
  Pattern varies per word so k-NN can actually learn distinct signatures.
"""

import time
import math
import random
import logging
import threading
import numpy as np

import config as cfg
from signal_processing  import SignalProcessor
from feature_extraction import FeatureExtractor

log = logging.getLogger(__name__)

try:
    import pygame
    if not pygame.mixer.get_init():
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    _PG = True
except Exception as e:
    _PG = False
    log.warning("pygame unavailable — audio cues disabled: %s", e)


def _tone(freq: int, dur: float = 0.18):
    """Play a simple sine tone. Non-critical — fails silently."""
    if not _PG:
        return
    try:
        fs   = 44100
        t    = np.linspace(0, dur, int(fs * dur), endpoint=False)
        wave = (np.sin(2 * np.pi * freq * t) * 28000).astype(np.int16)
        snd  = pygame.sndarray.make_sound(wave)
        snd.play()
        time.sleep(dur + 0.04)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  Simulated EEG per word (for --simulate mode)
# ══════════════════════════════════════════════════════════════════════════════
def _sim_window(word: str, word_list: list) -> list:
    """
    Generate a fake 2-channel EEG window with a word-specific signature.
    Each word gets a unique dominant frequency so k-NN can distinguish them.
    Returns list of 2 lists (channels), each WINDOW_SAMPLES long.

    Intentionally harder than the original clean sine version:
      - Amplitude jitter per trial (real EEG varies +-30% trial-to-trial)
      - Slow DC drift (common electrode artifact)
      - Correlated noise between channels (real EEG channels share common noise)
      - Occasional high-amplitude transient (muscle/blink artifact)
    This tightens the sim/real distribution gap so k-NN failures show up
    during simulation rather than only on hardware.
    """
    idx        = word_list.index(word) if word in word_list else 0
    base_freq  = 8.0 + idx * 2.5        # 8 Hz, 10.5 Hz, 13 Hz per word
    amp        = 20e-6 * (1.0 + np.random.uniform(-0.3, 0.3))  # +-30% jitter
    n          = cfg.WINDOW_SAMPLES
    fs         = cfg.SAMPLE_RATE
    t          = np.arange(n) / fs

    # Slow DC drift (0.1-0.3 Hz) common in real electrode recordings
    drift_freq = np.random.uniform(0.1, 0.3)
    drift      = amp * 0.4 * np.sin(2 * np.pi * drift_freq * t)

    # Correlated noise shared between channels (volume-conducted noise)
    common_noise = np.random.normal(0, amp * 0.1, n)

    def ch(phase_shift=0.0):
        sig  = amp * np.sin(2 * np.pi * base_freq * t + phase_shift)
        sig += amp * 0.3 * np.sin(2 * np.pi * (base_freq * 2) * t)
        sig += drift
        sig += common_noise                          # shared component
        sig += np.random.normal(0, amp * 0.2, n)    # channel-private noise
        # 5% chance of a transient artifact per window
        if np.random.random() < 0.05:
            art_idx = np.random.randint(0, n)
            sig[art_idx] += amp * np.random.uniform(3, 6)
        return sig.tolist()

    return [ch(0.0), ch(0.3)]


# ══════════════════════════════════════════════════════════════════════════════
#  Training Session
# ══════════════════════════════════════════════════════════════════════════════
class TrainingSession:

    def __init__(self, word, n_trials, reader, word_manager, knn,
                 simulate=False, emit_fn=None, voter=None):
        self.word      = word.upper().strip()
        self.n_trials  = n_trials
        self._reader   = reader
        self._wm       = word_manager
        self._knn      = knn
        self._voter    = voter   # v5.2: EMAVoter ref for reset after refit (FIX 17)
        self._sim      = simulate
        self._emit     = emit_fn or (lambda e, d: None)
        self._proc     = SignalProcessor()
        self._ext      = FeatureExtractor()
        self._running  = False
        self._thread   = None
        self.collected = 0

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._run, daemon=True, name=f"Train-{self.word}")
        self._thread.start()

    def stop(self):
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def _run(self):
        log.info("TrainingSession._run(): starting '%s' — %d trials", self.word, self.n_trials)
        self._emit("training_started", {"word": self.word, "trials": self.n_trials})

        for trial in range(1, self.n_trials + 1):
            if not self._running:
                log.debug("TrainingSession._run(): stopped at trial %d/%d", trial, self.n_trials)
                break

            log.debug("TrainingSession._run(): trial %d/%d — THINK cue", trial, self.n_trials)
            self._emit("training_cue", {
                "phase": "THINK", "word": self.word,
                "trial": trial,   "total": self.n_trials,
            })
            _tone(880, 0.15)

            log.debug("TrainingSession._run(): recording EEG window for trial %d", trial)
            fv = self._record()
            if fv is not None:
                self._wm.append_sample(self.word, fv)
                self.collected += 1
                log.debug("TrainingSession._run(): trial %d accepted — total collected=%d",
                          trial, self.collected)
                self._emit("training_sample", {
                    "word":      self.word,
                    "trial":     trial,
                    "total":     self.n_trials,
                    "collected": self.collected,
                    "progress":  round(self.collected / self.n_trials, 2),
                    "knn_ready": self._knn.is_ready,
                })
            else:
                log.warning("TrainingSession._run(): trial %d rejected (bad signal)", trial)
                self._emit("training_sample", {
                    "word": self.word, "trial": trial, "total": self.n_trials,
                    "collected": self.collected,
                    "progress": round(self.collected / self.n_trials, 2),
                    "error": "Noisy signal — relax and try again",
                })

            if not self._running:
                break

            log.debug("TrainingSession._run(): trial %d — REST cue (%.1fs)",
                      trial, cfg.V3_REST_SECONDS)
            self._emit("training_cue", {"phase": "REST", "trial": trial, "total": self.n_trials})
            _tone(440, 0.15)
            time.sleep(cfg.V3_REST_SECONDS)

        log.debug("TrainingSession._run(): all trials complete — collected=%d, refitting k-NN",
                  self.collected)
        knn_updated = False
        if self.collected > 0:
            knn_updated = self._knn.update(self._wm)
            if knn_updated:
                log.info("k-NN refit on full dataset after session complete.")
                # v5.2 FIX 17: sync EMAVoter class list with newly fitted k-NN
                if self._voter is not None:
                    new_classes = self._knn.classes + (
                        ["REST"] if "REST" not in self._knn.classes else []
                    )
                    self._voter.reset(classes=new_classes)
                    log.debug("TrainingSession: EMAVoter reset with classes=%s", new_classes)
            else:
                log.warning("k-NN refit skipped — not enough data across words yet.")

        self._running = False
        log.info("TrainingSession._run(): done — '%s' %d/%d samples collected",
                 self.word, self.collected, self.n_trials)
        self._emit("training_complete", {
            "word":        self.word,
            "collected":   self.collected,
            "requested":   self.n_trials,
            "knn_ready":   self._knn.is_ready,
            "knn_updated": knn_updated,
            "counts":      self._wm.sample_counts(),
        })

    def _record(self):
        """
        Capture one EEG window and return a validated feature vector, or None.
        """
        log.debug("TrainingSession._record(): starting — simulate=%s", self._sim)
        if self._sim:
            log.debug("TrainingSession._record(): generating simulated window for '%s'", self.word)
            raw   = _sim_window(self.word, self._wm.words)
            clean = self._proc.process(raw)
        else:
            needed = cfg.WINDOW_SAMPLES
            t0_samples = self._reader.samples_available()
            target     = t0_samples + needed
            deadline   = time.time() + cfg.V3_THINK_SECONDS + 1.5
            log.debug("TrainingSession._record(): waiting for %d samples (have %d, target=%d)",
                      needed, t0_samples, target)
            while self._reader.samples_available() < target:
                if time.time() > deadline:
                    log.warning(
                        "_record: deadline hit (have %d, need %d)",
                        self._reader.samples_available(), target,
                    )
                    return None
                time.sleep(0.02)
            log.debug("TrainingSession._record(): sample target reached, fetching window")
            raw = self._reader.get_latest(needed)
            if raw is None:
                log.warning("TrainingSession._record(): get_latest returned None")
                return None
            log.debug("TrainingSession._record(): processing raw window through SignalProcessor")
            clean = self._proc.process(raw)

        if clean is None:
            log.debug("TrainingSession._record(): SignalProcessor rejected window (artifact)")
            return None

        log.debug("TrainingSession._record(): extracting features")
        fv = self._ext.extract(clean)

        if fv is None:
            log.warning("_record: FeatureExtractor returned None")
            return None
        fv = np.asarray(fv, dtype=np.float32)
        if not np.all(np.isfinite(fv)):
            log.warning("_record: feature vector contains NaN/Inf — discarding trial")
            return None
        if not hasattr(self, "_expected_fv_len"):
            self._expected_fv_len = len(fv)
            log.debug("TrainingSession._record(): expected feature length set to %d", len(fv))
        elif len(fv) != self._expected_fv_len:
            log.error(
                "_record: feature length changed %d -> %d — FeatureExtractor drift detected",
                self._expected_fv_len, len(fv),
            )
            return None

        log.debug("TrainingSession._record(): feature vector OK, length=%d", len(fv))
        return fv


# ══════════════════════════════════════════════════════════════════════════════
#  Session Manager
# ══════════════════════════════════════════════════════════════════════════════
class SessionManager:
    """Ensures only one training session runs at a time."""

    def __init__(self, reader, word_manager, knn, voter=None, simulate=False):
        log.debug("SessionManager.__init__(): initialising (simulate=%s)", simulate)
        self._reader  = reader
        self._wm      = word_manager
        self._knn     = knn
        self._voter   = voter   # v5.2: passed through to TrainingSession for reset (FIX 17)
        self._sim     = simulate
        self._session = None

    def start(self, word: str, n_trials: int, emit_fn) -> tuple:
        log.debug("SessionManager.start(): word='%s' trials=%d", word, n_trials)
        if self._session and self._session.is_running():
            log.warning("SessionManager.start(): rejected — session already running")
            return False, "A session is already running."
        if word.upper() not in self._wm.words:
            log.warning("SessionManager.start(): rejected — '%s' not in word list %s",
                        word, self._wm.words)
            return False, f"'{word}' not in word list."
        log.debug("SessionManager.start(): creating TrainingSession for '%s'", word)
        self._session = TrainingSession(
            word=word, n_trials=n_trials,
            reader=self._reader, word_manager=self._wm,
            knn=self._knn, simulate=self._sim, emit_fn=emit_fn,
            voter=self._voter,   # v5.2: for EMAVoter reset after refit (FIX 17)
        )
        self._session.start()
        log.info("SessionManager.start(): TrainingSession started for '%s'", word)
        return True, f"Training '{word}' started."

    def stop(self) -> str:
        log.debug("SessionManager.stop(): called")
        if self._session and self._session.is_running():
            self._session.stop()
            log.info("SessionManager.stop(): active session stopped")
            return "Session stopped."
        log.debug("SessionManager.stop(): no active session to stop")
        return "No active session."

    def is_busy(self) -> bool:
        return bool(self._session and self._session.is_running())