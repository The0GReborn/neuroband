"""
simulator.py — EEG Simulation Module for NeuroBand v3
Place in: /home/pi/neuroband/simulator.py

Generates synthetic EEG windows that plug directly into:
    SignalProcessor → FeatureExtractor → Calibrator → KNNClassifier → EMAVoter

Input format matches SignalProcessor.process() exactly:
    list[list[float]]  shape (NUM_CHANNELS, WINDOW_SAMPLES)

Two generation modes:
    Mode A — Simple:   base vector + Gaussian noise (fast debug)
    Mode B — Realistic: multi-sine + drift + spikes (stress test)

Usage:
    from simulator import EEGSimulator, run_simulation_through_pipeline
    sim = EEGSimulator(mode="realistic", noise_level=0.15)
    window = sim.generate_window("YES")
    run_simulation_through_pipeline(["YES", "NO", "YES", "REST"])
"""

from __future__ import annotations

import sys
import time
import logging
import numpy as np
from typing import Iterator, List, Optional

import config

# Force UTF-8 on Windows terminals (cp1252 can't print box-drawing / arrow chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Word base vectors  (Mode A — simple)
#  Shape: (NUM_CHANNELS,)  — one scalar per channel, slightly overlapping
#  Units: volts (same as real ADC output)
# ─────────────────────────────────────────────────────────────────────────────
# These are intentionally close together to stress-test k-NN separability.
# overlap_strength scales how similar adjacent words are (0=fully separated,
# 1=completely identical).
_DEFAULT_BASE_VECTORS: dict[str, list[float]] = {
    "YES":   [25e-6,  10e-6],
    "NO":    [10e-6,  28e-6],
    "HELLO": [30e-6,  30e-6],
    "HELP":  [ 8e-6,  20e-6],
    "REST":  [12e-6,  12e-6],
}

# ─────────────────────────────────────────────────────────────────────────────
#  Frequency params per word  (Mode B — realistic)
#  (base_hz, alpha_amp, beta_amp, phase_offset, drift_bias)
#
#  Words are spread across different EEG bands so that band-power features
#  (delta/theta/alpha/beta) differ substantially after z-score normalisation.
#    YES  — alpha-dominant  (10 Hz fundamental, 20 Hz harmonic)
#    NO   — beta-dominant   (15 Hz fundamental, 30 Hz harmonic)
#    HELLO— theta+alpha     ( 6 Hz fundamental, 12 Hz harmonic)
#    HELP — high-beta       (18 Hz fundamental, 36 Hz harmonic)
#    REST — slow alpha      ( 9 Hz fundamental, low beta)
# ─────────────────────────────────────────────────────────────────────────────
_WORD_FREQ_PARAMS: dict[str, tuple] = {
    "YES":   (10.0, 28e-6,  6e-6,  0.00,  2e-7),
    "NO":    (15.0,  6e-6, 28e-6,  0.80, -2e-7),
    "HELLO": ( 6.0, 24e-6, 14e-6,  1.50,  3e-7),
    "HELP":  (18.0,  8e-6, 26e-6,  2.30, -3e-7),
    "REST":  ( 9.0, 22e-6,  4e-6,  0.00,  0.0 ),
}


def _freq_params_for_word(word: str, word_list: list[str]) -> tuple:
    """Return frequency params for a word, generating them if not predefined."""
    w = word.upper()
    if w in _WORD_FREQ_PARAMS:
        return _WORD_FREQ_PARAMS[w]
    # Auto-generate for unknown words based on position in word_list
    idx       = word_list.index(w) if w in word_list else 0
    base_hz   = 8.0 + idx * 2.5
    alpha_amp = 18e-6 if idx % 2 == 0 else 10e-6
    beta_amp  =  8e-6 if idx % 2 == 0 else 16e-6
    phase     = 0.15 * idx
    drift     = (idx % 3 - 1) * 1e-7
    return (base_hz, alpha_amp, beta_amp, phase, drift)


def _base_vector_for_word(word: str, word_list: list[str], n_channels: int) -> np.ndarray:
    """Return a base amplitude vector for Mode A, auto-generating if needed."""
    w = word.upper()
    if w in _DEFAULT_BASE_VECTORS:
        vec = np.array(_DEFAULT_BASE_VECTORS[w][:n_channels], dtype=np.float64)
        if len(vec) < n_channels:
            vec = np.pad(vec, (0, n_channels - len(vec)), constant_values=vec[-1])
        return vec
    idx = word_list.index(w) if w in word_list else 0
    base = 15e-6 + idx * 1.5e-6
    return np.array([base + ch * 0.5e-6 for ch in range(n_channels)])


# ═════════════════════════════════════════════════════════════════════════════
#  Core EEG Window Generators
# ═════════════════════════════════════════════════════════════════════════════

class EEGSimulator:
    """
    Generates synthetic EEG windows compatible with SignalProcessor.process().

    Parameters
    ----------
    mode            : "simple" (Mode A) or "realistic" (Mode B)
    noise_level     : Gaussian noise scale as fraction of signal amplitude
    drift_strength  : Slow DC drift amplitude multiplier
    spike_probability: Per-window probability of injecting an artifact spike
    overlap_strength: 0=words well-separated, 1=words identical (tests EMAVoter)
    overlap_target  : "rest" (default) — blend toward REST
                      "neighbor"       — blend toward nearest spectrally-adjacent word
                      (neighbor mode simulates confusion between similar words)
    word_list       : List of words to use (defaults to config words)
    fs              : Sampling frequency (defaults to config.SAMPLE_RATE)
    seed            : RNG seed for reproducibility (None = random)
    """

    def __init__(
        self,
        mode:              str   = "realistic",
        noise_level:       float = 0.15,
        drift_strength:    float = 0.3,
        spike_probability: float = 0.05,
        overlap_strength:  float = 0.2,
        overlap_target:    str   = "rest",
        word_list:         Optional[List[str]] = None,
        fs:                int   = None,
        seed:              Optional[int] = None,
    ):
        self.mode              = mode.lower()
        self.noise_level       = float(noise_level)
        self.drift_strength    = float(drift_strength)
        self.spike_probability = float(spike_probability)
        self.overlap_strength  = float(overlap_strength)
        self.overlap_target    = overlap_target.lower()   # "rest" or "neighbor"
        self.word_list         = [w.upper() for w in (word_list or config.load_words())]
        self.fs                = int(fs or config.SAMPLE_RATE)
        self.n_channels        = config.NUM_CHANNELS
        self.window_samples    = config.WINDOW_SAMPLES
        self._rng              = np.random.default_rng(seed)
        self._t                = 0.0           # global time cursor (realistic mode)
        # Stable per-session random frequencies (computed once, reused across windows)
        self._drift_freqs: dict[str, float] = {}   # drift Hz per word
        self._theta_hz: float | None = None         # theta Hz for this session

        log.debug(
            "EEGSimulator: mode=%s noise=%.2f drift=%.2f spike_p=%.3f overlap=%.2f "
            "channels=%d window=%d fs=%d",
            self.mode, self.noise_level, self.drift_strength,
            self.spike_probability, self.overlap_strength,
            self.n_channels, self.window_samples, self.fs,
        )

    # ── Mode A: Simple ────────────────────────────────────────────────────────

    def _generate_simple(self, word: str) -> list[list[float]]:
        """
        Each channel = base_amplitude + Gaussian noise.
        overlap_strength blends the word's base vector toward the REST vector.
        """
        w       = word.upper()
        base    = _base_vector_for_word(w, self.word_list, self.n_channels)
        rest    = _base_vector_for_word("REST", self.word_list, self.n_channels)
        blended = base * (1.0 - self.overlap_strength) + rest * self.overlap_strength

        n   = self.window_samples
        out = []
        for ch in range(self.n_channels):
            amp    = blended[ch]
            noise  = self._rng.standard_normal(n) * amp * self.noise_level
            signal = np.full(n, amp) + noise
            out.append(signal.tolist())
        log.debug("EEGSimulator._generate_simple('%s'): base=%s", w, blended)
        return out

    # ── Mode B: Realistic ─────────────────────────────────────────────────────

    def _generate_realistic(self, word: str) -> list[list[float]]:
        """
        Per-channel time-series built from:
          1. Alpha-band sine at word-specific frequency
          2. Beta harmonic at 2× that frequency
          3. Theta undertone (shared across channels, slight volume-conduction)
          4. Gaussian noise
          5. Slow DC drift (word-specific bias direction)
          6. Occasional artifact spike (5% per window by default)
        Channels are NOT identical — different phase offsets + amplitude swap.
        overlap_strength blends all non-REST words toward the REST signature.
        """
        w = word.upper()
        base_hz, alpha_amp, beta_amp, phase_off, drift_bias = _freq_params_for_word(w, self.word_list)

        # ── Overlap blending ────────────────────────────────────────────────
        if self.overlap_strength > 0 and w != "REST":
            if self.overlap_target == "neighbor":
                # Blend toward nearest spectrally-adjacent word (not REST).
                # "Nearest" = smallest |base_hz difference| among other words.
                # This simulates k-NN confusion between similar-sounding words.
                neighbors = [v for v in self.word_list if v != w and v != "REST"]
                if neighbors:
                    target_word = min(
                        neighbors,
                        key=lambda nb: abs(
                            _freq_params_for_word(nb, self.word_list)[0] - base_hz
                        ),
                    )
                else:
                    target_word = "REST"   # fallback if only one non-REST word
            else:
                target_word = "REST"

            t_hz, t_alpha, t_beta, t_phase, t_drift = _freq_params_for_word(target_word, self.word_list)
            s = self.overlap_strength
            base_hz    = base_hz    * (1 - s) + t_hz    * s
            alpha_amp  = alpha_amp  * (1 - s) + t_alpha  * s
            beta_amp   = beta_amp   * (1 - s) + t_beta   * s
            phase_off  = phase_off  * (1 - s) + t_phase  * s
            drift_bias = drift_bias * (1 - s) + t_drift  * s
            log.debug(
                "_generate_realistic('%s'): overlap=%.2f blended toward '%s'",
                w, s, target_word,
            )

        n  = self.window_samples
        dt = 1.0 / self.fs
        t  = self._t + np.arange(n) * dt

        # Shared volume-conducted theta — frequency randomized once per session
        # (real theta band spans 4–7 Hz and varies between individuals)
        if self._theta_hz is None:
            self._theta_hz = float(self._rng.uniform(4.0, 7.0))
            log.debug("EEGSimulator: session theta_hz=%.2f", self._theta_hz)
        theta_amp    = alpha_amp * 0.3
        common_theta = theta_amp * np.sin(2 * np.pi * self._theta_hz * t)

        # Correlated noise component (shared between channels, as in real EEG)
        common_noise = self._rng.standard_normal(n) * alpha_amp * 0.08

        # FIX 2: drift frequency is stable per word across windows (not re-randomized)
        if w not in self._drift_freqs:
            self._drift_freqs[w] = float(self._rng.uniform(0.1, 0.3))
        drift_freq   = self._drift_freqs[w]
        drift_signal = drift_bias * self.drift_strength * np.sin(2 * np.pi * drift_freq * t)

        # FIX 3: amplitude jitter computed once, shared across channels
        # Per-channel independent jitter was exaggerating asymmetry unrealistically
        trial_jitter = float(self._rng.uniform(0.7, 1.3))

        out = []
        for ch in range(self.n_channels):
            # Channel asymmetry: swap alpha/beta dominance on ch1, add phase shift
            if ch == 0:
                a_amp    = alpha_amp
                b_amp    = beta_amp
                ch_phase = phase_off
            else:
                a_amp    = beta_amp  * 0.9
                b_amp    = alpha_amp * 1.1
                ch_phase = phase_off + 0.3 + 0.15 * (self.word_list.index(w) if w in self.word_list else 0)

            sig  = a_amp * np.sin(2 * np.pi * base_hz       * t + ch_phase)
            sig += b_amp * np.sin(2 * np.pi * (base_hz * 2) * t + ch_phase * 0.5)
            sig += common_theta
            sig += drift_signal
            sig += common_noise
            sig += self._rng.standard_normal(n) * alpha_amp * self.noise_level  # private noise

            # FIX 3: apply shared trial jitter (not per-channel independent)
            sig *= trial_jitter

            out.append(sig.tolist())

        # Artifact spike: decided once per window, injected on ALL channels
        # at the same sample index — real artifacts (blinks, movement) are
        # volume-conducted and appear simultaneously across electrodes.
        if self._rng.random() < self.spike_probability:
            spike_idx = int(self._rng.integers(0, n))
            spike_amp = float(self._rng.uniform(3, 6)) * alpha_amp
            for ch_sig in out:
                ch_sig[spike_idx] += spike_amp
            log.debug(
                "EEGSimulator: artifact spike at sample %d amp=%.2euV (all channels)",
                spike_idx, spike_amp * 1e6,
            )

        # Advance global time cursor so consecutive windows are continuous
        self._t += n * dt
        return out

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_window(self, word: str) -> list[list[float]]:
        """
        Generate one EEG window for *word*.

        Returns
        -------
        list[list[float]] — shape (NUM_CHANNELS, WINDOW_SAMPLES)
        Compatible with SignalProcessor.process() directly.
        """
        if self.mode == "simple":
            return self._generate_simple(word)
        return self._generate_realistic(word)

    def reset_time(self):
        """Reset global time cursor (start a new simulated session)."""
        self._t = 0.0
        log.debug("EEGSimulator.reset_time(): time cursor reset to 0")


# ═════════════════════════════════════════════════════════════════════════════
#  Batch Generator  (A)
# ═════════════════════════════════════════════════════════════════════════════

def generate_stream(
    sequence:            List[str],
    duration_per_word:   float = 1.0,
    fs:                  int   = None,
    mode:                str   = "realistic",
    noise_level:         float = 0.15,
    drift_strength:      float = 0.3,
    spike_probability:   float = 0.05,
    overlap_strength:    float = 0.2,
    seed:                Optional[int] = None,
) -> list[dict]:
    """
    Batch generator — returns a list of window dicts.

    Each dict has:
      "word"    : str                     — ground-truth label
      "window"  : list[list[float]]       — raw EEG window
      "t_start" : float                   — start time of window (seconds)

    Parameters
    ----------
    sequence          : list of word labels to generate in order
    duration_per_word : seconds of EEG to generate per word
                        (determines how many windows per word)
    fs                : sampling frequency (default: config.SAMPLE_RATE)
    """
    _fs    = int(fs or config.SAMPLE_RATE)
    n_win  = max(1, int(duration_per_word * _fs / config.WINDOW_SAMPLES))
    sim    = EEGSimulator(
        mode=mode, noise_level=noise_level, drift_strength=drift_strength,
        spike_probability=spike_probability, overlap_strength=overlap_strength,
        fs=_fs, seed=seed,
    )
    results = []
    t_cursor = 0.0
    for word in sequence:
        for _ in range(n_win):
            window = sim.generate_window(word)
            results.append({"word": word.upper(), "window": window, "t_start": t_cursor})
            t_cursor += config.WINDOW_SAMPLES / _fs
    log.debug("generate_stream: %d words → %d windows total", len(sequence), len(results))
    return results


# ═════════════════════════════════════════════════════════════════════════════
#  Real-Time Generator  (B)
# ═════════════════════════════════════════════════════════════════════════════

def stream_generator(
    sequence:            List[str],
    duration_per_word:   float = 1.0,
    fs:                  int   = None,
    mode:                str   = "realistic",
    noise_level:         float = 0.15,
    drift_strength:      float = 0.3,
    spike_probability:   float = 0.05,
    overlap_strength:    float = 0.2,
    realtime:            bool  = False,
    seed:                Optional[int] = None,
) -> Iterator[dict]:
    """
    Real-time generator — yields one window dict at a time.

    Plug directly into an inference loop:

        for frame in stream_generator(["YES", "NO", "YES"]):
            clean = processor.process(frame["window"])
            ...

    Parameters
    ----------
    realtime : if True, sleep between yields to match real hardware timing
    """
    _fs       = int(fs or config.SAMPLE_RATE)
    win_dur   = config.WINDOW_SAMPLES / _fs
    step_dur  = config.V3_STEP_SAMPLES / _fs
    n_windows = max(1, int(duration_per_word / step_dur))
    sim = EEGSimulator(
        mode=mode, noise_level=noise_level, drift_strength=drift_strength,
        spike_probability=spike_probability, overlap_strength=overlap_strength,
        fs=_fs, seed=seed,
    )
    t_cursor = 0.0
    for word in sequence:
        for _ in range(n_windows):
            window = sim.generate_window(word)
            yield {"word": word.upper(), "window": window, "t_start": t_cursor}
            # FIX 1: advance by step_dur (V3_STEP_SAMPLES/fs), not win_dur
            # n_windows was computed from step_dur so t_cursor must match
            t_cursor += step_dur
            if realtime:
                time.sleep(step_dur)


# ═════════════════════════════════════════════════════════════════════════════
#  Pipeline Integration Helper
# ═════════════════════════════════════════════════════════════════════════════

def run_simulation_through_pipeline(
    sequence:            List[str],
    duration_per_word:   float = 2.0,
    mode:                str   = "realistic",
    noise_level:         float = 0.15,
    drift_strength:      float = 0.3,
    spike_probability:   float = 0.05,
    overlap_strength:    float = 0.2,
    knn_model_path:      Optional[str] = None,
    retrain:             bool  = False,
    seed:                Optional[int] = 42,
    verbose:             bool  = True,
):
    """
    Run simulated EEG through the full NeuroBand pipeline and print results.

    Pipeline:
        EEGSimulator → SignalProcessor → FeatureExtractor
            → Calibrator → KNNClassifier → EMAVoter → output

    Parameters
    ----------
    sequence          : list of words to simulate (ground-truth labels)
    duration_per_word : seconds of signal per word
    verbose           : print per-frame output

    Returns
    -------
    list[dict] — one entry per emitted command:
        {"word": ground_truth, "predicted": label, "confidence": float}
    """
    # Lazy imports — keeps simulator standalone if pipeline isn't importable
    from signal_processing  import SignalProcessor
    from feature_extraction import FeatureExtractor
    from calibration        import Calibrator
    from classifier         import KNNClassifier, EMAVoter

    proc      = SignalProcessor()
    ext       = FeatureExtractor()
    cal       = Calibrator()
    knn       = KNNClassifier()

    # ── Retrain k-NN on simulated data so features match ─────────────
    if retrain:
        knn = retrain_knn_from_simulator(
            knn, words=list(dict.fromkeys([w.upper() for w in sequence])),
            mode=mode, noise_level=noise_level, verbose=verbose,
        )

    all_classes = list(dict.fromkeys([w.upper() for w in sequence] + ["REST"]))
    voter     = EMAVoter(classes=all_classes)

    if verbose:
        print("\n" + "=" * 60)
        print("  NeuroBand Simulator — Pipeline Integration Test")
        print("=" * 60)
        print(f"  Mode           : {mode}")
        print(f"  Sequence       : {[w.upper() for w in sequence]}")
        print(f"  Duration/word  : {duration_per_word}s")
        print(f"  Noise level    : {noise_level}")
        print(f"  Overlap        : {overlap_strength}")
        print(f"  KNN ready      : {knn.is_ready}")
        print(f"  KNN classes    : {knn.classes}")
        print("=" * 60 + "\n")

    # FIX 4: enforce KNN readiness — untrained KNN gives meaningless REST outputs
    if not knn.is_ready:
        if verbose:
            print("  [!] k-NN is not trained. No meaningful predictions possible.")
            print("  [!] Run a training session first:")
            print("       python main.py --simulate")
            print("     Then retrain and re-run the simulation.")
            print("=" * 60 + "\n")
        log.warning("run_simulation_through_pipeline: k-NN untrained — returning empty results")
        return []

    # ── Warn about untrained words ──────────────────────────────────────
    # If the k-NN has never seen a word it will always output 0 belief for it.
    # Predictions for those words are meaningless — tell the user now.
    knn_classes_upper = [str(c).upper() for c in knn.classes]
    untrained = [
        w.upper() for w in sequence
        if w.upper() not in knn_classes_upper and w.upper() != "REST"
    ]
    untrained = list(dict.fromkeys(untrained))   # deduplicate, preserve order
    if untrained and verbose:
        print(f"  [!] Words not in k-NN training data: {untrained}")
        print(f"  [!] Predictions for these words will be biased/wrong.")
        print(f"      Train them via the web UI then re-run.\n")
        log.warning(
            "run_simulation_through_pipeline: untrained words %s — results will be misleading",
            untrained,
        )

    # -- Filter warmup ---------------------------------------------------
    # sosfilt_zi initialises IIR state for a 1 V DC step, not for a uV EEG
    # signal.  The first ~4 real windows have a large transient that trips
    # the 200 uV artifact gate.  Running 8 silent REST windows through the
    # processor lets the filter state converge before real data arrives.
    _wu = EEGSimulator(mode="simple", noise_level=0.02, spike_probability=0.0,
                       overlap_strength=0.0, seed=0)
    for _ in range(8):
        proc.process(_wu.generate_window("REST"))
    log.debug("run_simulation_through_pipeline: filter warmup complete (8 windows)")

    results   = []
    prev_word = None
    n_windows = 0
    n_cmd     = 0

    for frame in stream_generator(
        sequence=sequence,
        duration_per_word=duration_per_word,
        mode=mode,
        noise_level=noise_level,
        drift_strength=drift_strength,
        spike_probability=spike_probability,
        overlap_strength=overlap_strength,
        seed=seed,
    ):
        word   = frame["word"]
        window = frame["window"]

        # Reset voter when word changes to simulate separate utterances
        if prev_word is not None and word != prev_word:
            voter.reset()
        prev_word = word

        # ── Signal Processing ────────────────────────────────────────────
        clean = proc.process(window)
        if clean is None:
            if verbose:
                print(f"  [w={n_windows:>4}] {word:<8} → REJECTED (artifact)")
            continue

        # ── Feature Extraction ───────────────────────────────────────────
        fv = ext.extract(clean)
        if fv is None:
            continue

        # ── Calibration Normalisation ────────────────────────────────────
        fv_cal = cal.transform(fv)

        # ── k-NN Classification ──────────────────────────────────────────
        proba        = knn.predict_proba_dict(fv_cal)
        winner, conf = voter.push(proba)
        lead, lconf  = voter.confidence_now()
        beliefs      = voter.belief_state()

        n_windows += 1

        if winner:
            n_cmd += 1
            results.append({
                "word":       word,
                "predicted":  winner,
                "confidence": round(conf, 3),
            })
            correct = "[OK]" if winner == word else "[X] "
            if verbose:
                print(f"  [w={n_windows:>4}] {word:<8} -> COMMAND: {winner:<8} "
                      f"conf={conf:.2f}  {correct}")
        else:
            if verbose and n_windows % 10 == 0:
                beliefs_str = "  ".join(
                    f"{k}={v:.3f}" for k, v in sorted(beliefs.items())
                )
                print(f"  [w={n_windows:>4}] {word:<8} -> lead={lead:<8} "
                      f"lconf={lconf:.2f}  [{beliefs_str}]")

    if verbose:
        print("\n" + "-" * 60)
        print(f"  Windows processed : {n_windows}")
        print(f"  Commands emitted  : {n_cmd}")
        if results:
            correct = sum(1 for r in results if r["predicted"] == r["word"])
            print(f"  Correct commands  : {correct}/{len(results)} "
                  f"({correct/len(results)*100:.0f}%)")
        else:
            print("  No commands emitted - k-NN may not be trained yet.")
            print("  Run a training session first, then re-run the simulation.")
        print("=" * 60 + "\n")

    return results


# ═════════════════════════════════════════════════════════════════════════════
#  Retrain k-NN from simulated data
# ═════════════════════════════════════════════════════════════════════════════

def retrain_knn_from_simulator(
    knn,
    words: List[str],
    n_trials_per_word: int = 50,
    mode: str = "realistic",
    noise_level: float = 0.15,
    verbose: bool = True,
    seed: int = 0,
):
    """
    Generate training data with the simulator and fit a fresh k-NN.

    Uses a DIFFERENT seed (default 0) than the test run (default 42) so
    training and test data are not identical.
    """
    from signal_processing  import SignalProcessor
    from feature_extraction import FeatureExtractor
    from calibration        import Calibrator

    if verbose:
        print("  [retrain] Generating simulated training data...")

    proc = SignalProcessor()
    ext  = FeatureExtractor()
    cal  = Calibrator()

    # Filter warmup
    wu = EEGSimulator(mode="simple", noise_level=0.02, spike_probability=0.0,
                      overlap_strength=0.0, seed=99)
    for _ in range(8):
        proc.process(wu.generate_window("REST"))

    sim = EEGSimulator(
        mode=mode, noise_level=noise_level, spike_probability=0.02,
        overlap_strength=0.0, seed=seed,
    )

    X_list, y_list = [], []
    for word in words:
        accepted = 0
        for _ in range(n_trials_per_word * 2):   # generate extra in case of rejections
            if accepted >= n_trials_per_word:
                break
            window = sim.generate_window(word.upper())
            clean  = proc.process(window)
            if clean is None:
                continue
            fv = ext.extract(clean)
            if fv is None:
                continue
            fv_cal = cal.transform(fv)
            X_list.append(fv_cal)
            y_list.append(word.upper())
            accepted += 1

    X = np.array(X_list)
    y = np.array(y_list)

    if verbose:
        from collections import Counter
        counts = Counter(y)
        print(f"  [retrain] Training k-NN on {len(X)} samples: {dict(counts)}")

    knn.fit(X, y)

    if verbose:
        print(f"  [retrain] k-NN fitted — classes={knn.classes}\n")

    return knn


# ═════════════════════════════════════════════════════════════════════════════
#  Standalone CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser(description="NeuroBand EEG Simulator")
    ap.add_argument("--mode",       default="realistic", choices=["simple", "realistic"],
                    help="Simulation mode (default: realistic)")
    ap.add_argument("--words",      nargs="+", default=None,
                    help="Word sequence to simulate (default: all config words)")
    ap.add_argument("--duration",   type=float, default=2.0,
                    help="Seconds per word (default: 2.0)")
    ap.add_argument("--noise",      type=float, default=0.15,
                    help="Noise level fraction (default: 0.15)")
    ap.add_argument("--drift",      type=float, default=0.3,
                    help="Drift strength (default: 0.3)")
    ap.add_argument("--spikes",     type=float, default=0.05,
                    help="Spike probability per window (default: 0.05)")
    ap.add_argument("--overlap",    type=float, default=0.2,
                    help="Overlap strength between words (default: 0.2)")
    ap.add_argument("--retrain",    action="store_true",
                    help="Retrain k-NN on simulated data before testing")
    ap.add_argument("--pipeline",   action="store_true",
                    help="Run through full pipeline (requires trained k-NN)")
    ap.add_argument("--seed",       type=int, default=42,
                    help="RNG seed (default: 42)")
    ap.add_argument("--debug",      action="store_true",
                    help="Enable DEBUG logging")
    args = ap.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    words = [w.upper() for w in args.words] if args.words else config.load_words()

    if args.pipeline:
        run_simulation_through_pipeline(
            sequence          = words,
            duration_per_word = args.duration,
            mode              = args.mode,
            noise_level       = args.noise,
            drift_strength    = args.drift,
            retrain           = args.retrain,
            spike_probability = args.spikes,
            overlap_strength  = args.overlap,
            seed              = args.seed,
        )
    else:
        # Quick window generation demo
        sim = EEGSimulator(
            mode              = args.mode,
            noise_level       = args.noise,
            drift_strength    = args.drift,
            spike_probability = args.spikes,
            overlap_strength  = args.overlap,
            seed              = args.seed,
        )
        print(f"\nEEGSimulator — mode={args.mode}  channels={config.NUM_CHANNELS}"
              f"  window={config.WINDOW_SAMPLES}  fs={config.SAMPLE_RATE}\n")
        for word in words:
            window = sim.generate_window(word)
            ch_stats = []
            for ch, samples in enumerate(window):
                a = np.array(samples)
                ch_stats.append(
                    f"ch{ch}: mean={a.mean()*1e6:+7.2f}uV  "
                    f"rms={np.sqrt(np.mean(a**2))*1e6:6.2f}uV  "
                    f"p2p={a.ptp()*1e6:6.2f}uV"
                )
            print(f"  {word:<10}  " + "  |  ".join(ch_stats))
        print()
