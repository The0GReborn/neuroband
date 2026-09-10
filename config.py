"""
config.py — NeuroBand Unified Configuration (v2 + v3)
Place in: /home/pi/neuroband/config.py

Single source of truth for both v2 and v3.
v3 is the default mode. Use --v2 flag in main.py to run legacy mode.

CHANGELOG (fixes applied on top of original):
  FIX-CFG-1  V5_EMA_MIN_WINDOWS raised 8 → 12. Eight windows at 200 ms/step =
             1.6 s warmup, which is shorter than one YES utterance. 12 windows
             gives 2.4 s — enough for the EMA to accumulate meaningful evidence
             before any commit is allowed.

  FIX-CFG-2  V5_EMA_ALPHA lowered 0.25 → 0.10.  A high alpha causes the
             belief state to lurch on single-frame noise; 0.10 gives a time
             constant of ~2 s at 200 ms/step, matching one utterance length.
             (Kept consistent with the hard-coded override in EMAVoter.__init__
             which was already 0.1 — this makes it explicit in config so it
             can be tuned without touching classifier.py.)

  FIX-CFG-3  V5_EMA_CONF_MIN raised 0.55 → 0.65.  Because the belief-share
             ratio can reach 1.0 after only one active frame (all other beliefs
             are 0), a low threshold admits false commits.  0.65 paired with
             V5_EMA_MIN_BELIEF_ABS (new, below) gives a two-gate commit.

  FIX-CFG-4  V5_EMA_MIN_BELIEF_ABS = 0.05 (NEW).  Absolute belief floor for
             the winning class before any commit.  Prevents conf=1.0 commits
             when the winning belief is e.g. 0.001 vs 0.000 for all others.
             EMAVoter.push() must check this; see classifier.py FIX-CLS-1.

  FIX-CFG-5  V5_EMA_REST_MARGIN raised 0.0 → 0.05.  A tiny positive margin
             means REST must beat the best non-REST belief by 5 % before it
             actively decays all beliefs.  This prevents spurious REST-decay
             on noisy frames where rest_p and best_nr are nearly equal.

  FIX-CFG-6  V3_COOLDOWN_SECONDS lowered 3.0 → 2.5.  The previous value
             suppressed the second/third commands in a 3-second word window.
             At 2.5 s each word still has a clean gap but back-to-back words
             no longer block each other.

  FIX-CFG-7  V3_KNN_METRIC = "pca_euclidean" (was implicitly "euclidean").
             PCA whitening equalises class covariance and dramatically improves
             HELLO's decision region, which shrank to near-zero in Euclidean
             space (only 2.5 % of random inputs mapped to HELLO in the probe).

  FIX-CFG-8  V3_KNN_K raised 5 → 7.  More neighbours reduce the effect of
             single outlier training samples; important when per-class N = 36.

  FIX-CFG-9  V3_TARGET_TRIALS raised 40 → 60 and V3_THINK_SECONDS raised
             2.0 → 2.5.  36 balanced samples (144 total) is too thin for a
             25-D feature space.  60 trials × 4 words = 240 raw samples gives
             the k-NN enough geometry to learn clean decision boundaries.
"""

import os
import json

# ══════════════════════════════════════════════════════════════════════════════
#  System Version
# ══════════════════════════════════════════════════════════════════════════════
VERSION = "3"          # default; overridden to "2" by --v2 flag at runtime

# ══════════════════════════════════════════════════════════════════════════════
#  ADC / Hardware
# ══════════════════════════════════════════════════════════════════════════════
SAMPLE_RATE      = 250
NUM_CHANNELS     = 2
ADC_BITS         = 24
ADC_VREF         = 2.5
ADC_GAIN         = 128

ADC_CLK_PIN      = 11
ADC_DATA_PIN_CH0 = 9
ADC_DATA_PIN_CH1 = 10

# ══════════════════════════════════════════════════════════════════════════════
#  Signal Processing
# ══════════════════════════════════════════════════════════════════════════════
BANDPASS_LOW     = 0.5
BANDPASS_HIGH    = 40.0
NOTCH_FREQ       = 50.0        # change to 60.0 for North America
FILTER_ORDER     = 4

# ══════════════════════════════════════════════════════════════════════════════
#  Frequency Bands
# ══════════════════════════════════════════════════════════════════════════════
BAND_DELTA       = (0.5,  4.0)
BAND_THETA       = (4.0,  8.0)
BAND_ALPHA       = (8.0, 12.0)
BAND_BETA        = (12.0,30.0)
BAND_GAMMA       = (30.0,40.0)

# ══════════════════════════════════════════════════════════════════════════════
#  Buffering
# ══════════════════════════════════════════════════════════════════════════════
WINDOW_SECONDS   = 2
WINDOW_SAMPLES   = SAMPLE_RATE * WINDOW_SECONDS    # 500
STEP_SAMPLES     = SAMPLE_RATE // 2                # 125  (v2 default)
BUFFER_SECONDS   = 10
BUFFER_SAMPLES   = SAMPLE_RATE * BUFFER_SECONDS

# ══════════════════════════════════════════════════════════════════════════════
#  V2 Classifier (LDA fallback)
# ══════════════════════════════════════════════════════════════════════════════
V2_LABELS            = ["YES", "NO", "REST"]
LABELS               = V2_LABELS               # alias kept for v2 imports
MODEL_PATH           = "models/neuroband_model.pkl"
CONFIDENCE_THRESH    = 0.70                    # v2 LDA threshold

# ══════════════════════════════════════════════════════════════════════════════
#  V2 Audio
# ══════════════════════════════════════════════════════════════════════════════
AUDIO_DIR            = "audio/"
AUDIO_FILES          = {
    "YES":  "audio/yes.wav",
    "NO":   "audio/no.wav",
    "REST": None,
}

# ══════════════════════════════════════════════════════════════════════════════
#  Logging
# ══════════════════════════════════════════════════════════════════════════════
LOG_DIR              = "logs/"
LOG_RAW              = True
LOG_FEATURES         = True
LOG_LABELS           = True

# ══════════════════════════════════════════════════════════════════════════════
#  Simulation
# ══════════════════════════════════════════════════════════════════════════════
SIMULATE_ADC         = False
SIM_NOISE_LEVEL      = 0.3

# ══════════════════════════════════════════════════════════════════════════════
#  Calibration
# ══════════════════════════════════════════════════════════════════════════════
CALIBRATION_DURATION  = 30
CALIBRATION_PATH      = "models/baseline.pkl"
CALIBRATION_REQUIRED  = True

# ══════════════════════════════════════════════════════════════════════════════
#  Signal Quality
# ══════════════════════════════════════════════════════════════════════════════
QUALITY_CHECK_DURATION = 5
QUALITY_MIN_AMPLITUDE  = 0.5e-6
QUALITY_MAX_AMPLITUDE  = 500e-6
QUALITY_NOISE_RATIO    = 0.6
QUALITY_BLOCK_ON_FAIL  = False

# ══════════════════════════════════════════════════════════════════════════════
#  Dashboard (v2)
# ══════════════════════════════════════════════════════════════════════════════
DASHBOARD_REFRESH_HZ  = 4
DASHBOARD_HISTORY     = 40

# ══════════════════════════════════════════════════════════════════════════════
#  V3 — Word System
# ══════════════════════════════════════════════════════════════════════════════
V3_WORDS_PATH         = "models/v3_words.json"
V3_MAX_WORDS          = 10
V3_DEFAULT_WORDS      = ["HELLO", "HELP", "YES", "NO"]

def load_words() -> list:
    if os.path.exists(V3_WORDS_PATH):
        try:
            with open(V3_WORDS_PATH) as f:
                data = json.load(f)
            return [w.upper().strip() for w in data.get("words", V3_DEFAULT_WORDS) if w.strip()]
        except Exception:
            pass
    return list(V3_DEFAULT_WORDS)

def save_words(words: list):
    os.makedirs(os.path.dirname(V3_WORDS_PATH), exist_ok=True)
    with open(V3_WORDS_PATH, "w") as f:
        json.dump({"words": [w.upper().strip() for w in words]}, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
#  V3 — Paths
# ══════════════════════════════════════════════════════════════════════════════
V3_KNN_MODEL_PATH     = "models/v3_knn.pkl"
V3_DATASET_DIR        = "logs/v3/"
V3_AUDIO_DIR          = "audio/v3/"

# ══════════════════════════════════════════════════════════════════════════════
#  V3 — Democracy / EMA Voting
# ══════════════════════════════════════════════════════════════════════════════
V3_VOTE_WINDOW        = 15      # sliding buffer size (legacy DemocracyVoter)
V3_CONFIDENCE_MIN     = 0.50    # below this → silent, no output
V3_COOLDOWN_SECONDS   = 2.5     # FIX-CFG-6: was 3.0; allows back-to-back words

# ══════════════════════════════════════════════════════════════════════════════
#  V3 — k-NN
# ══════════════════════════════════════════════════════════════════════════════
V3_KNN_K              = 7       # FIX-CFG-8: was 5; more neighbours → smoother boundaries
V3_MIN_SAMPLES        = 3       # minimum samples per word before predicting
V3_KNN_METRIC         = "pca_euclidean"  # FIX-CFG-7: PCA whitening fixes HELLO dead-zone

# ══════════════════════════════════════════════════════════════════════════════
#  V3 — Training Session
# ══════════════════════════════════════════════════════════════════════════════
V3_THINK_SECONDS      = 2.5     # FIX-CFG-9: was 2.0; longer window = richer features
V3_REST_SECONDS       = 2.0
V3_TARGET_TRIALS      = 60      # FIX-CFG-9: was 40; 60 trials × 4 words = 240 samples

# ══════════════════════════════════════════════════════════════════════════════
#  V3 — Inference
# ══════════════════════════════════════════════════════════════════════════════
V3_STEP_SAMPLES       = SAMPLE_RATE // 5    # 50 samples = 200 ms

# ══════════════════════════════════════════════════════════════════════════════
#  V3 — App Server
# ══════════════════════════════════════════════════════════════════════════════
V3_HOST               = "0.0.0.0"
V3_PORT               = 5000

# ══════════════════════════════════════════════════════════════════════════════
#  V5 — EMA Voter Tuning
# ══════════════════════════════════════════════════════════════════════════════
# FIX-CFG-1: Warmup guard.  Must accumulate this many windows before any
#            commit is allowed.  12 × 200 ms = 2.4 s.
V5_EMA_MIN_WINDOWS    = 12

# FIX-CFG-2: EMA smoothing factor.  Lower = slower to react but more stable.
#            Time constant ≈ 1/alpha steps ≈ 10 × 200 ms = 2 s.
V5_EMA_ALPHA          = 0.10

# FIX-CFG-3: Minimum belief-share (ratio) to allow a commit.
V5_EMA_CONF_MIN       = 0.65

# FIX-CFG-4 (NEW): Minimum absolute belief value for the winning class.
#            Two-gate commit: winner must exceed both this AND V5_EMA_CONF_MIN.
#            Prevents conf=1.0 spurious commits when all beliefs are near-zero.
V5_EMA_MIN_BELIEF_ABS = 0.05

# FIX-CFG-5: REST must beat best non-REST by this margin before decaying.
V5_EMA_REST_MARGIN    = 0.05

# Unchanged EMA parameters
V5_EMA_REST_DECAY     = 0.85
V5_EMA_CONF_HIGH      = 0.80
V5_EMA_CD_SHRINK_RATE = 3.0
V5_EMA_COOLDOWN_MIN   = 0.8
V5_EMA_BELIEF_FLOOR   = 1e-4

# Optional SHA-256 integrity check on saved models (off by default)
V5_MODEL_HASH_CHECK   = False

# ══════════════════════════════════════════════════════════════════════════════
#  Runtime settings (user-configurable via setup_wizard.py)
# ══════════════════════════════════════════════════════════════════════════════
START_SERVER          = True
LOG_LEVEL             = "INFO"

# ══════════════════════════════════════════════════════════════════════════════
#  Apply saved user configuration (config/user_config.json), if any.
#  Safe no-op on a fresh checkout with no wizard run yet.
# ══════════════════════════════════════════════════════════════════════════════
try:
    import sys as _sys
    import config_manager as _config_manager
    _config_manager.apply_overrides(_sys.modules[__name__])
except Exception as _exc:  # pragma: no cover - never block startup on this
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "config: could not apply user overrides (%s)", _exc)