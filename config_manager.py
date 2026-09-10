"""
config_manager.py — NeuroBand User Configuration & Setup State
Place in: /home/pi/neuroband/config_manager.py

Owns two small JSON files, kept deliberately separate from config.py's
hardware/tuning constants:

  config/user_config.json   — device / audio / EEG-hardware-snapshot /
                               runtime settings chosen via the setup wizard.
                               Vocabulary itself stays owned by config.py's
                               existing load_words()/save_words()
                               (models/v3_words.json) — this file only
                               records the wizard's *last confirmed* word
                               list for state-machine bookkeeping.

  config/setup_state.json   — the setup progress state machine:
                               NOT_CONFIGURED -> CONFIGURED -> TRAINED -> READY

Design notes:
  - config.py remains the single source of truth for values that require a
    code change to alter safely (GPIO pins, filter constants, classifier
    tuning). This module only manages values that are meant to be
    user-editable at runtime.
  - apply_overrides() is called once, early, by config.py itself so that
    every other module that does `import config` transparently sees the
    user's saved choices without needing to know config_manager exists.
  - Writes are atomic (write to temp file, then rename) so a power loss
    mid-write can't corrupt either JSON file.
"""

import os
import json
import tempfile
import logging

log = logging.getLogger(__name__)

CONFIG_DIR         = "config"
USER_CONFIG_PATH   = os.path.join(CONFIG_DIR, "user_config.json")
SETUP_STATE_PATH   = os.path.join(CONFIG_DIR, "setup_state.json")

# ══════════════════════════════════════════════════════════════════════════════
#  Setup state machine
# ══════════════════════════════════════════════════════════════════════════════
NOT_CONFIGURED = "NOT_CONFIGURED"
CONFIGURED     = "CONFIGURED"
TRAINED        = "TRAINED"
READY          = "READY"

_VALID_STATES = (NOT_CONFIGURED, CONFIGURED, TRAINED, READY)

DEFAULT_USER_CONFIG = {
    "device": {
        "name": "NeuroBand",
    },
    "runtime": {
        "log_level":     "INFO",
        "start_server":  True,
        "simulate":      False,
    },
    "audio": {
        "engine": "espeak-ng",
        "voice":  "en",
        "speed":  160,   # words per minute, espeak-ng -s
        "pitch":  50,    # 0-99, espeak-ng -p
    },
    "eeg": {
        # Snapshot of the hardware config the user confirmed in the wizard.
        # These mirror config.py's defaults; changing them here does NOT
        # rewire GPIOs by itself — it's a record of what was reviewed.
        "channels":        None,   # filled from config.py default at first run
        "sample_rate":     None,
        "adc_data_pin_ch0": None,
        "adc_data_pin_ch1": None,
        "adc_clk_pin":      None,
    },
}


def _atomic_write(path: str, data: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _read_json(path: str, default: dict) -> dict:
    if not os.path.exists(path):
        return dict(default)
    try:
        with open(path) as f:
            data = json.load(f)
        merged = dict(default)
        merged.update(data)
        return merged
    except Exception as exc:
        log.warning("failed to read %s (%s) — using defaults", path, exc)
        return dict(default)


# ── User config ──────────────────────────────────────────────────────────────
def load_user_config() -> dict:
    return _read_json(USER_CONFIG_PATH, DEFAULT_USER_CONFIG)


def save_user_config(cfg: dict):
    _atomic_write(USER_CONFIG_PATH, cfg)
    log.info("user_config.json saved")


def user_config_exists() -> bool:
    return os.path.exists(USER_CONFIG_PATH)


def apply_overrides(config_module):
    """
    Called by config.py at import time. Overlays any saved user_config.json
    values onto the config module's constants. Safe no-op if no user
    config exists yet (first run, before the wizard has ever completed).
    """
    if not user_config_exists():
        return
    cfg = load_user_config()

    eeg = cfg.get("eeg", {})
    if eeg.get("channels") is not None:
        config_module.NUM_CHANNELS = eeg["channels"]
    if eeg.get("sample_rate") is not None:
        config_module.SAMPLE_RATE = eeg["sample_rate"]
        config_module.WINDOW_SAMPLES = config_module.SAMPLE_RATE * config_module.WINDOW_SECONDS
        config_module.BUFFER_SAMPLES = config_module.SAMPLE_RATE * config_module.BUFFER_SECONDS
        config_module.V3_STEP_SAMPLES = config_module.SAMPLE_RATE // 5
    if eeg.get("adc_data_pin_ch0") is not None:
        config_module.ADC_DATA_PIN_CH0 = eeg["adc_data_pin_ch0"]
    if eeg.get("adc_data_pin_ch1") is not None:
        config_module.ADC_DATA_PIN_CH1 = eeg["adc_data_pin_ch1"]
    if eeg.get("adc_clk_pin") is not None:
        config_module.ADC_CLK_PIN = eeg["adc_clk_pin"]

    runtime = cfg.get("runtime", {})
    if "start_server" in runtime:
        config_module.START_SERVER = runtime["start_server"]
    if "log_level" in runtime:
        config_module.LOG_LEVEL = runtime["log_level"]

    log.debug("config_manager: overrides applied from user_config.json")


def snapshot_eeg_defaults(config_module) -> dict:
    """Read current (possibly already-overridden) EEG values from config.py,
    for the wizard to display as 'current values' before asking to keep them."""
    return {
        "channels":         config_module.NUM_CHANNELS,
        "sample_rate":      config_module.SAMPLE_RATE,
        "adc_data_pin_ch0": config_module.ADC_DATA_PIN_CH0,
        "adc_data_pin_ch1": config_module.ADC_DATA_PIN_CH1,
        "adc_clk_pin":      config_module.ADC_CLK_PIN,
    }


# ── Setup state ───────────────────────────────────────────────────────────────
def load_setup_state() -> str:
    if not os.path.exists(SETUP_STATE_PATH):
        return NOT_CONFIGURED
    try:
        with open(SETUP_STATE_PATH) as f:
            data = json.load(f)
        state = data.get("state", NOT_CONFIGURED)
        return state if state in _VALID_STATES else NOT_CONFIGURED
    except Exception as exc:
        log.warning("failed to read setup_state.json (%s) — treating as NOT_CONFIGURED", exc)
        return NOT_CONFIGURED


def save_setup_state(state: str):
    if state not in _VALID_STATES:
        raise ValueError(f"Invalid setup state: {state}")
    _atomic_write(SETUP_STATE_PATH, {"state": state})
    log.info("setup state -> %s", state)


def is_ready() -> bool:
    return load_setup_state() == READY
