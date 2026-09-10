"""
setup_wizard.py — NeuroBand First-Time Setup / Reconfiguration Wizard
Place in: /home/pi/neuroband/setup_wizard.py

Drives the interactive configuration flow described in the setup spec:

  [1/6] Device configuration
  [2/6] Audio configuration     (espeak-ng voice / speed / pitch)
  [3/6] EEG hardware review     (display current, confirm — no blind GPIO edits)
  [4/6] Vocabulary              (thin wrapper over word_manager.WordManager)
  [5/6] Training                (drives pattern_trainer.SessionManager)
  [6/6] Calibration             (resting-baseline capture, same as main.py)

Entry points (see main.py --setup / --voice / --retrain):
    run_full_wizard(simulate=False)   — full first-run flow, ends in READY
    run_voice_only()                  — just the audio step, no retraining
    run_retrain(simulate=False)       — vocabulary + training + calibration only

State handling:
    Uses config_manager's NOT_CONFIGURED -> CONFIGURED -> TRAINED -> READY
    state machine so a power loss mid-wizard resumes at the right step
    instead of silently claiming setup succeeded.

Config vs. training — kept separate on purpose:
    Changing the voice/speed/pitch or device name never touches the
    trained model. Changing vocabulary only triggers *new* training for
    the words that don't have data yet; words already trained keep their
    samples.
"""

import os
import sys
import time
import logging

import config
import config_manager
from word_manager import WordManager
from audio_output  import list_voices, speak_direct, AudioOutput

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Small terminal helpers
# ══════════════════════════════════════════════════════════════════════════════
def _rule(title=""):
    print("\n" + "─" * 36)
    if title:
        print(title)
        print("─" * 36)


def _confirm(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    ans = input(f"{prompt} {suffix}: ").strip().lower()
    if not ans:
        return default
    return ans.startswith("y")


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    ans = input(f"{prompt}{suffix}: ").strip()
    return ans or default


def _ask_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print("  Please enter a whole number.")


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1 — Device configuration
# ══════════════════════════════════════════════════════════════════════════════
def step_device(cfg: dict) -> dict:
    _rule("1. DEVICE CONFIGURATION")
    current = cfg["device"]["name"]
    print(f"Device name: {current}")
    if not _confirm("Keep?"):
        cfg["device"]["name"] = _ask("New device name", current)
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2 — Audio configuration
# ══════════════════════════════════════════════════════════════════════════════
def step_audio(cfg: dict) -> dict:
    _rule("2. AUDIO CONFIGURATION")
    print("Audio engine: eSpeak-NG\n")

    voices = list_voices()
    if not voices:
        print("WARNING: espeak-ng not found or no voices detected.")
        print("Install with: sudo apt install espeak-ng")
        print("Keeping existing audio settings.\n")
        return cfg

    print("Detecting available voices...\n")
    for i, v in enumerate(voices, 1):
        print(f"{i:>3}. {v['code']:<10} {v['name']}")

    current_voice = cfg["audio"]["voice"]
    default_idx = next((i for i, v in enumerate(voices, 1) if v["code"] == current_voice), 1)

    while True:
        raw = input(f"\nSelect voice [1-{len(voices)}] (current: {current_voice}): ").strip()
        if not raw:
            chosen = voices[default_idx - 1]["code"]
            break
        try:
            idx = int(raw)
            if 1 <= idx <= len(voices):
                chosen = voices[idx - 1]["code"]
                break
        except ValueError:
            pass
        print("  Invalid selection, try again.")

    speed = _ask_int("Speech speed (words/min)", cfg["audio"]["speed"])
    pitch = _ask_int("Speech pitch (0-99)", cfg["audio"]["pitch"])

    print("\nTesting selected voice...")
    ok = speak_direct(f"{cfg['device']['name']} is ready.", voice=chosen, speed=speed, pitch=pitch)
    if not ok:
        print("  (Could not play audio here — continuing anyway.)")

    if _confirm("Keep this voice?"):
        cfg["audio"]["voice"] = chosen
        cfg["audio"]["speed"] = speed
        cfg["audio"]["pitch"] = pitch
    else:
        print("  Voice change discarded — keeping previous settings.")

    return cfg


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 — EEG hardware review (display + confirm only; no blind GPIO edits)
# ══════════════════════════════════════════════════════════════════════════════
def step_eeg(cfg: dict) -> dict:
    _rule("3. EEG CONFIGURATION")
    current = config_manager.snapshot_eeg_defaults(config)

    print(f"Channels: {current['channels']}")
    print(f"Sample rate: {current['sample_rate']} Hz")
    print(f"ADC 1 data GPIO: {current['adc_data_pin_ch0']}")
    print(f"ADC 2 data GPIO: {current['adc_data_pin_ch1']}")
    print(f"ADC clock GPIO: {current['adc_clk_pin']}")

    if _confirm("\nKeep these settings?"):
        cfg["eeg"] = current
        return cfg

    print("\nGPIO wiring is fixed by the hardware build and isn't editable here.")
    print("Only channel count and sample rate can be adjusted.")
    current["channels"]    = _ask_int("Channels", current["channels"])
    current["sample_rate"] = _ask_int("Sample rate (Hz)", current["sample_rate"])
    cfg["eeg"] = current
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4 — Vocabulary (wraps WordManager; does not reimplement it)
# ══════════════════════════════════════════════════════════════════════════════
def step_vocabulary(word_mgr: WordManager) -> list:
    _rule("4. VOCABULARY")
    print(f"Available vocabulary slots: {config.V3_MAX_WORDS}\n")
    print("Current words:")
    for i, w in enumerate(word_mgr.words, 1):
        print(f"{i}. {w}")

    if not _confirm("\nAdd/change words?", default=False):
        return word_mgr.words

    while True:
        print(f"\nCurrent words: {word_mgr.words}")
        action = _ask("(a)dd / (r)emove / (d)one", "d").lower()
        if action.startswith("d"):
            break
        elif action.startswith("a"):
            word = _ask("Word to add").strip()
            if word:
                ok, msg = word_mgr.add_word(word)
                print(f"  {msg}")
        elif action.startswith("r"):
            word = _ask("Word to remove").strip()
            if word:
                ok, msg = word_mgr.remove_word(word)
                print(f"  {msg}")

    return word_mgr.words


# ══════════════════════════════════════════════════════════════════════════════
#  Step 5 — Training (drives pattern_trainer.SessionManager)
# ══════════════════════════════════════════════════════════════════════════════
def step_training(word_mgr: WordManager, simulate: bool) -> bool:
    _rule("5. TRAINING")

    ready = set(word_mgr.ready_words())
    pending = [w for w in word_mgr.words if w not in ready]

    if not pending:
        print("All words already have training data. Skipping training.")
        return True

    print(f"Words needing training data: {', '.join(pending)}\n")
    if not _confirm(f"Start EEG training for {len(pending)} word(s)?"):
        print("Training skipped — device will not be usable until trained.")
        return False

    from adc_reader import ADCReader
    from classifier import KNNClassifier
    from pattern_trainer import SessionManager

    reader = ADCReader(simulate=simulate, mode="v3", word=pending[0], word_list=word_mgr.words)
    reader.start()
    knn = KNNClassifier()
    session_mgr = SessionManager(reader, word_mgr, knn, simulate=simulate)

    try:
        for word in pending:
            print(f"\n--- Training '{word}' ---")
            print("Think the word clearly on each cue.\n")

            progress = {"trial": 0, "total": config.V3_TARGET_TRIALS}

            def _emit(event, data, _p=progress):
                if event == "training_cue":
                    _p["trial"], _p["total"] = data["trial"], data["total"]
                    print(f"\r  [{_p['trial']}/{_p['total']}] THINK: {word}   ", end="", flush=True)
                elif event == "training_complete":
                    print(f"\n  '{word}' training complete — {data.get('collected', '?')} samples.")

            ok, msg = session_mgr.start(word, config.V3_TARGET_TRIALS, _emit)
            if not ok:
                print(f"  Could not start training for '{word}': {msg}")
                continue

            while session_mgr._session and session_mgr._session.is_running():
                time.sleep(0.2)
            print()
    finally:
        reader.stop()

    return True


# ══════════════════════════════════════════════════════════════════════════════
#  Step 6 — Calibration (mirrors main.py's resting-baseline capture)
# ══════════════════════════════════════════════════════════════════════════════
def step_calibration(simulate: bool) -> bool:
    _rule("6. CALIBRATION")

    if simulate:
        print("Calibration skipped (simulation mode).")
        return True

    from adc_reader import ADCReader
    from signal_processing import SignalProcessor
    from feature_extraction import FeatureExtractor
    from calibration import Calibrator
    from quality_check import QualityChecker
    import numpy as np
    import pickle

    reader = ADCReader(simulate=simulate, mode="v3")
    reader.start()
    try:
        print("Checking signal quality...")
        report = QualityChecker().check(reader)
        report.print()

        print(f"\nSit still, close your eyes. Calibrating for {config.CALIBRATION_DURATION}s...")
        for i in range(3, 0, -1):
            print(f"  {i}...", flush=True)
            time.sleep(1)
        print("  GO!\n")

        processor = SignalProcessor()
        extractor = FeatureExtractor()
        feats, deadline = [], time.time() + config.CALIBRATION_DURATION
        last_s, step = None, config.SAMPLE_RATE // 4

        while time.time() < deadline:
            avail = reader.samples_available()
            if avail < config.WINDOW_SAMPLES:
                time.sleep(0.01); continue
            if last_s is None:
                last_s = avail
            elif avail - last_s < step:
                time.sleep(0.01); continue
            else:
                last_s = avail
            raw = reader.get_latest(config.WINDOW_SAMPLES)
            clean = processor.process(raw)
            if clean is None:
                continue
            fv = extractor.extract(clean)
            if fv is not None:
                feats.append(fv)
            print(f"\r  Calibrating... {int(deadline - time.time())}s ({len(feats)} windows)",
                  end="", flush=True)
        print()

        if len(feats) < 3:
            print("WARNING: calibration failed — not enough clean windows captured.")
            return False

        calibrator = Calibrator()
        X = np.array(feats)
        calibrator._mean = X.mean(axis=0)
        calibrator._std = X.std(axis=0)
        os.makedirs(os.path.dirname(config.CALIBRATION_PATH), exist_ok=True)
        with open(config.CALIBRATION_PATH, "wb") as f:
            pickle.dump({"mean": calibrator._mean, "std": calibrator._std}, f)
        print("Calibration complete.\n")
        return True
    finally:
        reader.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  Orchestration
# ══════════════════════════════════════════════════════════════════════════════
def _banner():
    print("╔══════════════════════════════════╗")
    print("║          NEUROBAND V3            ║")
    print("║        First-Time Setup          ║")
    print("╚══════════════════════════════════╝")
    print("\nWelcome to NeuroBand. Let's configure your device.")


def run_full_wizard(simulate: bool = False) -> bool:
    """Full first-run flow: device -> audio -> eeg -> vocabulary -> training -> calibration."""
    _banner()
    cfg = config_manager.load_user_config()

    # Fill EEG defaults from config.py on very first run.
    if cfg["eeg"].get("channels") is None:
        cfg["eeg"] = config_manager.snapshot_eeg_defaults(config)

    cfg = step_device(cfg)
    cfg = step_audio(cfg)
    cfg = step_eeg(cfg)
    config_manager.save_user_config(cfg)
    config_manager.apply_overrides(config)
    config_manager.save_setup_state(config_manager.CONFIGURED)

    word_mgr = WordManager()
    step_vocabulary(word_mgr)

    trained_ok = step_training(word_mgr, simulate)
    if trained_ok:
        config_manager.save_setup_state(config_manager.TRAINED)

    calibrated_ok = step_calibration(simulate)
    if trained_ok and calibrated_ok:
        config_manager.save_setup_state(config_manager.READY)

    _rule()
    print("        NEUROBAND ONLINE" if (trained_ok and calibrated_ok) else "  SETUP INCOMPLETE")
    print(f"Voice: {cfg['audio']['voice']}")
    print(f"Words: {len(word_mgr.words)}")
    print(f"EEG: {cfg['eeg']['channels']} channels @ {cfg['eeg']['sample_rate']} Hz")
    return trained_ok and calibrated_ok


def run_voice_only():
    """python main.py --voice — reopen just the audio configuration step."""
    cfg = config_manager.load_user_config()
    cfg = step_audio(cfg)
    config_manager.save_user_config(cfg)
    config_manager.apply_overrides(config)
    print("\nVoice settings saved. No retraining needed.")


def run_retrain(simulate: bool = False):
    """python main.py --retrain — vocabulary + training + calibration only."""
    word_mgr = WordManager()
    step_vocabulary(word_mgr)
    config_manager.save_setup_state(config_manager.CONFIGURED)

    ok = step_training(word_mgr, simulate)
    if ok:
        config_manager.save_setup_state(config_manager.TRAINED)
    if ok and step_calibration(simulate):
        config_manager.save_setup_state(config_manager.READY)
        print("\nRetraining complete — NeuroBand is ready.")
    else:
        print("\nRetraining incomplete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_full_wizard(simulate="--simulate" in sys.argv)
