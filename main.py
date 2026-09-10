"""
main.py — Unified NeuroBand Entry Point (v2 + v3)
Place in: /home/pi/neuroband/main.py

Default mode is V3. Use --v2 to run legacy pipeline.

V3 usage:
  python main.py                        # v3 inference + web app
  python main.py --simulate             # v3 simulation, no hardware
  python main.py --no-calib             # skip calibration
  python main.py --no-quality           # skip signal quality check
  python main.py --no-server            # no web app, terminal only

V2 usage (legacy fallback):
  python main.py --v2                   # v2 inference with LDA
  python main.py --v2 --simulate        # v2 simulation
  python main.py --v2 --train YES       # collect v2 training data
  python main.py --v2 --train NO
  python main.py --v2 --train REST
  python main.py --v2 --no-dash         # disable pygame GUI

Startup sequence (v3):
  1. ADC reader start
  2. Signal quality check (5s)
  3. Auto-calibration (30s)
  4. Flask app server (background thread)
  5. Inference loop — 200ms steps, democracy vote, 3s cooldown

Startup sequence (v2):
  1. Pygame GUI dashboard
  2. ADC reader start
  3. Signal quality check
  4. Auto-calibration
  5. Inference loop — 500ms steps, 3-window majority vote
"""

import os
import sys
import time
import signal
import logging
import argparse
import threading
import collections
import numpy as np

import config
import config_manager
from adc_reader        import ADCReader
from signal_processing import SignalProcessor
from feature_extraction import FeatureExtractor
from calibration       import Calibrator
from quality_check     import QualityChecker
from classifier        import UnifiedClassifier, KNNClassifier, EMAVoter, LDAClassifier
from audio_output      import AudioOutput
from word_manager      import WordManager

# ── Logging ────────────────────────────────────────────────────────────────────
os.makedirs(config.LOG_DIR,        exist_ok=True)
os.makedirs("models",              exist_ok=True)
os.makedirs(config.V3_DATASET_DIR, exist_ok=True)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers= [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(config.LOG_DIR, "neuroband.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("main")


# ══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ══════════════════════════════════════════════════════════════════════════════
class WindowManager:
    """Sliding window over ADCReader ring buffer."""
    def __init__(self, reader, step):
        self._reader = reader
        self._step   = step
        self._last   = None

    def next(self):
        avail = self._reader.samples_available()
        if avail < config.WINDOW_SAMPLES:
            return None
        if self._last is None:
            self._last = avail
            return self._reader.get_latest(config.WINDOW_SAMPLES)
        if avail - self._last < self._step:
            return None
        self._last = avail
        return self._reader.get_latest(config.WINDOW_SAMPLES)


def _run_quality_check(reader, simulate):
    if simulate:
        print("Quality check skipped (simulation).\n")
        return "GOOD"
    print("\n[1/2] Signal quality check...")
    report = QualityChecker().check(reader)
    report.print()
    return report.overall


def _run_calibration(reader, processor, extractor, calibrator, simulate):
    if simulate:
        print("Calibration skipped (simulation).\n")
        return
    print("\n[2/2] Auto-calibration (30s) — sit still, close eyes.")
    for i in range(3, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1)
    print("  GO!\n", flush=True)

    feats, deadline = [], time.time() + config.CALIBRATION_DURATION
    last_s, step    = None, config.SAMPLE_RATE // 4

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

        raw   = reader.get_latest(config.WINDOW_SAMPLES)
        clean = processor.process(raw)
        if clean is None: continue
        fv = extractor.extract(clean)
        if fv is not None: feats.append(fv)
        print(f"\r  Calibrating... {int(deadline-time.time())}s "
              f"({len(feats)} windows)", end="", flush=True)

    print()
    if len(feats) >= 3:
        import pickle
        X = np.array(feats)
        calibrator._mean   = X.mean(axis=0)
        calibrator._std    = X.std(axis=0)
        calibrator._loaded = True
        os.makedirs(os.path.dirname(config.CALIBRATION_PATH), exist_ok=True)
        with open(config.CALIBRATION_PATH, "wb") as f:
            pickle.dump({"mean": calibrator._mean, "std": calibrator._std}, f)
        print("Calibration complete.\n")
    else:
        print("WARNING: calibration failed — proceeding without baseline.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  V3 Inference
# ══════════════════════════════════════════════════════════════════════════════
def run_v3(simulate=False, skip_calib=False, skip_quality=False, no_server=False):
    log.info("=== NeuroBand v3 (simulate=%s) ===", simulate)
    config.VERSION = "3"

    word_mgr   = WordManager()
    knn        = KNNClassifier()
    # v5.2: EMAVoter replaces DemocracyVoter; initialise with all word classes + REST
    _ema_classes = list(word_mgr.words) + (["REST"] if "REST" not in word_mgr.words else [])
    voter      = EMAVoter(classes=_ema_classes)
    processor  = SignalProcessor()
    extractor  = FeatureExtractor()
    calibrator = Calibrator()
    audio      = AudioOutput(mode="v3")

    audio.pregenerate(word_mgr.words)

    reader = ADCReader(
        simulate  = simulate,
        mode      = "v3",
        word      = word_mgr.words[0] if word_mgr.words else "HELLO",
        word_list = word_mgr.words,
    )
    reader.start()

    print("DEBUG: before pattern_trainer import", flush=True)
    from pattern_trainer import SessionManager
    print("DEBUG: after import, before init", flush=True)
    session_mgr = SessionManager(reader, word_mgr, knn, voter=voter, simulate=simulate)
    print("DEBUG: SessionManager init done", flush=True)

    _running = [True]
    signal.signal(signal.SIGINT,  lambda s, f: _running.__setitem__(0, False))
    signal.signal(signal.SIGTERM, lambda s, f: _running.__setitem__(0, False))

    # ── Quality + calibration ─────────────────────────────────────────────────
    if not skip_quality:
        _run_quality_check(reader, simulate)
    if not skip_calib:
        _run_calibration(reader, processor, extractor, calibrator, simulate)

    # ── App server ────────────────────────────────────────────────────────────
    push_pred = lambda l, c, v: None

    if not no_server:
        try:
            from app_server import create_app
            app, socketio = create_app(word_mgr, knn, voter, session_mgr, audio)

            def _push(label, conf, votes):
                socketio.emit("prediction", {
                    "label":      label,
                    "confidence": round(conf, 3),
                    "votes":      votes,
                    "timestamp":  time.time(),
                })
            push_pred = _push

            threading.Thread(
                target=lambda: socketio.run(
                    app, host=config.V3_HOST, port=config.V3_PORT,
                    debug=False, use_reloader=False, log_output=False,
                ),
                daemon=True, name="AppServer",
            ).start()
            print(f"Web app → http://192.168.4.1:{config.V3_PORT}\n")
        except ImportError as e:
            print(f"App server disabled ({e}).\n"
                  "pip install flask flask-socketio eventlet\n")

    # ── Inference loop ────────────────────────────────────────────────────────
    log.info("V3 inference loop. Step=%d samples (%.0f ms)",
             config.V3_STEP_SAMPLES,
             config.V3_STEP_SAMPLES / config.SAMPLE_RATE * 1000)

    win_mgr = WindowManager(reader, config.V3_STEP_SAMPLES)
    n_win   = 0
    n_cmd   = 0

    while _running[0]:
        raw = win_mgr.next()
        if raw is None:
            time.sleep(0.005); continue

        clean = processor.process(raw)
        if clean is None: continue
        fv = extractor.extract(clean)
        if fv is None: continue
        fv_cal = calibrator.transform(fv)

        # v5.2: feed soft probabilities to EMAVoter (preserves uncertainty — FIX 15)
        proba                = knn.predict_proba_dict(fv_cal)
        winner, conf         = voter.push(proba)
        lead, lead_conf      = voter.confidence_now()
        votes                = voter.belief_state()
        n_win               += 1

        # Push live state to web app every window
        push_pred(lead, lead_conf, votes)

        if winner:
            n_cmd += 1
            log.info("COMMAND: %s  conf=%.2f", winner, conf)
            print(f"\n>>> {winner}  {conf:.0%}\n", flush=True)
            audio.speak(winner)
        else:
            if n_win % 25 == 0:
                _conf_min = getattr(config, "V5_EMA_CONF_MIN", getattr(config, "V3_CONFIDENCE_MIN", 0.55))
                sup = "(suppressed)" if lead_conf < _conf_min else ""
                print(f"  w={n_win}  lead={lead}  conf={lead_conf:.2f} {sup}",
                      flush=True)

    # ── Teardown ──────────────────────────────────────────────────────────────
    reader.stop()
    audio.shutdown()
    print(f"\nSession ended — windows={n_win}  commands={n_cmd}")
    log.info("V3 session ended: windows=%d commands=%d", n_win, n_cmd)


# ══════════════════════════════════════════════════════════════════════════════
#  V2 EEG Logger  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════
class EEGLogger:
    def __init__(self, label):
        self.label = label
        self._X, self._y = [], []

    def record(self, fv, raw_window=None):
        self._X.append(fv)
        self._y.append(self.label)

    def save(self):
        path = os.path.join(config.LOG_DIR, "training_data.npz")
        if os.path.exists(path):
            prev  = np.load(path, allow_pickle=True)
            X_all = np.concatenate([prev["X"], np.array(self._X)], axis=0)
            y_all = np.concatenate([prev["y"], np.array(self._y)], axis=0)
        else:
            X_all = np.array(self._X, dtype=np.float32)
            y_all = np.array(self._y)
        np.savez(path, X=X_all, y=y_all)
        log.info("Saved %d '%s' samples → %d total",
                 len(self._X), self.label, len(X_all))
        print(f"Saved {len(self._X)} '{self.label}' windows. Total: {len(X_all)}")


# ══════════════════════════════════════════════════════════════════════════════
#  V2 Vote Buffer  (original 3-window majority)
# ══════════════════════════════════════════════════════════════════════════════
class V2VoteBuffer:
    def __init__(self, window=3):
        self._buf = collections.deque(maxlen=window)

    def push(self, label):
        self._buf.append(label)
        if len(self._buf) < self._buf.maxlen:
            return None
        counts = collections.Counter(self._buf)
        winner, count = counts.most_common(1)[0]
        if count >= (self._buf.maxlen // 2 + 1):
            return winner
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  V2 Inference
# ══════════════════════════════════════════════════════════════════════════════
def run_v2(simulate=False, skip_calib=False, skip_quality=False, no_dash=False,
           train_label=None, train_duration=60):
    log.info("=== NeuroBand v2 (simulate=%s) ===", simulate)
    config.VERSION = "2"

    # ── Training mode ─────────────────────────────────────────────────────────
    if train_label:
        _run_v2_training(train_label, train_duration, simulate)
        return

    processor  = SignalProcessor()
    extractor  = FeatureExtractor()
    calibrator = Calibrator()
    audio      = AudioOutput(mode="v2")
    lda        = LDAClassifier()
    voter      = V2VoteBuffer(window=3)

    reader = ADCReader(
        simulate  = simulate,
        mode      = "v2",
        sim_label = "REST",
    )

    # ── GUI dashboard (v2 only) ───────────────────────────────────────────────
    dash = None
    if not no_dash:
        try:
            from gui_dashboard import GUIDashboard
            dash = GUIDashboard()
            dash.start()
        except ImportError:
            log.warning("gui_dashboard not available.")

    _running = [True]
    signal.signal(signal.SIGINT,  lambda s, f: _running.__setitem__(0, False))
    signal.signal(signal.SIGTERM, lambda s, f: _running.__setitem__(0, False))

    reader.start()

    # ── Quality + calibration ─────────────────────────────────────────────────
    quality = "GOOD"
    if not skip_quality:
        quality = _run_quality_check(reader, simulate)
        if quality == "FAIL" and config.QUALITY_BLOCK_ON_FAIL:
            print("Blocked: poor signal quality. Fix electrodes and restart.")
            reader.stop()
            if dash: dash.stop()
            return
    if not skip_calib:
        _run_calibration(reader, processor, extractor, calibrator, simulate)

    # ── Inference loop ────────────────────────────────────────────────────────
    log.info("V2 inference loop. Step=%d samples", config.STEP_SAMPLES)
    win_mgr      = WindowManager(reader, config.STEP_SAMPLES)
    last_command = None
    n_win        = 0
    n_cmd        = 0

    while _running[0]:
        raw = win_mgr.next()
        if raw is None:
            time.sleep(0.01); continue

        clean = processor.process(raw)
        if clean is None: continue
        fv = extractor.extract(clean)
        if fv is None: continue
        fv_cal = calibrator.transform(fv)

        label, conf = lda.predict(fv_cal)
        n_win      += 1

        if dash:
            dash.update(
                clean_window = clean,
                features     = fv,
                label        = label,
                confidence   = conf,
                quality      = quality,
                n_windows    = n_win,
                n_commands   = n_cmd,
            )
        else:
            if n_win % 10 == 0:
                print(f"  [w={n_win}] {label:<5} conf={conf:.2f}", flush=True)

        voted = voter.push(label)
        if voted and voted != "REST" and voted != last_command:
            log.info("V2 COMMAND: %s conf=%.2f", voted, conf)
            audio.speak(voted)
            last_command = voted
            n_cmd       += 1
            print(f"\n>>> {voted}  {conf:.0%}\n", flush=True)
        elif voted == "REST":
            last_command = None

    # ── Teardown ──────────────────────────────────────────────────────────────
    if dash: dash.stop()
    reader.stop()
    audio.shutdown()
    print(f"\nSession ended — windows={n_win}  commands={n_cmd}")
    log.info("V2 session ended: windows=%d commands=%d", n_win, n_cmd)


# ══════════════════════════════════════════════════════════════════════════════
#  V2 Training
# ══════════════════════════════════════════════════════════════════════════════
def _run_v2_training(label, duration_s, simulate):
    log.info("V2 training mode: label=%s duration=%ds", label, duration_s)
    print(f"\nCollecting '{label}' data for {duration_s}s.")
    print("Think about your chosen command clearly.\n")
    for i in range(3, 0, -1):
        print(f"{i}...", flush=True); time.sleep(1)
    print("GO!\n", flush=True)

    reader    = ADCReader(simulate=simulate, mode="v2", sim_label=label)
    processor = SignalProcessor()
    extractor = FeatureExtractor()
    win_mgr   = WindowManager(reader, config.STEP_SAMPLES)
    logger    = EEGLogger(label)

    reader.start()
    deadline = time.time() + duration_s
    n_win    = 0

    _running = [True]
    signal.signal(signal.SIGINT, lambda s, f: _running.__setitem__(0, False))

    while _running[0] and time.time() < deadline:
        raw = win_mgr.next()
        if raw is None:
            time.sleep(0.01); continue
        clean = processor.process(raw)
        if clean is None: continue
        fv = extractor.extract(clean)
        if fv is None: continue
        logger.record(fv)
        n_win += 1
        if n_win % 10 == 0:
            print(f"  {int(deadline-time.time())}s remaining | {n_win} windows",
                  flush=True)

    reader.stop()
    logger.save()
    print("\nDone. Run: python classifier.py --train")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI Entrypoint
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="NeuroBand — v3 default, --v2 for legacy")

    # Mode
    ap.add_argument("--v2",         action="store_true", help="Run v2 legacy mode")
    ap.add_argument("--simulate",   action="store_true", help="Simulate ADC (no hardware)")

    # Shared flags
    ap.add_argument("--no-calib",   action="store_true", help="Skip calibration")
    ap.add_argument("--no-quality", action="store_true", help="Skip signal quality check")

    # V3 flags
    ap.add_argument("--no-server",  action="store_true", help="[v3] Disable web app server")

    # V2 flags
    ap.add_argument("--train",      choices=config.V2_LABELS, metavar="LABEL",
                    help="[v2] Collect training data for LABEL (YES/NO/REST)")
    ap.add_argument("--duration",   type=int, default=60,
                    help="[v2] Training duration in seconds")
    ap.add_argument("--no-dash",    action="store_true",
                    help="[v2] Disable pygame GUI dashboard")

    # Setup wizard flags [v3]
    ap.add_argument("--setup",   action="store_true",
                    help="[v3] Force the full configuration/training/calibration wizard")
    ap.add_argument("--voice",   action="store_true",
                    help="[v3] Reopen just the audio (voice/speed/pitch) configuration step")
    ap.add_argument("--retrain", action="store_true",
                    help="[v3] Reopen vocabulary + training + calibration (keeps device/audio config)")

    args = ap.parse_args()

    if args.v2:
        run_v2(
            simulate     = args.simulate,
            skip_calib   = args.no_calib,
            skip_quality = args.no_quality,
            no_dash      = args.no_dash,
            train_label  = args.train,
            train_duration = args.duration,
        )
    elif args.voice:
        import setup_wizard
        setup_wizard.run_voice_only()
    elif args.retrain:
        import setup_wizard
        setup_wizard.run_retrain(simulate=args.simulate)
    else:
        # First-run (or explicit --setup) detection: don't jump straight into
        # inference until the device has been configured, trained, and
        # calibrated at least once. Safe to resume — setup state is persisted
        # after each stage, so a power loss mid-wizard just re-enters here.
        just_calibrated = False
        if args.setup or not config_manager.is_ready():
            import setup_wizard
            ok = setup_wizard.run_full_wizard(simulate=args.simulate)
            if not ok:
                print("\nSetup did not complete. Run 'python main.py --setup' to resume.")
                sys.exit(1)
            if args.setup:
                # --setup was explicit; don't also launch inference this run.
                sys.exit(0)
            just_calibrated = True  # wizard already captured a fresh baseline
        else:
            _cfg = config_manager.load_user_config()
            print("Starting NeuroBand...\n")
            print("Configuration found \u2713")
            print("Model found \u2713" if os.path.exists(config.V3_KNN_MODEL_PATH) else "Model found \u2717")
            print(f"Voice: {_cfg['audio']['voice']} \u2713\n")

        run_v3(
            simulate     = args.simulate,
            skip_calib   = args.no_calib or just_calibrated,
            skip_quality = args.no_quality,
            no_server    = args.no_server,
        )