"""
v3_engine.py — Shared NeuroBand V3 engine construction
Place in: /home/pi/neuroband/v3_engine.py

Extracted from main.py so that main.py's own run_v3(), the setup wizard,
and the new dashboard CLI (dashboard_cli.py) can all build the exact same
object graph and reuse the exact same quality-check/calibration logic
instead of each reimplementing it. This is plumbing only — no new
behavior, no new dashboard implementation.

Also owns small network helpers (_get_local_ip / port_in_use) used to:
  - detect the device's real LAN IP instead of a hard-coded address
  - detect an already-running Flask/SocketIO server so callers never
    launch a duplicate one
"""

import time
import socket
import logging

import config

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Network helpers
# ══════════════════════════════════════════════════════════════════════════════
def get_local_ip() -> str:
    """
    Best-effort detection of this device's LAN IP. Never hard-coded —
    tries the UDP-connect trick first (no packet actually sent), then
    hostname resolution, then falls back to 'localhost'.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        pass
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    return "localhost"


def port_in_use(host: str, port: int) -> bool:
    """True if something is already listening on host:port."""
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((probe_host, port)) == 0


def dashboard_url() -> str:
    return f"http://{get_local_ip()}:{config.V3_PORT}"


# ══════════════════════════════════════════════════════════════════════════════
#  Sliding window over the ADC ring buffer
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


# ══════════════════════════════════════════════════════════════════════════════
#  Quality check / calibration (identical to main.py's original logic)
# ══════════════════════════════════════════════════════════════════════════════
def run_quality_check(reader, simulate):
    from quality_check import QualityChecker
    if simulate:
        print("Quality check skipped (simulation).\n")
        return "GOOD"
    print("\n[1/2] Signal quality check...")
    report = QualityChecker().check(reader)
    report.print()
    return report.overall


def run_calibration(reader, processor, extractor, calibrator, simulate):
    import os
    import pickle
    import numpy as np

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
#  Engine construction — the shared V3 object graph
# ══════════════════════════════════════════════════════════════════════════════
def build_v3_engine(simulate: bool = False, start_reader: bool = True) -> dict:
    """
    Build the shared V3 object graph: word_mgr, knn, voter, processor,
    extractor, calibrator, audio, reader, session_mgr.

    Does NOT run quality-check, calibration, the inference loop, or any
    training — callers (run_v3, the dashboard CLI, etc.) decide what to
    do with the returned engine. KNNClassifier loads any previously
    saved model from disk automatically, so knn.is_ready reflects real
    on-disk state even before any inference has run in this process.
    """
    from adc_reader         import ADCReader
    from signal_processing  import SignalProcessor
    from feature_extraction import FeatureExtractor
    from calibration        import Calibrator
    from classifier         import KNNClassifier, EMAVoter
    from audio_output       import AudioOutput
    from word_manager       import WordManager
    from pattern_trainer    import SessionManager

    word_mgr = WordManager()
    knn      = KNNClassifier()
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
    if start_reader:
        reader.start()

    session_mgr = SessionManager(reader, word_mgr, knn, voter=voter, simulate=simulate)

    return {
        "word_mgr":    word_mgr,
        "knn":         knn,
        "voter":       voter,
        "processor":   processor,
        "extractor":   extractor,
        "calibrator":  calibrator,
        "audio":       audio,
        "reader":      reader,
        "session_mgr": session_mgr,
    }
