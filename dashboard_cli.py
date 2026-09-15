"""
dashboard_cli.py — `-dashboard -server` / `-dashboard -terminal` entry points
Place in: /home/pi/neuroband/dashboard_cli.py

Neither dashboard is reimplemented here:
  - Web dashboard  = the existing app_server.py Flask/SocketIO app
                      (unchanged — create_app() is called as-is).
  - Terminal dashboard = the existing dashboard.py rich-based Dashboard
                      (unchanged — make_dashboard() is called as-is;
                       it was already written to be fed from an inference
                       loop, just never wired up until now).

This module only:
  1. Builds the shared V3 engine (v3_engine.build_v3_engine) that both
     dashboards need (word list, k-NN model, ADC reader, etc.)
  2. Decides whether a server is already running (avoids duplicates)
  3. Detects the real device IP instead of hard-coding one
  4. Runs the inference loop for the terminal dashboard, or hands the
     engine to the unmodified web app for the web dashboard

Starting either dashboard never starts training or retraining on its
own — training only happens if the person explicitly triggers it (a
button in the web UI, or a separate `--train` / `--retrain` command).
"""

import time
import logging

import config
import v3_engine

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Web dashboard  (neuroband -dashboard -server / neuroband -dashboard)
# ══════════════════════════════════════════════════════════════════════════════
def launch_web_dashboard(simulate: bool = False):
    url = v3_engine.dashboard_url()

    if v3_engine.port_in_use(config.V3_HOST, config.V3_PORT):
        print("Web dashboard is already running — not starting a duplicate server.")
        print(f"  → {url}")
        return

    from app_server import create_app  # existing web dashboard, unmodified

    print("Starting web dashboard...")
    engine = v3_engine.build_v3_engine(simulate=simulate, start_reader=True)
    try:
        app, socketio = create_app(
            engine["word_mgr"], engine["knn"], engine["voter"],
            engine["session_mgr"], engine["audio"],
        )
    except ImportError as e:
        print(f"Web dashboard unavailable ({e}).\n"
              "pip install flask flask-socketio eventlet")
        engine["reader"].stop()
        return

    print(f"Web dashboard ready → {url}")
    print("(Ctrl+C to stop. Nothing trains automatically — use the page's "
          "training controls or `--retrain` for that.)\n")
    try:
        socketio.run(
            app, host=config.V3_HOST, port=config.V3_PORT,
            debug=False, use_reloader=False, log_output=False,
            allow_unsafe_werkzeug=True,
        )
    except KeyboardInterrupt:
        pass
    finally:
        engine["reader"].stop()
        engine["audio"].shutdown()


# ══════════════════════════════════════════════════════════════════════════════
#  Terminal dashboard  (neuroband -dashboard -terminal)
# ══════════════════════════════════════════════════════════════════════════════
def launch_terminal_dashboard(simulate: bool = False,
                               skip_quality: bool = False,
                               skip_calib: bool = False):
    from dashboard import make_dashboard  # existing terminal dashboard, unmodified

    engine = v3_engine.build_v3_engine(simulate=simulate, start_reader=True)
    reader, processor  = engine["reader"], engine["processor"]
    extractor          = engine["extractor"]
    calibrator, knn    = engine["calibrator"], engine["knn"]
    voter, audio       = engine["voter"], engine["audio"]

    if not skip_quality:
        v3_engine.run_quality_check(reader, simulate)
    if not skip_calib:
        v3_engine.run_calibration(reader, processor, extractor, calibrator, simulate)

    dash = make_dashboard()
    dash.start()

    win_mgr = v3_engine.WindowManager(reader, config.V3_STEP_SAMPLES)
    n_win = n_cmd = 0

    try:
        while True:
            raw = win_mgr.next()
            if raw is None:
                time.sleep(0.005)
                continue

            clean = processor.process(raw)
            if clean is None:
                continue
            fv = extractor.extract(clean)
            if fv is None:
                continue
            fv_cal = calibrator.transform(fv)

            proba           = knn.predict_proba_dict(fv_cal)
            winner, conf     = voter.push(proba)
            lead, lead_conf  = voter.confidence_now()
            n_win           += 1

            if winner:
                n_cmd += 1
                audio.speak(winner)

            dash.update(
                clean_window=clean, features=fv, label=lead,
                confidence=lead_conf, quality="GOOD",
                n_windows=n_win, n_commands=n_cmd,
            )
    except KeyboardInterrupt:
        pass
    finally:
        dash.stop()
        reader.stop()
        audio.shutdown()
        print(f"\nSession ended — windows={n_win}  commands={n_cmd}")
