"""
app_server.py — Flask + SocketIO Web Server for NeuroBand v3
Place in: /home/pi/neuroband/app_server.py

Install: pip install flask flask-socketio eventlet

REST API:
  GET  /                        serve web_app/index.html
  GET  /api/status              full system status
  GET  /api/words               word list + counts
  POST /api/words/add           {"word": "WATER"}
  POST /api/words/remove        {"word": "WATER"}
  POST /api/words/clear         {"word": "WATER"}
  POST /api/train/start         {"word": "HELLO", "trials": 40}
  POST /api/train/stop
  GET  /api/train/status

WebSocket events (server → client):
  prediction        {label, confidence, timestamp}
  training_started  {word, trials}
  training_cue      {phase, word, trial, total}
  training_sample   {word, trial, collected, total, progress, knn_ready}
  training_complete {word, collected, counts}
  words_updated     {words, counts}
  status_update     {counts, knn_ready, training}
"""

import time
import threading
import logging

import config as cfg

log = logging.getLogger(__name__)

try:
    from flask import Flask, jsonify, request, send_from_directory
    from flask_socketio import SocketIO
    _FLASK = True
except ImportError:
    _FLASK = False


def create_app(word_mgr, knn, voter, session_mgr, audio):
    if not _FLASK:
        raise ImportError("pip install flask flask-socketio eventlet")

    log.debug("create_app(): initialising Flask app and SocketIO")
    app      = Flask(__name__, static_folder="web_app")
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
    log.debug("create_app(): Flask + SocketIO ready")

    def _emit(event, data):
        log.debug("_emit(): event='%s' data=%s", event, data)
        socketio.emit(event, data)

    # ── Serve UI ──────────────────────────────────────────────────────────────
    @app.route("/")
    def index():
        log.debug("GET / — serving web_app/index.html")
        return send_from_directory("web_app", "index.html")

    # ── Status ────────────────────────────────────────────────────────────────
    @app.route("/api/status")
    def api_status():
        log.debug("GET /api/status")
        lead, lead_conf = voter.confidence_now()
        status = {
            "words":        word_mgr.words,
            "counts":       word_mgr.sample_counts(),
            "total":        word_mgr.total_samples(),
            "knn":          knn.status(),
            "training":     session_mgr.is_busy(),
            "leading":      lead,
            "confidence":   round(lead_conf, 3),
            "thresholds": {
                "min_confidence": cfg.V3_CONFIDENCE_MIN,
                "cooldown":       cfg.V3_COOLDOWN_SECONDS,
                "vote_window":    cfg.V3_VOTE_WINDOW,
            },
        }
        log.debug("GET /api/status: returning knn_ready=%s training=%s lead='%s' conf=%.2f",
                  status["knn"]["fitted"], status["training"], lead, lead_conf)
        return jsonify(status)

    # ── Words ─────────────────────────────────────────────────────────────────
    @app.route("/api/words")
    def api_words():
        log.debug("GET /api/words: words=%s", word_mgr.words)
        return jsonify({"words": word_mgr.words, "counts": word_mgr.sample_counts()})

    @app.route("/api/words/add", methods=["POST"])
    def api_add():
        word = (request.json or {}).get("word", "").strip()
        log.debug("POST /api/words/add: word='%s'", word)
        ok, msg = word_mgr.add_word(word)
        if ok:
            log.debug("api_add(): word '%s' added — pregenerating audio and updating k-NN", word)
            audio.pregenerate([word])
            knn.update(word_mgr)
        else:
            log.debug("api_add(): word '%s' rejected — %s", word, msg)
        _emit("words_updated", word_mgr.status())
        return jsonify({"ok": ok, "msg": msg, "words": word_mgr.words})

    @app.route("/api/words/remove", methods=["POST"])
    def api_remove():
        word = (request.json or {}).get("word", "").strip()
        log.debug("POST /api/words/remove: word='%s'", word)
        ok, msg = word_mgr.remove_word(word)
        if ok:
            log.debug("api_remove(): word '%s' removed — updating k-NN", word)
            knn.update(word_mgr)
        _emit("words_updated", word_mgr.status())
        return jsonify({"ok": ok, "msg": msg, "words": word_mgr.words})

    @app.route("/api/words/clear", methods=["POST"])
    def api_clear():
        word = (request.json or {}).get("word", "").strip()
        log.debug("POST /api/words/clear: word='%s'", word)
        ok, msg = word_mgr.clear_word(word)
        if ok:
            log.debug("api_clear(): data cleared for '%s' — updating k-NN", word)
            knn.update(word_mgr)
        return jsonify({"ok": ok, "msg": msg})

    # ── Training ──────────────────────────────────────────────────────────────
    @app.route("/api/train/start", methods=["POST"])
    def api_train_start():
        body   = request.json or {}
        word   = body.get("word", "").strip().upper()
        trials = max(5, min(int(body.get("trials", cfg.V3_TARGET_TRIALS)), 100))
        log.debug("POST /api/train/start: word='%s' trials=%d", word, trials)
        ok, msg = session_mgr.start(word, trials, _emit)
        log.debug("api_train_start(): ok=%s msg='%s'", ok, msg)
        return jsonify({"ok": ok, "msg": msg})

    @app.route("/api/train/stop", methods=["POST"])
    def api_train_stop():
        log.debug("POST /api/train/stop")
        result = session_mgr.stop()
        log.debug("api_train_stop(): %s", result)
        return jsonify({"ok": True, "msg": result})

    @app.route("/api/train/status")
    def api_train_status():
        busy = session_mgr.is_busy()
        log.debug("GET /api/train/status: busy=%s", busy)
        return jsonify({"busy": busy, "counts": word_mgr.sample_counts()})

    # ── WebSocket ─────────────────────────────────────────────────────────────
    @socketio.on("connect")
    def on_connect():
        log.info("Client connected")
        log.debug("on_connect(): emitting initial status_update")
        socketio.emit("status_update", {
            "counts":   word_mgr.sample_counts(),
            "knn_ready": knn.is_ready,
            "training": session_mgr.is_busy(),
        })

    # Periodic broadcast every 5s
    def _broadcast():
        log.debug("_broadcast(): broadcast thread started")
        while True:
            time.sleep(5)
            try:
                payload = {
                    "counts":    word_mgr.sample_counts(),
                    "knn_ready": knn.is_ready,
                    "training":  session_mgr.is_busy(),
                }
                log.debug("_broadcast(): emitting status_update — knn_ready=%s training=%s",
                          payload["knn_ready"], payload["training"])
                socketio.emit("status_update", payload)
            except Exception:
                pass

    threading.Thread(target=_broadcast, daemon=True, name="Broadcast").start()
    log.debug("create_app(): app creation complete")
    return app, socketio