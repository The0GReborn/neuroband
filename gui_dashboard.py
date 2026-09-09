"""
gui_dashboard.py -- Pygame Graphical Dashboard for NeuroBand
Place in: /home/pi/neuroband/gui_dashboard.py

Shows in a separate window:
  - Live EEG waveforms (both channels, scrolling)
  - Band power bars (delta/theta/alpha/beta/gamma)
  - Signal quality indicator with color
  - Calibration progress bar + status messages
  - Command output panel (YES/NO/REST with confidence)
  - Session stats (uptime, windows, commands)

Requires: pygame (already installed)
"""

import time
import math
import threading
import collections
import numpy as np
import pygame

import config

# ── Colors ─────────────────────────────────────────────────────────────────────
BLACK       = (10,  10,  20)
DARK_GRAY   = (30,  30,  45)
MID_GRAY    = (60,  60,  80)
LIGHT_GRAY  = (140, 140, 160)
WHITE       = (220, 220, 235)
CYAN        = (0,   200, 220)
MAGENTA     = (200, 80,  220)
GREEN       = (50,  220, 100)
RED         = (220, 60,  60)
YELLOW      = (220, 200, 50)
ORANGE      = (220, 140, 40)
BLUE        = (60,  120, 220)
DARK_GREEN  = (20,  80,  40)
DARK_RED    = (80,  20,  20)
DARK_BLUE   = (20,  40,  80)

BAND_COLORS = {
    "Delta": (80,  100, 220),
    "Theta": (80,  180, 220),
    "Alpha": (50,  220, 100),
    "Beta":  (220, 200, 50),
    "Gamma": (220, 80,  80),
}

LABEL_COLORS = {
    "YES":  GREEN,
    "NO":   RED,
    "REST": YELLOW,
}

QUALITY_COLORS = {
    "GOOD": GREEN,
    "WARN": YELLOW,
    "FAIL": RED,
}

# ── Layout constants ────────────────────────────────────────────────────────────
W, H         = 1100, 700
FPS          = 30
WAVE_HISTORY = 300    # samples to show in waveform


class GUIDashboard:
    """
    Pygame-based graphical dashboard for NeuroBand.

    Runs in a background thread so it never blocks the inference loop.

    Usage:
        dash = GUIDashboard()
        dash.start()
        # in inference loop:
        dash.update(clean_window, features, label, confidence, quality,
                    n_windows, n_commands)
        dash.show_message("Calibrating...", progress=0.5)
        dash.stop()
    """

    def __init__(self):
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None

        # EEG waveform history (in uV)
        self._ch0_wave = collections.deque([0.0] * WAVE_HISTORY, maxlen=WAVE_HISTORY)
        self._ch1_wave = collections.deque([0.0] * WAVE_HISTORY, maxlen=WAVE_HISTORY)

        # Confidence history
        self._conf_hist = collections.deque([0.0] * 60, maxlen=60)

        # Current state
        self._label      = "REST"
        self._conf       = 0.0
        self._quality    = "GOOD"
        self._features   = None
        self._n_windows  = 0
        self._n_commands = 0
        self._start_time = time.time()
        self._last_cmd   = "—"
        self._last_cmd_t = "—"

        # Message overlay (calibration / quality check)
        self._message    = ""
        self._submessage = ""
        self._progress   = -1.0   # -1 = hide progress bar

        # Flash effect on new command
        self._flash_timer = 0
        self._flash_label = ""

    # ── Lifecycle ───────────────────────────────────────────────────────────────
    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._render_loop,
            name="GUIDashboard",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    # ── Data update (called from inference loop) ────────────────────────────────
    def update(self, clean_window, features, label, confidence,
               quality="GOOD", n_windows=0, n_commands=0):
        with self._lock:
            if clean_window is not None:
                # Append latest samples to waveform history (in uV)
                n = min(clean_window.shape[1], 25)
                for v in clean_window[0, -n:]:
                    self._ch0_wave.append(v * 1e6)
                for v in clean_window[1, -n:]:
                    self._ch1_wave.append(v * 1e6)

            self._conf_hist.append(confidence)
            self._label      = label
            self._conf       = confidence
            self._quality    = quality
            self._features   = features.copy() if features is not None else None
            self._n_windows  = n_windows
            self._n_commands = n_commands

            if label != "REST" and label != self._last_cmd:
                self._last_cmd   = label
                self._last_cmd_t = time.strftime("%H:%M:%S")
                self._flash_timer = FPS * 2   # flash for 2 seconds
                self._flash_label = label

    def show_message(self, message: str, submessage: str = "", progress: float = -1.0):
        """Show an overlay message (used during quality check and calibration)."""
        with self._lock:
            self._message    = message
            self._submessage = submessage
            self._progress   = progress

    def clear_message(self):
        with self._lock:
            self._message    = ""
            self._submessage = ""
            self._progress   = -1.0

    # ── Render loop ─────────────────────────────────────────────────────────────
    def _render_loop(self):
        pygame.display.init()
        pygame.font.init()

        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("NeuroBand BCI — Live Monitor")

        # Fonts
        font_lg   = pygame.font.SysFont("Consolas", 22, bold=True)
        font_md   = pygame.font.SysFont("Consolas", 16)
        font_sm   = pygame.font.SysFont("Consolas", 13)
        font_xl   = pygame.font.SysFont("Consolas", 48, bold=True)

        clock = pygame.time.Clock()

        while self._running:
            # Handle window close
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False

            screen.fill(BLACK)

            with self._lock:
                self._draw_frame(screen, font_lg, font_md, font_sm, font_xl)

            pygame.display.flip()
            clock.tick(FPS)

        pygame.display.quit()

    # ── Drawing ─────────────────────────────────────────────────────────────────
    def _draw_frame(self, screen, font_lg, font_md, font_sm, font_xl):
        now = time.time()
        uptime = int(now - self._start_time)
        h, m, s = uptime // 3600, (uptime % 3600) // 60, uptime % 60

        # ── Header bar ──────────────────────────────────────────────────────
        pygame.draw.rect(screen, DARK_BLUE, (0, 0, W, 45))
        self._text(screen, font_lg, "NeuroBand BCI — Live Monitor",
                   (15, 12), WHITE)
        self._text(screen, font_md,
                   f"Uptime: {h:02d}:{m:02d}:{s:02d}   "
                   f"Windows: {self._n_windows}   "
                   f"Commands: {self._n_commands}",
                   (W - 420, 14), LIGHT_GRAY)

        # ── Signal quality badge ─────────────────────────────────────────────
        qcol = QUALITY_COLORS.get(self._quality, WHITE)
        pygame.draw.rect(screen, qcol, (W - 120, 8, 105, 28), border_radius=5)
        self._text(screen, font_md, f"Signal: {self._quality}",
                   (W - 115, 15), BLACK)

        # ── EEG Waveforms (top half, left) ───────────────────────────────────
        wave_x, wave_y, wave_w, wave_h = 15, 55, 680, 280
        self._draw_panel(screen, wave_x, wave_y, wave_w, wave_h, "EEG Waveforms", font_md)

        ch0_y  = wave_y + 30
        ch1_y  = wave_y + 165
        ch_h   = 120

        self._draw_waveform(screen, list(self._ch0_wave),
                            wave_x + 10, ch0_y, wave_w - 20, ch_h,
                            CYAN, "CH0  F3-Cz", font_sm)
        self._draw_waveform(screen, list(self._ch1_wave),
                            wave_x + 10, ch1_y, wave_w - 20, ch_h,
                            MAGENTA, "CH1  F4-Cz", font_sm)

        # ── Band Powers (top half, right) ────────────────────────────────────
        bp_x, bp_y, bp_w, bp_h = 710, 55, 375, 280
        self._draw_panel(screen, bp_x, bp_y, bp_w, bp_h, "Band Powers", font_md)

        bands      = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
        feat_idx   = [5, 6, 7, 8, 9]
        bar_max_w  = 150
        row_h      = 44

        for i, (name, idx) in enumerate(zip(bands, feat_idx)):
            by = bp_y + 40 + i * row_h
            # CH0
            ch0_val = float(self._features[idx])     if self._features is not None else 0
            ch1_val = float(self._features[idx + 12]) if self._features is not None else 0
            mx = max(ch0_val, ch1_val, 1e-30)
            color = BAND_COLORS[name]

            self._text(screen, font_sm, name, (bp_x + 10, by + 4), LIGHT_GRAY)

            # CH0 bar
            w0 = int(ch0_val / mx * bar_max_w)
            pygame.draw.rect(screen, MID_GRAY,  (bp_x + 65, by,     bar_max_w, 14), border_radius=3)
            pygame.draw.rect(screen, color,      (bp_x + 65, by,     w0,        14), border_radius=3)
            self._text(screen, font_sm, "F3", (bp_x + 55, by), LIGHT_GRAY)

            # CH1 bar
            w1 = int(ch1_val / mx * bar_max_w)
            pygame.draw.rect(screen, MID_GRAY,  (bp_x + 65, by + 16, bar_max_w, 14), border_radius=3)
            pygame.draw.rect(screen, color,      (bp_x + 65, by + 16, w1,        14), border_radius=3)
            self._text(screen, font_sm, "F4", (bp_x + 55, by + 16), LIGHT_GRAY)

        # ── Command Output (bottom left) ─────────────────────────────────────
        cmd_x, cmd_y, cmd_w, cmd_h = 15, 345, 400, 220
        self._draw_panel(screen, cmd_x, cmd_y, cmd_w, cmd_h, "Command Output", font_md)

        label_col = LABEL_COLORS.get(self._label, WHITE)

        # Flash background on new command
        if self._flash_timer > 0:
            alpha = min(255, self._flash_timer * 8)
            flash_col = LABEL_COLORS.get(self._flash_label, WHITE)
            flash_surf = pygame.Surface((cmd_w - 4, cmd_h - 4), pygame.SRCALPHA)
            flash_surf.fill((*flash_col, min(60, alpha)))
            screen.blit(flash_surf, (cmd_x + 2, cmd_y + 2))
            self._flash_timer -= 1

        # Current label (large)
        self._text(screen, font_xl, self._label,
                   (cmd_x + 20, cmd_y + 45), label_col)

        # Confidence bar
        conf_bar_w = cmd_w - 40
        pygame.draw.rect(screen, MID_GRAY,
                         (cmd_x + 20, cmd_y + 115, conf_bar_w, 18), border_radius=4)
        filled = int(self._conf * conf_bar_w)
        pygame.draw.rect(screen, label_col,
                         (cmd_x + 20, cmd_y + 115, filled, 18), border_radius=4)
        self._text(screen, font_md,
                   f"Confidence: {self._conf*100:.0f}%",
                   (cmd_x + 20, cmd_y + 140), WHITE)

        self._text(screen, font_sm,
                   f"Last command: {self._last_cmd}  at {self._last_cmd_t}",
                   (cmd_x + 20, cmd_y + 170), LIGHT_GRAY)

        # ── Confidence History (bottom middle) ───────────────────────────────
        ch_x, ch_y, ch_w, ch_h2 = 425, 345, 280, 220
        self._draw_panel(screen, ch_x, ch_y, ch_w, ch_h2, "Confidence History", font_md)
        self._draw_sparkline(screen, list(self._conf_hist),
                             ch_x + 10, ch_y + 35, ch_w - 20, ch_h2 - 50,
                             GREEN, font_sm)

        # ── Session Info (bottom right) ──────────────────────────────────────
        si_x, si_y, si_w, si_h = 715, 345, 370, 220
        self._draw_panel(screen, si_x, si_y, si_w, si_h, "Session Info", font_md)

        info_lines = [
            ("Sample Rate",  f"{config.SAMPLE_RATE} Hz"),
            ("Window Size",  f"{config.WINDOW_SECONDS}s ({config.WINDOW_SAMPLES} pts)"),
            ("Channels",     f"{config.NUM_CHANNELS}  (F3-Cz, F4-Cz)"),
            ("Filter",       f"{config.BANDPASS_LOW}-{config.BANDPASS_HIGH} Hz BP"),
            ("Notch",        f"{config.NOTCH_FREQ} Hz"),
            ("Threshold",    f"{config.CONFIDENCE_THRESH*100:.0f}% confidence"),
            ("Model",        "LDA Classifier"),
            ("Baseline",     "Loaded" if self._progress == -1.0 else "Calibrating..."),
        ]

        for i, (k, v) in enumerate(info_lines):
            iy = si_y + 35 + i * 22
            self._text(screen, font_sm, f"{k}:", (si_x + 12, iy), LIGHT_GRAY)
            self._text(screen, font_sm, v,        (si_x + 180, iy), WHITE)

        # ── Message Overlay (calibration / quality check) ────────────────────
        if self._message:
            self._draw_overlay(screen, font_lg, font_md, font_sm)

    # ── Overlay for calibration / quality messages ───────────────────────────
    def _draw_overlay(self, screen, font_lg, font_md, font_sm):
        # Semi-transparent background
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))

        # Box
        bw, bh = 600, 300
        bx, by = (W - bw) // 2, (H - bh) // 2
        pygame.draw.rect(screen, DARK_GRAY, (bx, by, bw, bh), border_radius=12)
        pygame.draw.rect(screen, BLUE,      (bx, by, bw, bh), width=2, border_radius=12)

        # Main message
        self._text_centered(screen, font_lg, self._message, W // 2, by + 50, WHITE)

        # Sub message
        if self._submessage:
            self._text_centered(screen, font_md, self._submessage, W // 2, by + 95, LIGHT_GRAY)

        # Progress bar
        if self._progress >= 0:
            pb_w, pb_h = 500, 24
            pb_x, pb_y = (W - pb_w) // 2, by + 150
            pygame.draw.rect(screen, MID_GRAY, (pb_x, pb_y, pb_w, pb_h), border_radius=6)
            filled = int(self._progress * pb_w)
            color  = GREEN if self._progress > 0.8 else (YELLOW if self._progress > 0.4 else BLUE)
            pygame.draw.rect(screen, color, (pb_x, pb_y, filled, pb_h), border_radius=6)
            pct = int(self._progress * 100)
            self._text_centered(screen, font_md, f"{pct}%", W // 2, pb_y + 35, WHITE)

        # Hint
        self._text_centered(screen, font_sm, "Please wait...", W // 2, by + 230, LIGHT_GRAY)

    # ── Drawing helpers ──────────────────────────────────────────────────────
    def _draw_panel(self, screen, x, y, w, h, title, font):
        pygame.draw.rect(screen, DARK_GRAY, (x, y, w, h), border_radius=8)
        pygame.draw.rect(screen, MID_GRAY,  (x, y, w, h), width=1, border_radius=8)
        self._text(screen, font, title, (x + 10, y + 6), CYAN)

    def _draw_waveform(self, screen, data, x, y, w, h, color, label, font):
        self._text(screen, font, label, (x, y), color)
        cy = y + h // 2 + 15
        if len(data) < 2:
            return
        mn, mx = min(data), max(data)
        span = max(mx - mn, 0.1)

        pts = []
        for i, v in enumerate(data):
            px = x + int(i / len(data) * w)
            py = cy - int((v - mn - span / 2) / span * (h - 20))
            pts.append((px, py))

        if len(pts) > 1:
            pygame.draw.lines(screen, color, False, pts, 1)

        # Zero line
        pygame.draw.line(screen, MID_GRAY, (x, cy), (x + w, cy), 1)

        # Scale label
        if span > 0:
            self._text(screen, font,
                       f"+{mx:.1f}uV", (x + w - 70, y + 18), LIGHT_GRAY)
            self._text(screen, font,
                       f"{mn:.1f}uV",  (x + w - 70, y + h - 5), LIGHT_GRAY)

    def _draw_sparkline(self, screen, data, x, y, w, h, color, font):
        if len(data) < 2:
            return
        mn, mx = 0.0, 1.0
        pts = []
        for i, v in enumerate(data):
            px = x + int(i / len(data) * w)
            py = y + h - int((v - mn) / (mx - mn + 1e-9) * h)
            py = max(y, min(y + h, py))
            pts.append((px, py))
        if len(pts) > 1:
            pygame.draw.lines(screen, color, False, pts, 2)
        # Grid line at 70% confidence threshold
        thresh_y = y + h - int(config.CONFIDENCE_THRESH * h)
        pygame.draw.line(screen, YELLOW, (x, thresh_y), (x + w, thresh_y), 1)
        self._text(screen, font, f"{config.CONFIDENCE_THRESH*100:.0f}%",
                   (x + w - 30, thresh_y - 14), YELLOW)

    @staticmethod
    def _text(screen, font, text, pos, color):
        surf = font.render(str(text), True, color)
        screen.blit(surf, pos)

    @staticmethod
    def _text_centered(screen, font, text, cx, y, color):
        surf = font.render(str(text), True, color)
        screen.blit(surf, (cx - surf.get_width() // 2, y))