"""
dashboard.py -- Live Terminal Dashboard for NeuroBand
Place in: /home/pi/neuroband/dashboard.py

Displays in real time:
  - EEG waveform sparklines for both channels
  - Band power bars (delta, theta, alpha, beta, gamma)
  - Confidence meter for current prediction
  - Last detected command with timestamp
  - Session statistics (windows, commands, uptime)
  - Signal quality indicator

Requires:
  pip install rich

Usage (called from main.py):
    dash = Dashboard()
    dash.update(clean_window, features, label, confidence, quality_rating)
    # call in the inference loop after every window
"""

import time
import collections
import logging
import numpy as np

log = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.table   import Table
    from rich.panel   import Panel
    from rich.layout  import Layout
    from rich.live    import Live
    from rich.text    import Text
    from rich.columns import Columns
    from rich         import box
    _RICH_OK = True
except ImportError:
    _RICH_OK = False
    log.warning("rich not installed. Run: pip install rich")

import config


# -- Sparkline helper -----------------------------------------------------------
def _sparkline(values: list[float], width: int = 20) -> str:
    """Convert a list of floats into a unicode sparkline string."""
    bars = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    if not values:
        return " " * width
    mn, mx = min(values), max(values)
    span = mx - mn or 1e-9
    result = []
    for v in values[-width:]:
        idx = int((v - mn) / span * (len(bars) - 1))
        result.append(bars[idx])
    # Pad left if shorter than width
    return " " * (width - len(result)) + "".join(result)


def _bar(value: float, max_val: float, width: int = 20, color: str = "green") -> str:
    """Render a simple ASCII power bar."""
    filled = int(min(value / (max_val + 1e-30), 1.0) * width)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return bar


# ══════════════════════════════════════════════════════════════════════════════
#  Dashboard
# ══════════════════════════════════════════════════════════════════════════════
class Dashboard:
    """
    Live terminal dashboard using rich.Live.

    Call dash.update(...) every inference window.
    Call dash.start() before the inference loop.
    Call dash.stop() on shutdown.
    """

    BAND_NAMES   = ["Delta", "Theta", "Alpha", "Beta ", "Gamma"]
    BAND_INDICES = [5, 6, 7, 8, 9]     # indices in per-channel feature vector (ch0)
    LABEL_COLORS = {"YES": "green", "NO": "red", "REST": "yellow"}
    QUALITY_COLORS = {"GOOD": "green", "WARN": "yellow", "FAIL": "red"}

    def __init__(self):
        self._console   = Console() if _RICH_OK else None
        self._live      = None
        self._running   = False

        # Rolling history
        self._ch0_hist  = collections.deque(maxlen=config.DASHBOARD_HISTORY)
        self._ch1_hist  = collections.deque(maxlen=config.DASHBOARD_HISTORY)
        self._conf_hist = collections.deque(maxlen=config.DASHBOARD_HISTORY)

        # Current state
        self._label     = "REST"
        self._conf      = 0.0
        self._features  = None
        self._quality   = "GOOD"
        self._n_windows = 0
        self._n_commands= 0
        self._last_cmd  = "—"
        self._last_cmd_time = "—"
        self._start_time = time.time()

    # -- Lifecycle --------------------------------------------------------------
    def start(self):
        if not _RICH_OK:
            log.info("Dashboard disabled (rich not installed).")
            return
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=config.DASHBOARD_REFRESH_HZ,
            screen=True,
        )
        self._live.start()
        self._running = True

    def stop(self):
        if self._live:
            self._live.stop()
        self._running = False

    # -- Update -----------------------------------------------------------------
    def update(self,
               clean_window: np.ndarray,
               features:     np.ndarray,
               label:        str,
               confidence:   float,
               quality:      str = "GOOD",
               n_windows:    int = 0,
               n_commands:   int = 0):
        """
        Call this every inference window to refresh the dashboard.

        Parameters
        ----------
        clean_window : np.ndarray (2, WINDOW_SAMPLES) -- filtered EEG
        features     : np.ndarray (N_FEATURES,)
        label        : str -- "YES", "NO", or "REST"
        confidence   : float -- 0.0 to 1.0
        quality      : str -- "GOOD", "WARN", or "FAIL"
        n_windows    : int -- total windows processed
        n_commands   : int -- total commands detected
        """
        # Update waveform history (use RMS of each half-window as proxy)
        if clean_window is not None:
            mid = clean_window.shape[1] // 2
            self._ch0_hist.append(float(np.std(clean_window[0, mid:]) * 1e6))
            self._ch1_hist.append(float(np.std(clean_window[1, mid:]) * 1e6))

        self._conf_hist.append(confidence)
        self._label     = label
        self._conf      = confidence
        self._features  = features
        self._quality   = quality
        self._n_windows = n_windows
        self._n_commands= n_commands

        if label != "REST":
            self._last_cmd      = label
            self._last_cmd_time = time.strftime("%H:%M:%S")

        if self._live and self._running:
            self._live.update(self._render())

    # -- Rendering --------------------------------------------------------------
    def _render(self):
        if not _RICH_OK:
            return ""

        uptime = int(time.time() - self._start_time)
        h, m, s = uptime // 3600, (uptime % 3600) // 60, uptime % 60

        # ── Header ──────────────────────────────────────────────────────────
        header = Text()
        header.append("  NeuroBand BCI  ", style="bold white on blue")
        header.append(f"  Uptime: {h:02d}:{m:02d}:{s:02d}  ", style="dim")
        header.append(f"  Windows: {self._n_windows}  ", style="dim")
        header.append(f"  Commands: {self._n_commands}  ", style="dim")

        quality_color = self.QUALITY_COLORS.get(self._quality, "white")
        header.append(f"  Signal: ", style="dim")
        header.append(f"{self._quality}  ", style=f"bold {quality_color}")

        # ── EEG Waveforms ───────────────────────────────────────────────────
        wave_table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        wave_table.add_column("Channel",  width=12)
        wave_table.add_column("Waveform (activity)", width=42)
        wave_table.add_column("Level", width=10)

        ch0_spark = _sparkline(list(self._ch0_hist), width=40)
        ch1_spark = _sparkline(list(self._ch1_hist), width=40)
        ch0_rms   = list(self._ch0_hist)[-1] if self._ch0_hist else 0
        ch1_rms   = list(self._ch1_hist)[-1] if self._ch1_hist else 0

        wave_table.add_row(
            "CH0  F3-Cz",
            Text(ch0_spark, style="cyan"),
            f"{ch0_rms:.2f} uV"
        )
        wave_table.add_row(
            "CH1  F4-Cz",
            Text(ch1_spark, style="magenta"),
            f"{ch1_rms:.2f} uV"
        )

        # ── Band Powers ─────────────────────────────────────────────────────
        band_table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        band_table.add_column("Band",   width=8)
        band_table.add_column("CH0 (F3)", width=24)
        band_table.add_column("CH1 (F4)", width=24)

        band_colors = ["blue", "cyan", "green", "yellow", "red"]

        if self._features is not None:
            for i, (name, feat_idx) in enumerate(
                zip(self.BAND_NAMES, self.BAND_INDICES)
            ):
                ch0_pow = float(self._features[feat_idx])
                ch1_pow = float(self._features[feat_idx + 12])
                max_pow = max(ch0_pow, ch1_pow, 1e-30)
                color   = band_colors[i]
                bar0    = _bar(ch0_pow, max_pow * 1.2, width=20)
                bar1    = _bar(ch1_pow, max_pow * 1.2, width=20)
                band_table.add_row(
                    name,
                    Text(bar0, style=color),
                    Text(bar1, style=color),
                )
        else:
            band_table.add_row("—", "No data yet", "No data yet")

        # ── Command & Confidence ────────────────────────────────────────────
        label_color = self.LABEL_COLORS.get(self._label, "white")
        conf_pct    = int(self._conf * 100)
        conf_bar    = _bar(self._conf, 1.0, width=30)

        cmd_text = Text()
        cmd_text.append("\n  Current:  ", style="dim")
        cmd_text.append(f" {self._label} ", style=f"bold white on {label_color}")
        cmd_text.append(f"  {conf_pct}%\n", style="bold")
        cmd_text.append(f"  {conf_bar}\n", style=label_color)
        cmd_text.append(f"\n  Last CMD: ", style="dim")
        cmd_text.append(f"{self._last_cmd}", style="bold green")
        cmd_text.append(f"  at {self._last_cmd_time}\n", style="dim")

        # ── Confidence sparkline ────────────────────────────────────────────
        conf_spark = _sparkline(list(self._conf_hist), width=40)
        conf_text  = Text()
        conf_text.append("  Confidence history:  ", style="dim")
        conf_text.append(conf_spark, style="green")

        # ── Assemble layout ─────────────────────────────────────────────────
        from rich.console import Group
        return Panel(
            Group(
                header,
                Text(""),
                Panel(wave_table,  title="[bold]EEG Signals",    border_style="cyan"),
                Panel(band_table,  title="[bold]Band Powers",     border_style="blue"),
                Panel(cmd_text,    title="[bold]Command Output",  border_style="green"),
                conf_text,
                Text(""),
                Text("  Press Ctrl+C to stop.", style="dim"),
            ),
            title="[bold blue]NeuroBand Live Monitor",
            border_style="blue",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Fallback plain-text dashboard (when rich is not installed)
# ══════════════════════════════════════════════════════════════════════════════
class PlainDashboard:
    """Simple print-based fallback when rich is unavailable."""

    def __init__(self):
        self._n = 0

    def start(self): pass
    def stop(self):  pass

    def update(self, clean_window, features, label, confidence,
               quality="GOOD", n_windows=0, n_commands=0):
        self._n += 1
        if self._n % 5 == 0:   # print every 5th window to avoid spam
            print(f"  [w={n_windows:>4}] {label:<5} conf={confidence:.2f} "
                  f"quality={quality}", flush=True)


def make_dashboard() -> "Dashboard | PlainDashboard":
    """Factory — returns rich Dashboard if available, else PlainDashboard."""
    if _RICH_OK:
        return Dashboard()
    log.info("rich not available, using plain dashboard. pip install rich")
    return PlainDashboard()