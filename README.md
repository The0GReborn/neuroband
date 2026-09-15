# NeuroBand BCI — Architecture & Operations Guide

---

> **Note on this document (added while wiring up the setup wizard / dashboard
> CLI):** most of the content below describes the **original V2 pipeline**
> (LDA classifier, `EEGLogger`, `VoteBuffer`, `SimulatedADC`) which predates
> the current V3 system (k-NN + `SessionManager` + `WordManager` +
> `EMAVoter`). It's kept here for historical/V2 reference, but if you're
> working with the current codebase, read the **V3 Operations (current)**
> section below first — the rest of this file may describe classes and
> files that either no longer exist or have been superseded.

---

## V3 Operations (current)

### Getting started

**Automated (recommended):**
```bash
bash install.sh
```
Installs system packages (`espeak-ng`, `python3-venv`, `python3-rpi.gpio`),
creates `.venv`, installs all Python dependencies, sets up the `neuroband`
command, and offers to launch the setup wizard — all idempotent, safe to
re-run. Flags: `--yes` (don't prompt), `--no-wizard` (skip the wizard-launch
offer), `--simulate` (if launching the wizard, launch it with `--simulate`).

**Manual, if you'd rather do it step by step:**
```bash
sudo apt update && sudo apt install -y espeak-ng python3-venv python3-rpi.gpio
python3 -m venv --system-site-packages .venv   # --system-site-packages is
source .venv/bin/activate                       # required for RPi.GPIO to
pip install flask flask-socketio numpy scipy scikit-learn joblib pygame rich
bash docs/install.sh    # sets up the `neuroband` command
python main.py --setup  # or: neuroband --setup
```

On a normal `python main.py` (or `neuroband`, no flags) after setup, the
wizard launches automatically the first time (when
`config/setup_state.json` isn't `READY` yet) — you don't have to remember
`--setup` for a genuinely first-ever run.

### The `neuroband` command (optional convenience)
`bin/neuroband` is a repo-tracked wrapper so you can type `neuroband --setup`
instead of `python main.py --setup` from any directory. `install.sh` sets
this up automatically; to do it on its own, without the rest of `install.sh`:
```bash
bash docs/install.sh
source ~/.bashrc   # or open a new shell
neuroband --help
```
It just `cd`s into the repo, activates `.venv` if present, and execs
`python3 main.py "$@"` — every flag documented for `main.py` works
identically through `neuroband`.

### New modules (not covered in the "Module Explanations" section below)
| Module | Role |
|---|---|
| `config_manager.py` | Owns `config/user_config.json` (device/audio/EEG) and `config/setup_state.json` (`NOT_CONFIGURED → CONFIGURED → TRAINED → READY`). `config.py` applies these overrides on import. |
| `setup_wizard.py` | Interactive first-run flow. Reuses `WordManager`, `SessionManager`, `Calibrator`, `QualityChecker` — doesn't reimplement them. |
| `v3_engine.py` | Shared V3 object-graph builder (`build_v3_engine`) and quality-check/calibration helpers, used by `main.py`, `setup_wizard.py`, and `dashboard_cli.py` so all three stay in sync. Also has `get_local_ip()`/`port_in_use()` so no dashboard URL is ever hard-coded and no command starts a duplicate Flask/SocketIO server. |
| `dashboard_cli.py` | Wires up the **existing** `app_server.py` (web) and `dashboard.py` (terminal, `rich`-based) for `-dashboard -server` / `-dashboard -terminal`. Neither dashboard implementation was rewritten — `dashboard.py` in particular existed in the repo already but was never actually called from anywhere until this. |

### Audio engine — espeak-ng, not pyttsx3
`audio_output.py` generates speech via a direct `espeak-ng` subprocess call
(voice/speed/pitch, dynamically detected via `espeak-ng --voices`), not
`pyttsx3` — pyttsx3 didn't expose real pitch control. Requires
`sudo apt install espeak-ng` on the Pi.

### Known issues (found while building the above, not yet all fixed)
- **Fixed:** `signal_processing.py`'s band-power helper used `np.trapz`,
  removed in NumPy 2.0+ (renamed `np.trapezoid`). Patched to support both.
- **Fixed:** a debug-log line in `SignalProcessor._normalise()` crashed
  unconditionally on NumPy 2.0+ (`float()` of a keepdims-shaped 1-element
  array), which was silently killing the web app's background thread —
  `main.py` printed "Web app → ..." even though the server had already
  crashed. Also needed `allow_unsafe_werkzeug=True` on `socketio.run()` for
  current `flask-socketio` versions to run outside Flask's debug/reloader
  context.
- **Not fixed / known limitation:** `SignalProcessor._apply_bandpass()` uses
  a **stateful streaming filter** (`sosfilt` with persistent `zi`), correct
  for a real continuous ADC stream but **not** for simulated training, which
  generates disconnected synthetic bursts per trial. Feeding those into the
  stateful filter destabilizes it (observed peak amplitude >1,000,000 µV
  after filtering), which trips the 200 µV artifact-rejection threshold and
  rejects every simulated training window. **This means `--setup --simulate`
  and `--retrain --simulate` will reliably show 0 samples collected per
  word during Training** — that's this known issue, not a new bug. Real
  hardware training should be unaffected since real EEG genuinely is
  continuous.

---

## Project File Structure

```
/home/pi/neuroband/
├── web_app
    └── index.html
├── config.py                ← shared constants (edit this first)
├── adc_reader.py            ← CS1237 hardware driver + simulator
├── signal_processing.py     ← filtering pipeline
├── feature_extraction.py    ← feature vector computation
├── classifier.py            ← LDA model + training CLI
├── audio_output.py          ← speaker playback
├── main.py                  ← system loop + training logger
├── models/
│   └── neuroband_model.pkl  ← trained classifier (auto-created)
├── audio/
│   ├── yes.wav
│   └── no.wav
└── logs/
    ├── neuroband.log
    ├── training_data.npz    ← accumulated labelled windows
    └── raw_YES_<ts>.npy     ← raw EEG recordings per session
```

---

## Module Explanations

### config.py
Single source of truth. All thresholds, pin numbers, sample rates, and paths
live here. **Change `NOTCH_FREQ` to 60.0 if you're in North America.**

### adc_reader.py
**Two-layer design:**
- `CS1237Driver` — bit-bang SPI driver that reads 24-bit signed samples
  from each chip. The CS1237 uses a non-standard single-wire protocol:
  DRDY signals data-ready, then you clock out 24 bits, then 3 extra clocks.
- `SimulatedADC` — generates synthetic alpha + beta sine waves + Gaussian
  noise. Lets you test the full pipeline on any laptop.
- `ADCReader` — spawns one thread per channel, each writing to a
  `collections.deque(maxlen=BUFFER_SAMPLES)` ring buffer.

**Why a deque (ring buffer)?**
A `deque` with `maxlen` automatically discards the oldest sample when full.
This gives you a rolling window with zero memory allocation overhead,
which is critical on the Pi Zero 2 W's 512 MB RAM.

### signal_processing.py
Five-step pipeline applied to every 2-second window:

| Step | Method | Purpose |
|------|--------|---------|
| 1 | Mean subtraction | Remove DC offset from electrode contact potential |
| 2 | IIR Notch 50 Hz | Eliminate mains power line interference |
| 3 | Butterworth bandpass 0.5–40 Hz | Keep only brain-relevant frequencies |
| 4 | Amplitude threshold ±200 µV | Reject blink/muscle artifacts |
| 5 | Z-score normalisation | Scale independent of electrode impedance |

**Key design decision:** Filter state (`zi`) is preserved between windows
using `sosfilt_zi`. This prevents ringing artifacts at window boundaries —
a common bug in EEG systems.

### feature_extraction.py
Extracts 25 features per window:
- 12 per channel × 2 channels = 24 features
- 1 cross-channel alpha asymmetry feature

**Alpha asymmetry** is the most important feature for YES/NO discrimination.
Research shows left-frontal alpha suppression (ERD) correlates with
approach motivation ("YES-like" states), while right-frontal suppression
correlates with withdrawal.

### classifier.py
**LDA (Linear Discriminant Analysis)** is chosen as the default because:
- Inference takes < 1 ms on Pi Zero 2 W (critical for < 500 ms latency)
- Works well with small datasets (100–500 samples per class)
- Robust to the high feature correlation typical of EEG data

**Random Forest** is available via `use_rf=True` and becomes preferable
once you have > 200 samples per class.

### audio_output.py
Uses pre-rendered WAV files via `pygame.mixer` for the lowest possible
latency (< 10 ms to first audio sample). Falls back to `pyttsx3` TTS
if WAV files are missing. A 2-second cooldown prevents rapid re-triggering.

### main.py
Orchestrates the full pipeline with two modes:

**Inference mode:**
```
ADCReader → WindowManager → SignalProcessor → FeatureExtractor
→ NeuroBandClassifier → VoteBuffer → AudioOutput
```

**Training mode:**
```
ADCReader → WindowManager → SignalProcessor → FeatureExtractor
→ EEGLogger → logs/training_data.npz
```

---

## How Buffering Works

```
ADC thread (250 Hz)     →  deque [maxlen=2500]  ←  main thread
                                ring buffer
```

The ADC thread writes samples continuously. The main thread reads the
**last N samples** every `STEP_SAMPLES` (125 samples = 0.5 seconds).
This 50% overlap gives you a new inference every 500 ms while each
window is still 2 seconds long — maximising context while hitting the
latency target.

**Timing budget for < 500 ms latency:**
| Stage | Time |
|-------|------|
| ADC read (hardware paced) | 0 ms (async) |
| Signal processing (scipy) | ~15 ms |
| Feature extraction (Welch) | ~30 ms |
| LDA inference | < 1 ms |
| Vote buffer (3 windows) | +500 ms max |
| Audio playback start | < 10 ms |
| **Total** | **~550 ms worst case** |

To cut latency below 500 ms: reduce `WINDOW_SECONDS` to 1.0 and
disable the vote buffer (set `window=1` in `VoteBuffer`). This trades
accuracy for speed.

---

## Training Workflow

### Step 1 — Install dependencies on Pi

```bash
pip install numpy scipy scikit-learn pygame pyttsx3 RPi.GPIO
```

### Step 2 — Generate audio files (on any machine first)

```bash
python audio_output.py --generate-audio
# Copy the audio/ folder to your Pi
```

### Step 3 — Collect labelled EEG data

Run each command in a separate session (60 seconds each):

```bash
# Collect REST baseline (eyes closed, relaxed, no mental effort)
python main.py --train REST --duration 60

# Collect YES data (focus on "yes", nod mentally, imagine agreement)
python main.py --train YES --duration 60

# Collect NO data (focus on "no", imagine disagreement or rejection)
python main.py --train NO --duration 60
```

Aim for **at least 3 sessions per label** (180 seconds total per class).
More = better. Re-run sessions on different days for robustness.

### Step 4 — Train the model

```bash
python classifier.py --train
```

You'll see cross-validation accuracy and a confusion matrix. A good
baseline is > 70% accuracy. If accuracy is low, collect more data.

### Step 5 — Run live inference

```bash
python main.py
# or with hardware:
python main.py
# simulation mode:
python main.py --simulate
```

---

## Testing Without Real Brain Data

### Option 1 — Full simulation mode
```bash
python main.py --simulate
```
The `SimulatedADC` class generates EEG-like waveforms. The system runs
end-to-end without any hardware connected.

### Option 2 — Inject recorded data
Save a numpy array of real or synthetic EEG data, then patch `adc_reader.py`
to replay it instead of reading from hardware. This is useful for
regression testing after changing the signal processing pipeline.

### Option 3 — Use public EEG datasets
Download a public Motor Imagery dataset (e.g. BCI Competition IV Dataset 1)
in EDF format, read it with `mne` (`pip install mne`), and replay it through
the processing pipeline to validate feature extraction.

```python
import mne
raw = mne.io.read_raw_edf("subject1.edf", preload=True)
data = raw.get_data()   # shape (n_channels, n_samples)
# Feed data[0] and data[1] as ch0 and ch1 to your pipeline
```

---

## Improving Accuracy

1. **More training data** — 300+ windows per class is the single biggest lever
2. **Better electrode contact** — clean skin with alcohol, use EEG gel
3. **Consistent mental strategy** — standardise your imagery task
4. **Subject-specific training** — always train on the user's own brain signals
5. **Increase window overlap** — try 75% overlap (STEP_SAMPLES = 62)
6. **Add Common Spatial Pattern (CSP)** — a powerful spatial filter for motor
   imagery, available in `mne` and `pyriemann`
7. **Switch to Random Forest** — set `use_rf=True` in `classifier.py` once
   you have > 200 samples/class
8. **Feature selection** — use `sklearn.feature_selection.SelectKBest` to
   drop noisy features before training

---

## Wiring Summary (Pi Zero 2 W GPIO)

```
CS1237 #1 (F3-Cz)      CS1237 #2 (F4-Cz)
  DOUT → GPIO 9           DOUT → GPIO 10
  SCLK → GPIO 11          SCLK → GPIO 11  (shared clock)
  VCC  → 3.3V             VCC  → 3.3V
  GND  → GND              GND  → GND
```

Both CS1237 chips share the clock line (GPIO 11) but have separate
data lines. Since the chips are read sequentially in separate threads,
consider adding a hardware mutex (or read them in the same thread) if
you see data corruption.
