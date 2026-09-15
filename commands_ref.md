# NeuroBand — CLI Command Reference

> All commands run from `/home/pi/neuroband/` (or your project root on Windows).
>
> **First time on this machine?** Run `bash install.sh` once — it installs
> everything (system packages, `.venv`, Python deps, the `neuroband`
> command) and offers to launch the setup wizard. See `README.md` for
> details.
>
> Every command below is shown as `python main.py ...`. Once `install.sh`
> (or just `bash docs/install.sh`) has run, you can use `neuroband ...`
> instead, from any directory — same flags, same behavior.

---

## `install.sh` — Full Environment Setup

```bash
bash install.sh                    # interactive — prompts before each step
bash install.sh --yes              # don't prompt — assume yes to everything
bash install.sh --no-wizard        # set up the environment, skip the
                                    # "launch wizard now?" offer at the end
bash install.sh --yes --no-wizard --simulate
                                    # fully non-interactive, environment only
```

### All `install.sh` Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--yes` | bool | off | Don't prompt before installing system packages or launching the wizard |
| `--no-wizard` | bool | off | Skip the "launch wizard now?" step entirely |
| `--simulate` | bool | off | If the wizard is launched, launch it with `--simulate` |

Idempotent — safe to re-run any time (e.g. after a `git pull` that adds a
new dependency).

---

## `main.py` — Main Entry Point

### First-time setup / configuration (v3)

```bash
python main.py --setup                  # Force the full config/train/calibrate wizard
python main.py --setup --simulate       # Same, without real hardware
python main.py --voice                  # Reopen just voice/speed/pitch config
python main.py --retrain                # Vocabulary + training + calibration only
python main.py --retrain --simulate     # Same, without real hardware
```

On a normal `python main.py` with no flags, the wizard launches automatically
the first time (when `config/setup_state.json` isn't `READY` yet) — you don't
have to remember `--setup` for a genuinely first-ever run.

### Dashboard modes (v3)

Separate from scanning/inference, training, configuration, and calibration.
Starting a dashboard never starts training on its own.

```bash
python main.py -dashboard               # Web dashboard (default)
python main.py -dashboard -server       # Web dashboard, explicit
python main.py -dashboard -terminal     # Terminal (rich) dashboard — works over
                                         # headless SSH / Raspberry Pi Connect
```

`-dashboard -server` detects an already-running server (from a separate
`python main.py` scanning session, or another `-dashboard -server` call) and
just prints its URL instead of starting a duplicate. The printed URL uses the
device's real detected LAN IP, never a hard-coded address.

### V3 Mode (default, once configured)

```bash
python main.py                          # V3 inference + web app (requires hardware)
python main.py --simulate               # V3 simulation, no hardware
python main.py --no-calib               # Skip 30s auto-calibration
python main.py --no-quality             # Skip 5s signal quality check
python main.py --no-server              # No Flask web app, terminal output only
python main.py --simulate --no-calib --no-quality --no-server   # Fastest dev start
```

### V2 Mode (legacy LDA pipeline)

```bash
python main.py --v2                     # V2 inference with LDA classifier
python main.py --v2 --simulate         # V2 simulation, no hardware
python main.py --v2 --no-dash          # Disable pygame GUI dashboard
python main.py --v2 --no-calib         # Skip calibration
python main.py --v2 --no-quality       # Skip signal quality check
```

### V2 Training Data Collection

```bash
python main.py --v2 --train YES        # Collect YES training data (60s default)
python main.py --v2 --train NO         # Collect NO training data
python main.py --v2 --train REST       # Collect REST training data
python main.py --v2 --train YES --duration 90   # Custom duration (seconds)
```

### All `main.py` Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--setup` | bool | off | *(v3)* Force the full config/train/calibrate wizard |
| `--voice` | bool | off | *(v3)* Reopen just voice/speed/pitch config |
| `--retrain` | bool | off | *(v3)* Vocabulary + training + calibration only |
| `-dashboard` | bool | off | *(v3)* Show the dashboard (default: web) |
| `-server` | bool | off | *(v3)* With `-dashboard`: use the web dashboard |
| `-terminal` | bool | off | *(v3)* With `-dashboard`: use the terminal dashboard |
| `--v2` | bool | off | Run legacy V2 LDA pipeline |
| `--simulate` | bool | off | Simulate ADC — no hardware needed |
| `--no-calib` | bool | off | Skip auto-calibration |
| `--no-quality` | bool | off | Skip signal quality check |
| `--no-server` | bool | off | *(V3)* Disable Flask web app |
| `--train` | choice | — | *(V2)* Collect training data: `YES`, `NO`, or `REST` |
| `--duration` | int | `60` | *(V2 train)* Recording duration in seconds |
| `--no-dash` | bool | off | *(V2)* Disable pygame GUI dashboard |

---

## `classifier.py` — Train / Inspect LDA Model

```bash
python classifier.py --train            # Train V2 LDA from logs/training_data.npz
python classifier.py --train --rf       # Train using Random Forest instead of LDA
python classifier.py --train --debug    # Train with DEBUG-level logging
```

> Training data must exist at `logs/training_data.npz`.  
> Collect it first with `python main.py --v2 --train <LABEL>`.

### All `classifier.py` Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--train` | bool | off | Run training from saved npz data |
| `--rf` | bool | off | Use Random Forest instead of LDA |
| `--debug` | bool | off | Enable DEBUG logging |

---

## `simulator.py` — EEG Simulation & Pipeline Testing

```bash
# Quick window generation (no pipeline, just prints channel stats)
python simulator.py
python simulator.py --words YES NO HELLO HELP
python simulator.py --mode simple
python simulator.py --mode realistic --noise 0.2 --drift 0.5

# Full pipeline test (requires trained k-NN model)
python simulator.py --pipeline
python simulator.py --pipeline --words YES NO HELLO HELP
python simulator.py --pipeline --words YES NO --duration 3.0
python simulator.py --pipeline --words YES NO HELLO HELP --duration 3.0 --overlap 0.2
python simulator.py --pipeline --debug

# Pipeline test with simulated retraining (recommended for testing)
python simulator.py --pipeline --retrain --words YES NO HELLO HELP
python simulator.py --pipeline --retrain --words YES NO HELLO HELP --duration 3.0
python simulator.py --pipeline --retrain --words YES NO HELLO HELP --duration 3.0 --overlap 0.2
```

### All `simulator.py` Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mode` | choice | `realistic` | `simple` or `realistic` generation mode |
| `--words` | list | config words | Word sequence to simulate |
| `--duration` | float | `2.0` | Seconds of EEG per word |
| `--noise` | float | `0.15` | Gaussian noise level (fraction of amplitude) |
| `--drift` | float | `0.3` | Slow DC drift strength multiplier |
| `--spikes` | float | `0.05` | Per-window artifact spike probability |
| `--overlap` | float | `0.2` | Word overlap strength (`0`=separated, `1`=identical) |
| `--retrain` | bool | off | Retrain k-NN on simulated data before pipeline test |
| `--pipeline` | bool | off | Run through full SignalProcessor→k-NN→EMAVoter pipeline |
| `--seed` | int | `42` | RNG seed for reproducibility |
| `--debug` | bool | off | Enable DEBUG logging |

---

## `adc_reader.py` — ADC Simulation Testing

```bash
python adc_reader.py --simulate                         # V3 sim (default), word=HELLO
python adc_reader.py --simulate --word HELLO            # Explicit word
python adc_reader.py --simulate --v3 --word YES         # V3 sim, specific word
python adc_reader.py --simulate --v2                    # V2 sim, label=REST
python adc_reader.py --simulate --v2 --label YES        # V2 sim, label=YES
python adc_reader.py --simulate --v2 --label NO
python adc_reader.py --simulate --v2 --label REST
python adc_reader.py --simulate --duration 10           # Run for 10 seconds
```

### All `adc_reader.py` Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--simulate` | bool | off | Required — run without hardware |
| `--v2` | bool | off | Use V2 simulator (alpha/beta/noise) |
| `--v3` | bool | off | Use V3 simulator (word-specific frequencies) |
| `--label` | str | `REST` | *(V2)* Sim label: `YES`, `NO`, or `REST` |
| `--word` | str | `HELLO` | *(V3)* Active word for frequency signature |
| `--duration` | int | `5` | Run duration in seconds |

> **Note:** `--v3` is the default if neither `--v2` nor `--v3` is specified.  
> `--v2` takes priority only when explicitly set without `--v3`.

---

## Startup Sequences

### Quickest simulation test (V3)
```bash
python main.py --simulate --no-calib --no-quality
```

### Full hardware startup (V3)
```bash
python main.py
# 1. Signal quality check (5s)
# 2. Auto-calibration (30s, eyes closed)
# 3. Web app → http://192.168.4.1:5000
# 4. Inference loop starts
```

### First-time V2 model training
```bash
python main.py --v2 --train YES
python main.py --v2 --train NO
python main.py --v2 --train REST
python classifier.py --train
python main.py --v2
```

### Pipeline regression test after code changes
```bash
python simulator.py --pipeline --retrain --words YES NO HELLO HELP --duration 3.0 --seed 42
```

> **Tip:** Use `--retrain` to train a fresh k-NN on simulated data before testing.
> Without it, the pipeline uses the saved k-NN model which may not match simulated signals.

---

## File & Path Reference

| Path | Description |
|------|-------------|
| `install.sh` | Full automated environment setup — system packages, `.venv`, Python deps, CLI, offers to launch the wizard. Run once: `bash install.sh` |
| `bin/neuroband` | Repo-tracked CLI wrapper — resolves its own location, works from any pwd |
| `docs/install.sh` | Sets up just the `neuroband` command on `PATH` (also called by `install.sh`) |
| `config_manager.py` | Owns `config/user_config.json` + `config/setup_state.json` |
| `setup_wizard.py` | Interactive first-run wizard (device/audio/EEG/vocab/train/calibrate) |
| `v3_engine.py` | Shared V3 object-graph builder, reused by `main.py`, the wizard, and the dashboard CLI |
| `dashboard_cli.py` | `-dashboard -server` / `-dashboard -terminal` entry points |
| `config/user_config.json` | Device name, audio voice/speed/pitch, EEG hardware snapshot |
| `config/setup_state.json` | `NOT_CONFIGURED` → `CONFIGURED` → `TRAINED` → `READY` |
| `models/neuroband_model.pkl` | V2 LDA / Random Forest model |
| `models/v3_knn.pkl` | V3 k-NN model (session-trained) |
| `models/baseline.pkl` | Calibration baseline (mean/std per feature) |
| `models/v3_words.json` | Active word list |
| `logs/training_data.npz` | V2 training dataset |
| `logs/v3/<WORD>.npz` | V3 per-word EEG feature datasets |
| `logs/neuroband.log` | Runtime log file |
| `audio/yes.wav` / `audio/no.wav` | V2 audio output files |
| `audio/v3/<WORD>.wav` | V3 per-word audio files (generated via espeak-ng, not pyttsx3) |
