# NeuroBand — CLI Command Reference

> All commands run from `/home/pi/neuroband/` (or your project root on Windows).

---

## `main.py` — Main Entry Point

### V3 Mode (default)

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
| `models/neuroband_model.pkl` | V2 LDA / Random Forest model |
| `models/v3_knn.pkl` | V3 k-NN model (session-trained) |
| `models/baseline.pkl` | Calibration baseline (mean/std per feature) |
| `models/v3_words.json` | Active word list |
| `logs/training_data.npz` | V2 training dataset |
| `logs/v3/<WORD>.npz` | V3 per-word EEG feature datasets |
| `logs/neuroband.log` | Runtime log file |
| `audio/yes.wav` / `audio/no.wav` | V2 audio output files |
| `audio/v3/<WORD>.wav` | V3 per-word audio files |
