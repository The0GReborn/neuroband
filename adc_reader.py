"""
adc_reader.py — Unified CS1237 ADC Reader for NeuroBand (v2 + v3)
Place in: /home/pi/neuroband/adc_reader.py

Hardware wiring (BCM GPIO, Pi Zero 2 W):
  CS1237 SCLK  → GPIO 11
  CS1237 DOUT1 → GPIO 9   (BioAmp 1: F3-Cz)
  CS1237 DOUT2 → GPIO 10  (BioAmp 2: F4-Cz)
  CS1237 VCC   → 3.3V
  CS1237 GND   → GND

Simulation modes:
  v2 sim — alpha + beta sine waves + pink noise (original behaviour)
  v3 sim — word-specific frequency signatures so k-NN can learn distinct patterns

CLI usage (standalone testing):
  python adc_reader.py --simulate              # v3 sim (default)
  python adc_reader.py --simulate --v2         # v2 sim, REST label
  python adc_reader.py --simulate --v2 --label YES
  python adc_reader.py --simulate --v3 --word HELLO
"""

import time
import math
import random
import threading
import collections
import logging
import argparse
import numpy as np

import config

log = logging.getLogger(__name__)

try:
    import RPi.GPIO as GPIO
    _GPIO_AVAILABLE = True
except ImportError:
    _GPIO_AVAILABLE = False
    log.warning("RPi.GPIO not found — ADC will run in simulation mode.")


# ══════════════════════════════════════════════════════════════════════════════
#  CS1237 Hardware Driver  (unchanged from v2)
# ══════════════════════════════════════════════════════════════════════════════
class CS1237Driver:
    SPEED_40SPS  = 0b00
    SPEED_250SPS = 0b10
    PGA_GAIN_1   = 0b00
    PGA_GAIN_128 = 0b11
    CH_A         = 0b00
    _T           = 0.000001

    def __init__(self, clk_pin, data_pin, channel_id=0):
        self.clk  = clk_pin
        self.data = data_pin
        self.id   = channel_id
        self._lock = threading.Lock()

    def setup(self):
        log.debug("CS1237 ch%d: setup() started — CLK=%d DATA=%d", self.id, self.clk, self.data)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.clk,  GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(self.data, GPIO.IN)
        log.debug("CS1237 ch%d: GPIO pins configured, calling _configure()", self.id)
        self._configure()
        log.info("CS1237 ch%d init DATA=%d CLK=%d", self.id, self.data, self.clk)
        log.debug("CS1237 ch%d: setup() complete", self.id)

    def _pulse_clk(self):
        GPIO.output(self.clk, GPIO.HIGH); time.sleep(self._T)
        GPIO.output(self.clk, GPIO.LOW);  time.sleep(self._T)

    def _wait_drdy(self, timeout=1.0):
        log.debug("CS1237 ch%d: waiting for DRDY (timeout=%.1fs)", self.id, timeout)
        deadline = time.time() + timeout
        while GPIO.input(self.data) == GPIO.HIGH:
            if time.time() > deadline:
                log.error("CS1237 ch%d: DRDY timeout after %.1fs", self.id, timeout)
                raise TimeoutError(f"CS1237 ch{self.id}: DRDY timeout")
            time.sleep(0.0001)
        log.debug("CS1237 ch%d: DRDY ready", self.id)

    def _read_24bits(self):
        value = 0
        for _ in range(24):
            GPIO.output(self.clk, GPIO.HIGH); time.sleep(self._T)
            bit = GPIO.input(self.data)
            GPIO.output(self.clk, GPIO.LOW);  time.sleep(self._T)
            value = (value << 1) | bit
        return value

    def _configure(self):
        log.debug("CS1237 ch%d: _configure() — writing cfg_byte (SPEED=250SPS, PGA=128, CH=A)", self.id)
        cfg_byte = (self.SPEED_250SPS << 4) | (self.PGA_GAIN_128 << 2) | self.CH_A
        log.debug("CS1237 ch%d: cfg_byte=0x%02X", self.id, cfg_byte)
        self._wait_drdy()
        self._read_24bits()
        for _ in range(3):
            self._pulse_clk()
        GPIO.setup(self.data, GPIO.OUT)
        for bit_pos in range(6, -1, -1):
            GPIO.output(self.data, (cfg_byte >> bit_pos) & 1)
            self._pulse_clk()
        GPIO.setup(self.data, GPIO.IN)
        GPIO.output(self.clk, GPIO.HIGH)
        log.debug("CS1237 ch%d: _configure() done", self.id)

    def read_raw(self):
        with self._lock:
            self._wait_drdy()
            raw = self._read_24bits()
            for _ in range(3):
                self._pulse_clk()
        if raw >= (1 << 23):
            raw -= (1 << 24)
        return raw

    def read_voltage(self):
        raw = self.read_raw()
        return (raw / (1 << 23)) * config.ADC_VREF / config.ADC_GAIN

    def cleanup(self):
        GPIO.cleanup()


# ══════════════════════════════════════════════════════════════════════════════
#  V2 Simulated ADC  (original behaviour — alpha/beta + pink noise)
# ══════════════════════════════════════════════════════════════════════════════
class SimulatedADC_V2:
    """
    Original v2 simulator.
    Generates alpha (10 Hz) + beta (20 Hz) + pink noise.
    Label drives asymmetry between channels to mimic YES/NO states.
    """

    def __init__(self, channel_id=0, label="REST"):
        self.id    = channel_id
        self.label = label
        self._t    = 0.0
        self._dt   = 1.0 / config.SAMPLE_RATE

    def _pink_noise(self):
        return random.gauss(0, config.SIM_NOISE_LEVEL * 1e-6)

    def read_voltage(self):
        t = self._t
        self._t += self._dt

        alpha_amp = 15e-6
        beta_amp  = 5e-6

        if self.label == "YES":
            alpha_amp = 5e-6  if self.id == 0 else 15e-6
            beta_amp  = 20e-6
        elif self.label == "NO":
            alpha_amp = 15e-6 if self.id == 0 else 5e-6
            beta_amp  = 20e-6

        v  = alpha_amp * math.sin(2 * math.pi * 10 * t)
        v += beta_amp  * math.sin(2 * math.pi * 20 * t)
        v += self._pink_noise()
        return v

    def read_raw(self):
        v = self.read_voltage()
        return int(v / config.ADC_VREF * (1 << 23) * config.ADC_GAIN)

    def cleanup(self):
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  V3 Simulated ADC  (word-specific frequency signatures)
# ══════════════════════════════════════════════════════════════════════════════
class SimulatedADC_V3:
    """
    V3 simulator.
    Each word gets a unique dominant frequency band so k-NN can learn
    distinct EEG signatures during simulated training sessions.

    Word index → base frequency:
      word[0] → 8.0 Hz   (alpha band)
      word[1] → 10.5 Hz
      word[2] → 13.0 Hz  (low beta)
      word[3] → 15.5 Hz
      ...spaced 2.5 Hz apart up to 10 words

    Phase offset between ch0/ch1 is also word-specific (asymmetry feature).
    """

    def __init__(self, channel_id=0, word="HELLO", word_list=None):
        self.id        = channel_id
        self.word      = word.upper()
        self.word_list = [w.upper() for w in (word_list or config.load_words())]
        self._t        = 0.0
        self._dt       = 1.0 / config.SAMPLE_RATE

    def _params(self):
        """Return (base_freq, phase_offset, alpha_amp, beta_amp) for current word."""
        words = self.word_list or [self.word]
        idx   = words.index(self.word) if self.word in words else 0
        base  = 8.0 + idx * 2.5                    # unique freq per word
        phase = 0.15 * idx                          # unique inter-channel phase
        # Channel 0 leads, channel 1 slightly suppressed for odd words
        a_amp = 18e-6 if (idx % 2 == 0) else 10e-6
        b_amp = 8e-6  if (idx % 2 == 0) else 18e-6
        if self.id == 1:                            # channel asymmetry
            a_amp, b_amp = b_amp * 0.8, a_amp * 1.2
        return base, phase, a_amp, b_amp

    def read_voltage(self):
        t = self._t
        self._t += self._dt
        base, phase, a_amp, b_amp = self._params()
        ch_phase = phase if self.id == 0 else phase + 0.3

        v  = a_amp * math.sin(2 * math.pi * base       * t + ch_phase)
        v += b_amp * math.sin(2 * math.pi * (base * 2) * t + ch_phase * 0.5)
        v += random.gauss(0, 2e-6)    # low noise
        return v

    def read_raw(self):
        v = self.read_voltage()
        return int(v / config.ADC_VREF * (1 << 23) * config.ADC_GAIN)

    def set_word(self, word: str):
        self.word = word.upper()

    def cleanup(self):
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  Sim window helper  (used by pattern_trainer.py for instant feature capture)
# ══════════════════════════════════════════════════════════════════════════════
def sim_window_v3(word: str, word_list: list) -> list:
    """
    Generate one full WINDOW_SAMPLES window of v3 simulated EEG.
    Returns list of 2 lists (one per channel).
    Used by pattern_trainer._record() in --simulate mode.
    """
    drivers = [SimulatedADC_V3(ch, word, word_list) for ch in range(config.NUM_CHANNELS)]
    return [[d.read_voltage() for _ in range(config.WINDOW_SAMPLES)] for d in drivers]


# ══════════════════════════════════════════════════════════════════════════════
#  ADCReader  — unified threaded reader (v2 + v3)
# ══════════════════════════════════════════════════════════════════════════════
class ADCReader:
    """
    Spawns one thread per channel. Pushes samples into ring buffers.

    Parameters
    ----------
    simulate  : bool   — use simulated ADC instead of hardware
    mode      : str    — "v3" (default) or "v2"
    sim_label : str    — v2 sim label ("YES"/"NO"/"REST")
    word      : str    — v3 sim active word (can be changed via set_word())
    word_list : list   — full word list for v3 sim frequency mapping

    Usage:
        reader = ADCReader(simulate=True, mode="v3", word="HELLO")
        reader.start()
        samples = reader.get_latest(500)
        reader.stop()
    """

    def __init__(self, simulate=False, mode="v3",
                 sim_label="REST", word="HELLO", word_list=None):
        self._simulate  = simulate or not _GPIO_AVAILABLE
        self._mode      = mode.lower()
        self._sim_label = sim_label
        self._word      = word.upper()
        self._word_list = word_list or config.load_words()

        self._buffers = [
            collections.deque(maxlen=config.BUFFER_SAMPLES)
            for _ in range(config.NUM_CHANNELS)
        ]
        self._threads = []
        self._running = threading.Event()

        if not self._simulate:
            GPIO.setmode(GPIO.BCM)
            self._drivers = [
                CS1237Driver(config.ADC_CLK_PIN, config.ADC_DATA_PIN_CH0, 0),
                CS1237Driver(config.ADC_CLK_PIN, config.ADC_DATA_PIN_CH1, 1),
            ]
            for d in self._drivers:
                d.setup()
        else:
            self._drivers = self._make_sim_drivers()
            log.info("ADCReader: simulation mode=%s", self._mode)

    def _make_sim_drivers(self):
        if self._mode == "v2":
            return [SimulatedADC_V2(ch, self._sim_label)
                    for ch in range(config.NUM_CHANNELS)]
        else:
            return [SimulatedADC_V3(ch, self._word, self._word_list)
                    for ch in range(config.NUM_CHANNELS)]

    # ── Threading ──────────────────────────────────────────────────────────────
    def start(self):
        log.debug("ADCReader.start() — spawning %d channel threads", config.NUM_CHANNELS)
        self._running.set()
        for idx, driver in enumerate(self._drivers):
            t = threading.Thread(
                target=self._read_loop, args=(idx, driver),
                name=f"ADC-ch{idx}", daemon=True)
            t.start()
            self._threads.append(t)
            log.debug("ADCReader: ch%d thread started", idx)
        log.info("ADCReader started (%d channels, mode=%s)", config.NUM_CHANNELS, self._mode)

    def stop(self):
        log.debug("ADCReader.stop() called — clearing running flag")
        self._running.clear()
        for t in self._threads:
            t.join(timeout=2.0)
        for d in self._drivers:
            d.cleanup()
        log.info("ADCReader stopped.")

    def _read_loop(self, ch_idx, driver):
        log.debug("ADCReader ch%d: _read_loop started (interval=%.4fs)", ch_idx, 1.0 / config.SAMPLE_RATE)
        interval = 1.0 / config.SAMPLE_RATE
        sample_count = 0
        while self._running.is_set():
            t0 = time.perf_counter()
            try:
                self._buffers[ch_idx].append(driver.read_voltage())
                sample_count += 1
                if sample_count % 250 == 0:
                    log.debug("ADCReader ch%d: %d samples collected, buffer=%d",
                              ch_idx, sample_count, len(self._buffers[ch_idx]))
            except Exception as exc:
                log.error("ADC ch%d error: %s", ch_idx, exc)
            if self._simulate:
                elapsed = time.perf_counter() - t0
                wait    = interval - elapsed
                if wait > 0:
                    time.sleep(wait)
        log.debug("ADCReader ch%d: _read_loop exited after %d samples", ch_idx, sample_count)

    # ── Data access ────────────────────────────────────────────────────────────
    def get_latest(self, n: int):
        for buf in self._buffers:
            if len(buf) < n:
                log.debug("ADCReader.get_latest(%d): not enough samples (have %d)", n, len(buf))
                return None
        log.debug("ADCReader.get_latest(%d): returning window", n)
        return [list(buf)[-n:] for buf in self._buffers]

    def samples_available(self) -> int:
        return min(len(b) for b in self._buffers)

    # ── Word control (v3 sim) ──────────────────────────────────────────────────
    def set_word(self, word: str):
        """Change active word for v3 simulation (used during training sessions)."""
        log.debug("ADCReader.set_word('%s') — updating sim drivers", word)
        self._word = word.upper()
        if self._simulate and self._mode == "v3":
            for d in self._drivers:
                d.set_word(word)
            log.debug("ADCReader: all sim drivers updated to word='%s'", word)

    # ── V2 label control ───────────────────────────────────────────────────────
    def set_sim_label(self, label: str):
        """Change label for v2 simulation."""
        self._sim_label = label
        if self._simulate and self._mode == "v2":
            for d in self._drivers:
                d.label = label


# ══════════════════════════════════════════════════════════════════════════════
#  CLI  — standalone simulation testing
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)

    ap = argparse.ArgumentParser(description="NeuroBand ADC Reader — simulation test")
    ap.add_argument("--simulate", action="store_true", help="Run in simulation mode")
    ap.add_argument("--v2",       action="store_true", help="Use v2 simulation (YES/NO/REST)")
    ap.add_argument("--v3",       action="store_true", help="Use v3 simulation (word-specific)")
    ap.add_argument("--label",    default="REST",       help="v2 sim label (YES/NO/REST)")
    ap.add_argument("--word",     default="HELLO",      help="v3 sim active word")
    ap.add_argument("--duration", type=int, default=5,  help="Seconds to run")
    args = ap.parse_args()

    if not args.simulate:
        print("Add --simulate to run without hardware.")
        sys.exit(0)

    # Determine mode — v3 is default unless --v2 explicitly set
    mode = "v2" if args.v2 and not args.v3 else "v3"
    print(f"\nNeuroBand ADC Simulation")
    print(f"  Mode     : {mode.upper()}")
    if mode == "v2":
        print(f"  Label    : {args.label}")
    else:
        print(f"  Word     : {args.word}")
    print(f"  Duration : {args.duration}s\n")

    reader = ADCReader(
        simulate  = True,
        mode      = mode,
        sim_label = args.label,
        word      = args.word,
    )
    reader.start()

    deadline = time.time() + args.duration
    while time.time() < deadline:
        avail = reader.samples_available()
        latest = reader.get_latest(10)
        if latest:
            ch0_rms = (sum(v**2 for v in latest[0]) / 10) ** 0.5
            ch1_rms = (sum(v**2 for v in latest[1]) / 10) ** 0.5
            print(f"\r  samples={avail:>5}  "
                  f"ch0_rms={ch0_rms*1e6:6.2f}uV  "
                  f"ch1_rms={ch1_rms*1e6:6.2f}uV  ", end="", flush=True)
        time.sleep(0.1)

    print("\n\nDone.")
    reader.stop()