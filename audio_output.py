"""
audio_output.py — Unified Audio Output for NeuroBand (v2 + v3)
Place in: /home/pi/neuroband/audio_output.py

V2 mode: speaks YES / NO only. Uses pre-rendered WAVs if present.
V3 mode: speaks any user-defined word. Generates + caches WAV on first use.

Strategy (both modes):
  1. Pre-loaded pygame Sound object → play instantly (< 10 ms)
  2. Generate WAV via pyttsx3 → cache to disk → load → play
  3. Direct pyttsx3 TTS (no file)
  4. Print to terminal (last resort)

Cooldown:
  V2 → 2.0 seconds  (original behaviour)
  V3 → 3.0 seconds  (democracy voter already enforces this,
                      audio layer adds a safety net)

Generate v2 WAV files (one-time, run on any machine with speakers):
  python audio_output.py --generate-audio
"""

import os
import time
import threading
import logging
import argparse

import config

log = logging.getLogger(__name__)

try:
    import pygame
    if not pygame.mixer.get_init():
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    _PG = True
    log.info("pygame mixer ready.")
except Exception as e:
    _PG = False
    log.warning("pygame unavailable: %s", e)

try:
    import pyttsx3
    _tts = pyttsx3.init()
    _tts.setProperty("rate", 140)
    _tts.setProperty("volume", 1.0)
    _TTS = True
except Exception:
    _TTS = False
    log.warning("pyttsx3 unavailable — terminal fallback only.")


# ══════════════════════════════════════════════════════════════════════════════
#  AudioOutput  — unified for v2 and v3
# ══════════════════════════════════════════════════════════════════════════════
class AudioOutput:
    """
    Thread-safe audio output with WAV caching and cooldown.

    Usage (v3):
        ao = AudioOutput(mode="v3")
        ao.speak("HELLO")          # non-blocking, generates WAV if needed
        ao.pregenerate(["HELLO", "HELP", "WATER"])

    Usage (v2):
        ao = AudioOutput(mode="v2")
        ao.speak("YES")
        ao.speak("NO")

    Shutdown:
        ao.shutdown()
    """

    def __init__(self, mode="v3"):
        log.debug("AudioOutput.__init__(): mode=%s cooldown=%.1fs", mode,
                  config.V3_COOLDOWN_SECONDS if mode == "v3" else 2.0)
        self._mode      = mode.lower()
        self._cooldown  = config.V3_COOLDOWN_SECONDS if mode == "v3" else 2.0
        self._last      = 0.0
        self._lock      = threading.Lock()
        self._cache: dict = {}

        os.makedirs(config.AUDIO_DIR,    exist_ok=True)
        os.makedirs(config.V3_AUDIO_DIR, exist_ok=True)

        log.debug("AudioOutput.__init__(): preloading v2 audio files")
        self._preload_v2()
        if mode == "v3":
            log.debug("AudioOutput.__init__(): preloading v3 audio cache")
            self._preload_v3()
        log.debug("AudioOutput.__init__(): ready — %d words cached", len(self._cache))

    # ── Preload ────────────────────────────────────────────────────────────────
    def _preload_v2(self):
        """Load YES/NO WAVs from audio/ (v2 originals)."""
        if not _PG:
            return
        for label, path in config.AUDIO_FILES.items():
            if path and os.path.exists(path):
                try:
                    self._cache[label] = pygame.mixer.Sound(path)
                    log.info("V2 audio loaded: %s", label)
                except Exception as exc:
                    log.warning("Could not load %s: %s", path, exc)

    def _preload_v3(self):
        """Load any cached WAVs from audio/v3/."""
        if not _PG:
            return
        for fname in os.listdir(config.V3_AUDIO_DIR):
            if fname.endswith(".wav"):
                word = fname[:-4].upper()
                if word not in self._cache:   # don't overwrite v2 YES/NO
                    path = os.path.join(config.V3_AUDIO_DIR, fname)
                    try:
                        self._cache[word] = pygame.mixer.Sound(path)
                        log.debug("V3 audio loaded: %s", word)
                    except Exception as exc:
                        log.warning("Could not load %s: %s", path, exc)
        log.info("AudioOutput ready — %d words cached.", len(self._cache))

    # ── Public API ─────────────────────────────────────────────────────────────
    def speak(self, word: str):
        """
        Speak a word. Non-blocking (daemon thread).
        Cooldown is enforced here as a safety net.
        """
        word = word.upper().strip()
        log.debug("AudioOutput.speak('%s'): requested", word)
        if not word or word == "REST":
            log.debug("AudioOutput.speak('%s'): skipped (empty or REST)", word)
            return

        now = time.time()
        with self._lock:
            elapsed = now - self._last
            if elapsed < self._cooldown:
                log.debug("AudioOutput.speak('%s'): suppressed (cooldown %.1fs remaining)",
                          word, self._cooldown - elapsed)
                return
            self._last = now

        log.debug("AudioOutput.speak('%s'): spawning audio thread", word)
        threading.Thread(
            target=self._play,
            args=(word,),
            daemon=True,
            name=f"Audio-{word}",
        ).start()

    def pregenerate(self, words: list):
        """
        Pre-generate and cache WAV files for a list of words.
        Call after adding new words so first playback is instant.
        """
        for w in words:
            w = w.upper().strip()
            if w not in self._cache:
                path = self._wav_path(w)
                if not os.path.exists(path):
                    self._gen_wav(w, path)
                self._load_to_cache(w, path)

    def invalidate(self, word: str):
        """Remove a word from cache (e.g. after renaming)."""
        word = word.upper().strip()
        self._cache.pop(word, None)
        path = self._wav_path(word)
        if os.path.exists(path):
            os.remove(path)

    def shutdown(self):
        if _PG:
            try:
                pygame.mixer.quit()
            except Exception:
                pass

    # ── Playback chain ─────────────────────────────────────────────────────────
    def _play(self, word: str):
        log.info("Speaking: %s", word)

        # 1 — cached pygame Sound
        if _PG and word in self._cache:
            log.debug("AudioOutput._play('%s'): using cached pygame Sound", word)
            try:
                self._cache[word].play()
                while pygame.mixer.get_busy():
                    time.sleep(0.01)
                log.debug("AudioOutput._play('%s'): pygame playback complete", word)
                return
            except Exception as exc:
                log.error("pygame play error: %s", exc)

        # 2 — generate WAV → load → play
        log.debug("AudioOutput._play('%s'): no cache hit, checking for WAV file", word)
        path = self._wav_path(word)
        if not os.path.exists(path):
            log.debug("AudioOutput._play('%s'): generating WAV via TTS → %s", word, path)
            self._gen_wav(word, path)
        if _PG and os.path.exists(path):
            log.debug("AudioOutput._play('%s'): loading WAV and playing via pygame", word)
            try:
                snd = pygame.mixer.Sound(path)
                with self._lock:
                    self._cache[word] = snd
                snd.play()
                while pygame.mixer.get_busy():
                    time.sleep(0.01)
                log.debug("AudioOutput._play('%s'): WAV playback complete", word)
                return
            except Exception as exc:
                log.error("pygame load+play error: %s", exc)

        # 3 — direct TTS
        log.debug("AudioOutput._play('%s'): falling back to direct TTS", word)
        if _TTS:
            try:
                _tts.say(word.lower())
                _tts.runAndWait()
                log.debug("AudioOutput._play('%s'): TTS complete", word)
                return
            except Exception as exc:
                log.error("TTS error: %s", exc)

        # 4 — terminal
        log.debug("AudioOutput._play('%s'): all audio methods failed — printing to terminal", word)
        print(f"[AUDIO] {word}", flush=True)

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _gen_wav(self, word: str, path: str):
        """Generate WAV via pyttsx3 and save to disk."""
        if not _TTS:
            log.warning("Cannot generate WAV for '%s': pyttsx3 unavailable.", word)
            return
        try:
            _tts.save_to_file(word.lower(), path)
            _tts.runAndWait()
            log.info("WAV generated: %s → %s", word, path)
        except Exception as exc:
            log.error("WAV generation failed for '%s': %s", word, exc)

    def _load_to_cache(self, word: str, path: str):
        if _PG and os.path.exists(path):
            try:
                self._cache[word] = pygame.mixer.Sound(path)
            except Exception:
                pass

    def _wav_path(self, word: str) -> str:
        """
        V2 words (YES/NO) use audio/.
        All other words use audio/v3/.
        """
        if self._mode == "v2" or word in config.V2_LABELS:
            fname = f"{word.lower()}.wav"
            return os.path.join(config.AUDIO_DIR, fname)
        return os.path.join(config.V3_AUDIO_DIR, f"{word}.wav")


# ══════════════════════════════════════════════════════════════════════════════
#  V2 one-time WAV generation  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════
def generate_v2_audio():
    """
    Pre-render YES/NO WAV files using pyttsx3.
    Run once on any machine that has speakers, copy audio/ to Pi.
    """
    if not _TTS:
        print("pyttsx3 not available. pip install pyttsx3")
        return
    os.makedirs(config.AUDIO_DIR, exist_ok=True)
    for word in ["YES", "NO"]:
        path = os.path.join(config.AUDIO_DIR, f"{word.lower()}.wav")
        _tts.save_to_file(word.lower(), path)
        _tts.runAndWait()
        print(f"Saved: {path}")
    print("Done. Copy audio/ to Pi.")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    ap = argparse.ArgumentParser(description="NeuroBand Audio Output")
    ap.add_argument("--generate-audio", action="store_true",
                    help="Generate v2 YES/NO WAV files")
    ap.add_argument("--test-word", default="",
                    help="Speak a word to test audio output")
    ap.add_argument("--v2", action="store_true", help="Use v2 mode")
    args = ap.parse_args()

    if args.generate_audio:
        generate_v2_audio()
    elif args.test_word:
        mode = "v2" if args.v2 else "v3"
        ao   = AudioOutput(mode=mode)
        ao.speak(args.test_word.upper())
        time.sleep(3)
        ao.shutdown()
    else:
        print("Use --generate-audio or --test-word <WORD>")