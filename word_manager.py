"""
word_manager.py — Word List & Dataset Manager for NeuroBand v3
Place in: /home/pi/neuroband/word_manager.py

Handles:
  - Active word list (up to 10 words, persisted as JSON)
  - Per-word EEG feature dataset (one .npz file per word)
  - Sample counts and readiness checks

Dataset layout:
  logs/v3/
    HELLO.npz   ← X: (n, N_FEATURES)   y: (n,) all "HELLO"
    HELP.npz
    YES.npz
    ...
"""

import os
import logging
import numpy as np
import config as cfg

log = logging.getLogger(__name__)


class WordManager:

    def __init__(self):
        log.debug("WordManager.__init__(): creating dataset dir '%s'", cfg.V3_DATASET_DIR)
        os.makedirs(cfg.V3_DATASET_DIR, exist_ok=True)
        self._words = cfg.load_words()
        log.info("WordManager ready — words: %s", self._words)

    # ── Word list ──────────────────────────────────────────────────────────────
    @property
    def words(self) -> list:
        return list(self._words)

    def add_word(self, word: str) -> tuple:
        word = word.upper().strip()
        log.debug("WordManager.add_word('%s'): current list=%s", word, self._words)
        if not word:
            log.warning("WordManager.add_word(): rejected — empty word")
            return False, "Word cannot be empty."
        if word in self._words:
            log.debug("WordManager.add_word('%s'): rejected — already exists", word)
            return False, f"'{word}' already exists."
        if len(self._words) >= cfg.V3_MAX_WORDS:
            log.warning("WordManager.add_word('%s'): rejected — max words (%d) reached",
                        word, cfg.V3_MAX_WORDS)
            return False, f"Maximum {cfg.V3_MAX_WORDS} words reached."
        self._words.append(word)
        cfg.save_words(self._words)
        log.info("Word added: %s — list now: %s", word, self._words)
        return True, f"'{word}' added."

    def remove_word(self, word: str) -> tuple:
        word = word.upper().strip()
        log.debug("WordManager.remove_word('%s')", word)
        if word not in self._words:
            log.warning("WordManager.remove_word('%s'): not found", word)
            return False, f"'{word}' not found."
        self._words.remove(word)
        cfg.save_words(self._words)
        path = self._path(word)
        if os.path.exists(path):
            os.remove(path)
            log.debug("WordManager.remove_word('%s'): dataset file deleted", word)
        log.info("Word removed: %s — list now: %s", word, self._words)
        return True, f"'{word}' removed."

    # ── Dataset ───────────────────────────────────────────────────────────────
    def append_sample(self, word: str, fv: np.ndarray):
        word = word.upper().strip()
        path = self._path(word)
        log.debug("WordManager.append_sample('%s'): fv shape=%s, path='%s'", word, fv.shape, path)
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            X = np.vstack([d["X"], fv.reshape(1, -1)])
            y = np.append(d["y"], word)
        else:
            X = fv.reshape(1, -1)
            y = np.array([word])
        np.savez(path, X=X.astype(np.float32), y=y)
        log.debug("WordManager.append_sample('%s'): saved, total samples=%d", word, len(X))

    def get_all_data(self):
        """Return (X, y) across all words, or None if not enough data."""
        log.debug("WordManager.get_all_data(): collecting data for words=%s", self._words)
        Xp, yp = [], []
        for w in self._words:
            p = self._path(w)
            if os.path.exists(p):
                d = np.load(p, allow_pickle=True)
                n = len(d["X"])
                if n >= cfg.V3_MIN_SAMPLES:
                    Xp.append(d["X"])
                    yp.append(d["y"])
                    log.debug("WordManager.get_all_data(): '%s' — %d samples included", w, n)
                else:
                    log.debug("WordManager.get_all_data(): '%s' — only %d samples, need %d — skipped",
                              w, n, cfg.V3_MIN_SAMPLES)
            else:
                log.debug("WordManager.get_all_data(): '%s' — no dataset file", w)
        if len(Xp) < 2:
            log.warning("WordManager.get_all_data(): not enough classes (%d) for training", len(Xp))
            return None
        log.debug("WordManager.get_all_data(): returning %d classes, total rows=%d",
                  len(Xp), sum(len(x) for x in Xp))
        return np.vstack(Xp), np.concatenate(yp)

    def sample_counts(self) -> dict:
        counts = {}
        for w in self._words:
            p = self._path(w)
            if os.path.exists(p):
                d = np.load(p, allow_pickle=True)
                counts[w] = int(len(d["X"]))
            else:
                counts[w] = 0
        log.debug("WordManager.sample_counts(): %s", counts)
        return counts

    def clear_word(self, word: str) -> tuple:
        log.debug("WordManager.clear_word('%s')", word)
        path = self._path(word.upper().strip())
        if os.path.exists(path):
            os.remove(path)
            log.info("WordManager.clear_word('%s'): dataset cleared", word)
            return True, f"Data cleared for '{word}'."
        log.debug("WordManager.clear_word('%s'): no dataset file to clear", word)
        return False, f"No data for '{word}'."

    def total_samples(self) -> int:
        return sum(self.sample_counts().values())

    def ready_words(self) -> list:
        return [w for w, n in self.sample_counts().items() if n >= cfg.V3_MIN_SAMPLES]

    def status(self) -> dict:
        counts = self.sample_counts()
        return {
            "words":        self._words,
            "counts":       counts,
            "total":        sum(counts.values()),
            "ready_words":  self.ready_words(),
            "max_words":    cfg.V3_MAX_WORDS,
        }

    def _path(self, word: str) -> str:
        return os.path.join(cfg.V3_DATASET_DIR, f"{word}.npz")