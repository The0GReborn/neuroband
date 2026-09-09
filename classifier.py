"""
classifier.py  —  NeuroBand Unified Classifier  (v5.4)
Place in:  /home/pi/neuroband/classifier.py

════════════════════════════════════════════════════════════════════════════════
 Architecture overview
════════════════════════════════════════════════════════════════════════════════

 V5 / V3 / V4 mode (default)
   Primary   —  k-NN  (live training, session-local geometry)
   Fallback  —  LDA   (persisted cross-session model)
   Fallback  —  RuleBasedClassifier  (zero-data heuristic)

 V2 mode (--v2 flag)
   Primary   —  LDA
   Fallback  —  RuleBasedClassifier

 Temporal fusion  (v5)
   EMAVoter — exponential moving average belief state, one scalar per class.
   No fixed window to tune.  Naturally adaptive: clear signals converge fast,
   ambiguous signals accumulate evidence slowly.  REST votes decay all beliefs
   instead of being excluded post-hoc, which eliminates the "silent buffer
   poisoning" problem of sliding-window designs.

 Training
   k-NN: knn.update(word_manager) once per completed training session.
   LDA : python classifier.py --train

════════════════════════════════════════════════════════════════════════════════
 Changelog
════════════════════════════════════════════════════════════════════════════════

 v3.5
   FIX 1  Jittered augmentation replaces exact-duplicate oversampling.
   FIX 2  DemocracyVoter separates decision_set / confidence denominator.
   FIX 3  KNNClassifier.is_stale renamed to is_untrained.

 v4.0
   UPG 1  Adaptive per-feature jitter sigma (fraction × std).
   UPG 2  Metric selection: euclidean / cosine / pca_euclidean.
   UPG 3  Confidence-adaptive sliding window + cooldown.

 v5.0
   FIX 4  Replaced sliding-window voter with EMAVoter.
   FIX 5  Cosine pipeline drops StandardScaler before Normalizer.
   FIX 6  Jitter sigma floor changed to percentile-based floor.
   FIX 7  REST handling in EMAVoter: decays ALL class beliefs.
   FIX 8  Cooldown computed from EMA belief state.

 v5.1
   FIX 9   RuleBasedClassifier beta normalisation fixed (soft-sigmoid).
   FIX 10  Unknown label logged once before being treated as REST.
   FIX 11  pickle replaced with joblib for model serialisation.
   FIX 12  EMAVoter REST-detection threshold made configurable.
   FIX 13  Belief floor prevents beliefs collapsing to numerical zero.

 v5.2
   FIX 14  predict_one() now applies CONFIDENCE_THRESH gate on k-NN path.
   FIX 15  KNNClassifier.predict_proba_dict() added for soft EMAVoter feeding.
   FIX 16  EMAVoter beliefs initialised consistently with reset().
   FIX 17  KNNClassifier.update() documents class-list sync contract.

 v5.3
   FIX 18  EMAVoter minimum-window warmup guard (V5_EMA_MIN_WINDOWS, default 8).
           Prevents conf=1.0 emission on frame 1 caused by the belief-share
           ratio (nr[winner]/total_nr) collapsing to 1.0 when only one class
           has a non-zero belief after reset.
   FIX 19  EMAVoter._window_count reset to 0 in reset() so every new utterance
           starts a fresh warmup period.

 v5.4
   FIX-CLS-1  EMAVoter: absolute belief gate added to push().
              Even after the warmup window the belief-share ratio can be 1.0
              when all beliefs are tiny (e.g. 0.001 vs 0.000).  A second gate
              requiring nr[winner] >= V5_EMA_MIN_BELIEF_ABS (default 0.05)
              prevents these phantom commits.

   FIX-CLS-2  EMAVoter: alpha now read from config (V5_EMA_ALPHA) instead of
              being hard-coded to 0.1.  The hard-code silently ignored the
              config value that was documented in the class docstring.

   FIX-CLS-3  EMAVoter: added flush_rest(n) public method.
              Injects n synthetic REST frames to drain all beliefs before the
              next utterance starts.  Used by simulator.py between words to
              prevent YES carry-over contaminating the next word's warmup.

   FIX-CLS-4  EMAVoter: reset() now also zeros _last_out so the cooldown
              timer does not suppress the very first commit of a new utterance
              when words follow each other closely.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import pickle
import time
from typing import Optional

import numpy as np

import config

log = logging.getLogger(__name__)

# FIX 11: prefer joblib for safer numpy/sklearn serialisation; fall back to pickle
try:
    import joblib as _joblib
    _JOBLIB = True
    log.debug("classifier: joblib available — using for model serialisation")
except ImportError:
    _joblib = None   # type: ignore[assignment]
    _JOBLIB = False
    log.debug("classifier: joblib not available — falling back to pickle")

try:
    from sklearn.neighbors             import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble              import RandomForestClassifier
    from sklearn.decomposition         import PCA
    from sklearn.pipeline              import Pipeline
    from sklearn.preprocessing        import StandardScaler, Normalizer
    from sklearn.model_selection       import StratifiedKFold, cross_val_score
    from sklearn.metrics               import classification_report
    _SK = True
    log.debug("classifier: scikit-learn loaded successfully")
except ImportError:
    _SK = False
    log.warning("scikit-learn not installed — rule-based fallback only.")


# ─────────────────────────────────────────────────────────────────────────────
#  Config helper
# ─────────────────────────────────────────────────────────────────────────────

def _cfg(attr: str, default):
    """Return config.<attr> if it exists, else *default*."""
    return getattr(config, attr, default)


# ─────────────────────────────────────────────────────────────────────────────
#  Model serialisation helpers  (FIX 11)
# ─────────────────────────────────────────────────────────────────────────────

def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_save(obj, path: str):
    """Save *obj* to *path* using joblib if available, else pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if _JOBLIB:
        _joblib.dump(obj, path)
        log.debug("_model_save: joblib.dump -> %s", path)
    else:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        log.debug("_model_save: pickle.dump -> %s", path)


def _model_load(path: str):
    """
    Load from *path*.  When V5_MODEL_HASH_CHECK is True in config, compare
    the file's SHA-256 against a sidecar <path>.sha256 file and raise
    ValueError on mismatch before deserialising anything.
    """
    log.debug("_model_load: loading from %s", path)
    if _cfg("V5_MODEL_HASH_CHECK", False):
        sidecar = path + ".sha256"
        if os.path.exists(sidecar):
            with open(sidecar) as f:
                expected = f.read().strip()
            actual = _file_sha256(path)
            if actual != expected:
                raise ValueError(
                    f"Model file integrity check failed for {path}\n"
                    f"  expected {expected}\n"
                    f"  got      {actual}"
                )
            log.debug("_model_load: hash check passed for %s", path)
        else:
            log.warning(
                "V5_MODEL_HASH_CHECK is True but no sidecar found at %s — "
                "skipping hash check.", sidecar
            )
    if _JOBLIB:
        return _joblib.load(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def _model_save_with_hash(obj, path: str):
    """Save model and write a SHA-256 sidecar for integrity checking."""
    _model_save(obj, path)
    if _cfg("V5_MODEL_HASH_CHECK", False):
        digest = _file_sha256(path)
        with open(path + ".sha256", "w") as f:
            f.write(digest + "\n")
        log.debug("Model hash written: %s -> %s", path + ".sha256", digest[:12])


# ═════════════════════════════════════════════════════════════════════════════
#  Rule-Based Fallback  (v2 — no sklearn required)
# ═════════════════════════════════════════════════════════════════════════════

class RuleBasedClassifier:
    """
    Alpha-asymmetry + beta heuristic.
    Zero-data last-resort fallback — works even without sklearn.
    """

    # Empirical beta threshold (µV² or normalised units).
    _BETA_THRESHOLD: float = 0.3

    def predict_proba(self, fv: np.ndarray) -> dict:
        asym     = fv[24]
        beta_avg = (fv[9] + fv[21]) / 2.0
        # FIX 9: soft-sigmoid x/(1+|x|) gives genuine gradient over beta range
        beta_norm = beta_avg / (1.0 + abs(beta_avg))
        log.debug("RuleBasedClassifier.predict_proba(): asym=%.4f beta_avg=%.4f beta_norm=%.4f",
                  asym, beta_avg, beta_norm)
        if asym > 0.15 and beta_norm > self._BETA_THRESHOLD:
            result = {"YES": 0.75, "NO": 0.15, "REST": 0.10}
        elif asym < -0.15 and beta_norm > self._BETA_THRESHOLD:
            result = {"YES": 0.10, "NO": 0.75, "REST": 0.15}
        else:
            result = {"YES": 0.10, "NO": 0.10, "REST": 0.80}
        log.debug("RuleBasedClassifier.predict_proba(): result=%s", result)
        return result

    def predict(self, fv: np.ndarray) -> str:
        p = self.predict_proba(fv)
        winner = max(p, key=p.get)
        log.debug("RuleBasedClassifier.predict(): winner='%s'", winner)
        return winner


# ═════════════════════════════════════════════════════════════════════════════
#  LDA Classifier  (v2 primary / v3–v5 fallback)
# ═════════════════════════════════════════════════════════════════════════════

class LDAClassifier:
    """
    Original v2 LDA / RandomForest classifier.
    Loaded from MODEL_PATH; falls back to RuleBasedClassifier if absent.
    """

    LABELS = config.V2_LABELS

    def __init__(self):
        self._model    = None
        self._fallback = RuleBasedClassifier()
        self._load()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self):
        log.debug("LDAClassifier._load(): checking for model at '%s'", config.MODEL_PATH)
        if not os.path.exists(config.MODEL_PATH):
            log.debug("LDAClassifier._load(): no saved model found")
            return
        try:
            self._model = _model_load(config.MODEL_PATH)   # FIX 11
            log.info("LDA model loaded <- %s", config.MODEL_PATH)
        except Exception as exc:
            log.warning("LDA load failed: %s", exc)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def predict(self, fv: np.ndarray) -> tuple[str, float]:
        log.debug("LDAClassifier.predict(): fv shape=%s", fv.shape if fv is not None else "None")
        if fv is None:
            log.debug("LDAClassifier.predict(): fv is None — returning REST")
            return "REST", 0.0
        fv2d = fv.reshape(1, -1)
        if self._model is not None and _SK:
            try:
                proba = self._model.predict_proba(fv2d)[0]
                idx   = int(np.argmax(proba))
                label = self.LABELS[idx]
                conf  = float(proba[idx])
                log.debug("LDAClassifier.predict(): label='%s' conf=%.3f (raw proba=%s)",
                          label, conf, [f"{p:.3f}" for p in proba])
            except Exception as exc:
                log.error("LDA predict error: %s", exc)
                return "REST", 0.0
        else:
            log.debug("LDAClassifier.predict(): model not ready — using RuleBased fallback")
            pd    = self._fallback.predict_proba(fv)
            label = max(pd, key=pd.get)
            conf  = pd[label]

        if conf < config.CONFIDENCE_THRESH:
            log.debug("LDAClassifier.predict(): conf=%.3f below CONFIDENCE_THRESH=%.3f — returning REST",
                      conf, config.CONFIDENCE_THRESH)
            return "REST", conf
        return label, conf

    def train(self, X: np.ndarray, y: np.ndarray, use_rf: bool = False):
        log.info("LDAClassifier.train(): starting — %d samples, use_rf=%s", len(X), use_rf)
        if not _SK:
            raise RuntimeError("scikit-learn required for LDA training.")
        label_to_idx = {l: i for i, l in enumerate(self.LABELS)}
        y_int = np.array([label_to_idx[l] for l in y])
        clf   = (
            RandomForestClassifier(
                n_estimators=100, max_depth=8,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
            if use_rf
            else LinearDiscriminantAnalysis(solver="svd")
        )
        log.debug("LDAClassifier.train(): classifier type=%s", type(clf).__name__)
        pipe   = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X, y_int, cv=cv, scoring="accuracy")
        print(f"CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
        log.info("LDAClassifier.train(): CV accuracy=%.3f +/- %.3f", scores.mean(), scores.std())
        pipe.fit(X, y_int)
        self._model = pipe
        print(classification_report(y_int, pipe.predict(X), target_names=self.LABELS))
        self.save()
        log.info("LDAClassifier.train(): complete")

    def save(self):
        log.debug("LDAClassifier.save(): saving to '%s'", config.MODEL_PATH)
        _model_save_with_hash(self._model, config.MODEL_PATH)   # FIX 11
        log.info("LDA model saved -> %s", config.MODEL_PATH)

    @property
    def is_ready(self) -> bool:
        return self._model is not None


# ═════════════════════════════════════════════════════════════════════════════
#  k-NN Classifier  (v3/v4/v5 primary)
# ═════════════════════════════════════════════════════════════════════════════

class KNNClassifier:
    """
    Session-local k-NN fitted once per training session.
    Target inference budget: < 2 ms on Pi Zero 2 W.
    Falls back: LDA -> RuleBased.

    ── Jitter augmentation (adaptive sigma, FIX 6) ──────────────────────────
    Minority classes are oversampled with per-feature Gaussian jitter:

        sigma_j = max(floor_j, V3_JITTER_FRACTION * std(X[:, j]))

    where floor_j is the 5th percentile of all nonzero sigmas (not an
    absolute constant).  This prevents dead/near-constant channels (std ≈ 0)
    from receiving disproportionate artificial noise.

    ── Metric selection (FIX 5) ─────────────────────────────────────────────
    config.V3_KNN_METRIC chooses the distance pipeline:

      "euclidean"  (default)
          StandardScaler -> KNN(euclidean, ball_tree)

      "cosine"
          Normalizer(l2) -> KNN(cosine, brute)
          FIX 5: StandardScaler OMITTED — preserves direction vectors.

      "pca_euclidean"
          StandardScaler -> PCA(whiten, V3_PCA_COMPONENTS) -> KNN(euclidean)
          Recommended: equalises class covariance and fixes narrow decision
          regions (e.g. HELLO was only reachable by 2.5% of input space in
          raw euclidean geometry).
    """

    _MIN_ROWS_FOR_ADAPTIVE_SIGMA = 4

    def __init__(self):
        self._pipe:      Optional[Pipeline] = None
        self._fitted:    bool               = False
        self._classes:   list               = []
        self._n:         int                = 0
        self._fitted_at: Optional[float]    = None
        self._lda_fallback                  = LDAClassifier()
        log.debug("KNNClassifier.__init__(): initialised — loading saved model if available")
        self.load()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _jitter_sigmas(self, X: np.ndarray) -> np.ndarray:
        """
        FIX 6: Per-feature jitter sigma with percentile-based floor.
        Dead channels (std=0) get the floor rather than zero or an arbitrary constant.
        """
        if len(X) < self._MIN_ROWS_FOR_ADAPTIVE_SIGMA:
            sigma = float(_cfg("V3_JITTER_SIGMA", 0.01))
            log.debug("_jitter_sigmas: too few rows (%d) — using fixed sigma=%.4f", len(X), sigma)
            return np.full(X.shape[1], sigma)

        fraction = float(_cfg("V3_JITTER_FRACTION", 0.05))
        raw      = fraction * X.std(axis=0)

        nonzero  = raw[raw > 0]
        floor    = float(np.percentile(nonzero, 5)) if len(nonzero) else 1e-9
        floor    = max(floor, 1e-9)
        sigmas   = np.where(raw < floor, floor, raw)
        log.debug("_jitter_sigmas: fraction=%.3f floor=%.2e p5=%.4f p95=%.4f",
                  fraction, floor,
                  float(np.percentile(sigmas, 5)), float(np.percentile(sigmas, 95)))
        return sigmas

    def _balance_with_jitter(
        self,
        X: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Oversample minority classes with per-feature jittered copies."""
        classes, counts = np.unique(y, return_counts=True)
        freq    = dict(zip(classes, counts))
        max_cnt = int(counts.max())
        sigmas  = self._jitter_sigmas(X)

        log.debug("_balance_with_jitter: class distribution before balancing: %s", freq)

        X_parts, y_parts = [X], [y]
        for cls in classes:
            n_need = max_cnt - freq[cls]
            if n_need <= 0:
                continue
            log.debug("_balance_with_jitter: oversampling '%s' by %d jittered copies", cls, n_need)
            idx    = np.where(y == cls)[0]
            extra  = rng.choice(idx, size=n_need, replace=True)
            jitter = rng.standard_normal(X[extra].shape) * sigmas
            X_parts.append(X[extra] + jitter)
            y_parts.append(y[extra])

        return np.vstack(X_parts), np.concatenate(y_parts)

    def _build_pipeline(self, k: int) -> Pipeline:
        """
        FIX 5: cosine pipeline drops StandardScaler.

        euclidean     : StandardScaler -> KNN(euclidean, ball_tree)
        cosine        : Normalizer(l2) -> KNN(cosine,    brute)
        pca_euclidean : StandardScaler -> PCA(whiten)   -> KNN(euclidean, ball_tree)
        """
        metric  = str(_cfg("V3_KNN_METRIC", "euclidean")).lower()
        log.debug("_build_pipeline: metric='%s' k=%d", metric, k)

        knn_euc = KNeighborsClassifier(
            n_neighbors=k, weights="distance",
            metric="euclidean", algorithm="ball_tree", n_jobs=1,
        )

        if metric == "cosine":
            log.debug("_build_pipeline: using cosine pipeline (Normalizer -> KNN, no StandardScaler)")
            return Pipeline([
                ("norm", Normalizer(norm="l2")),
                ("knn",  KNeighborsClassifier(
                    n_neighbors=k, weights="distance",
                    metric="cosine", algorithm="brute", n_jobs=1,
                )),
            ])

        if metric == "pca_euclidean":
            n_comp = _cfg("V3_PCA_COMPONENTS", 0.95)
            log.debug("_build_pipeline: using pca_euclidean pipeline (n_components=%s)", n_comp)
            return Pipeline([
                ("sc",  StandardScaler()),
                ("pca", PCA(n_components=n_comp, whiten=True, random_state=42)),
                ("knn", knn_euc),
            ])

        if metric != "euclidean":
            log.warning("Unknown V3_KNN_METRIC %r — using euclidean.", metric)
        log.debug("_build_pipeline: using euclidean pipeline (StandardScaler -> KNN)")
        return Pipeline([("sc", StandardScaler()), ("knn", knn_euc)])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray):
        log.debug("KNNClassifier.fit(): starting — X shape=%s, classes=%s",
                  X.shape, list(np.unique(y)))
        if not _SK or len(X) == 0:
            log.warning("KNNClassifier.fit(): skipped — sklearn unavailable or empty dataset")
            return

        rng          = np.random.default_rng(seed=42)
        X_bal, y_bal = self._balance_with_jitter(X, y, rng)

        classes, counts = np.unique(y, return_counts=True)
        freq            = dict(zip(classes, counts))
        k               = max(3, min(config.V3_KNN_K, len(X_bal)))
        metric          = str(_cfg("V3_KNN_METRIC", "euclidean")).lower()

        log.debug("KNNClassifier.fit(): balanced dataset — %d raw -> %d balanced, k=%d, metric=%s",
                  len(X), len(X_bal), k, metric)

        self._pipe      = self._build_pipeline(k)
        self._pipe.fit(X_bal, y_bal)
        self._classes   = list(classes)
        self._fitted    = True
        self._n         = len(X)
        self._fitted_at = time.time()

        sigmas     = self._jitter_sigmas(X)
        counts_str = ", ".join(f"{c}:{freq[c]}" for c in classes)
        log.info(
            "k-NN fitted: %d raw -> %d balanced | metric=%s k=%d [%s] "
            "jitter-sigma p5=%.4f p95=%.4f",
            len(X), len(X_bal), metric, k, counts_str,
            float(np.percentile(sigmas, 5)), float(np.percentile(sigmas, 95)),
        )
        log.debug("KNNClassifier.fit(): complete — classes=%s fitted_at=%.0f",
                  self._classes, self._fitted_at)

    def update(self, word_manager) -> bool:
        """
        Refit from WordManager.  Call once at end of each training session.
        Returns True on success, False if data is insufficient.

        IMPORTANT — class-list sync (FIX 17)
        After a successful update the k-NN may have a different set of class
        labels (e.g. a new word was added).  The EMAVoter must be reset with
        the new class list:

            if knn.update(word_manager):
                voter.reset(classes=knn.classes + ["REST"])
        """
        log.debug("KNNClassifier.update(): fetching all data from WordManager")
        data = word_manager.get_all_data()
        if data is None:
            log.warning("KNNClassifier.update(): not enough data across word classes — refit skipped")
            return False
        log.debug("KNNClassifier.update(): got %d samples — refitting", len(data[0]))
        self.fit(data[0], data[1])
        self.save()
        log.info("KNNClassifier.update(): refit complete — classes=%s", self._classes)
        return True

    @property
    def classes(self) -> list:
        """Current class labels the k-NN was fitted on (excluding REST)."""
        return list(self._classes)

    def predict_one(self, fv: np.ndarray) -> tuple[str, float]:
        """
        Returns (label, confidence) from argmax — used by UnifiedClassifier.predict().
        Falls back: LDA -> ("REST", 0.0).

        FIX 14: applies CONFIDENCE_THRESH gate (was missing from k-NN path in v5.1).
        Call predict_proba_dict() instead when feeding EMAVoter directly.
        """
        log.debug("KNNClassifier.predict_one(): fv shape=%s", fv.shape)
        if not self._fitted or self._pipe is None:
            log.debug("KNNClassifier.predict_one(): k-NN not ready — checking LDA fallback")
            if self._lda_fallback.is_ready:
                log.debug("KNNClassifier.predict_one(): k-NN not ready — using LDA fallback")
                return self._lda_fallback.predict(fv)
            log.debug("KNNClassifier.predict_one(): LDA also not ready — returning REST")
            return "REST", 0.0
        try:
            p    = self._pipe.predict_proba(fv.reshape(1, -1))[0]
            idx  = int(np.argmax(p))
            conf = float(p[idx])
            log.debug("KNNClassifier.predict_one(): raw proba=%s argmax_idx=%d conf=%.3f",
                      [f"{v:.3f}" for v in p], idx, conf)
            # FIX 14: confidence gate — low-confidence argmax becomes REST
            if conf < float(_cfg("CONFIDENCE_THRESH", 0.35)):
                log.debug("KNNClassifier.predict_one(): conf=%.3f below CONFIDENCE_THRESH=%.3f — returning REST",
                          conf, float(_cfg("CONFIDENCE_THRESH", 0.35)))
                return "REST", conf
            label = str(self._pipe.classes_[idx])
            log.debug("KNNClassifier.predict_one(): label='%s' conf=%.3f", label, conf)
            return label, conf
        except Exception as exc:
            log.error("k-NN predict error: %s", exc)
            return "REST", 0.0

    def predict_proba_dict(self, fv: np.ndarray) -> dict[str, float]:
        """
        Return full probability dict {class: probability, ...} for this frame.

        v5.4 PATCH:
          ✔ Always returns ALL classes (EMA-safe)
          ✔ Always normalized
          ✔ REST estimated from uncertainty
          ✔ Robust fallback handling
        """
        log.debug("KNNClassifier.predict_proba_dict(): fv shape=%s", fv.shape)

        # ─────────────────────────────────────────────────────────────
        # Fallback: k-NN not ready → use LDA or REST
        # ─────────────────────────────────────────────────────────────
        if not self._fitted or self._pipe is None:
            log.debug("predict_proba_dict(): k-NN not ready — trying LDA fallback")

            if self._lda_fallback.is_ready:
                label, conf = self._lda_fallback.predict(fv)

                # FULL distribution (FIX)
                result = {c: 0.0 for c in (self._classes or [label])}
                result[label] = float(conf)

                # REST from uncertainty
                result["REST"] = max(0.0, 1.0 - conf)

                # Normalize
                total = sum(result.values())
                if total > 0:
                    result = {k: v / total for k, v in result.items()}

                log.debug("predict_proba_dict(): LDA fallback result=%s", result)
                return result

            # No model at all → pure REST
            return {"REST": 1.0}

        # ─────────────────────────────────────────────────────────────
        # Normal k-NN path
        # ─────────────────────────────────────────────────────────────
        try:
            proba   = self._pipe.predict_proba(fv.reshape(1, -1))[0]
            classes = [str(c) for c in self._pipe.classes_]

            # Base result
            result = dict(zip(classes, (float(p) for p in proba)))

            # ── FIX 1: Ensure ALL expected classes exist ───────────────
            for c in self._classes:
                if c not in result:
                    result[c] = 0.0

            # ── FIX 2: REST from uncertainty ──────────────────────────
            max_p = max(result.values()) if result else 0.0
            result["REST"] = max(0.0, 1.0 - max_p)

            # ── FIX 3: Normalize distribution ─────────────────────────
            total = sum(result.values())
            if total > 0:
                result = {k: v / total for k, v in result.items()}

            log.debug("predict_proba_dict(): proba=%s",
                      {k: f"{v:.3f}" for k, v in result.items()})

            return result

        except Exception as exc:
            log.error("k-NN predict_proba_dict error: %s", exc)
            return {"REST": 1.0}

    def save(self):
        if not self._fitted:
            log.debug("KNNClassifier.save(): skipped — not fitted yet")
            return
        log.debug("KNNClassifier.save(): saving to '%s'", config.V3_KNN_MODEL_PATH)
        try:
            payload = {
                "pipe":      self._pipe,
                "classes":   self._classes,
                "n":         self._n,
                "fitted_at": self._fitted_at,
            }
            _model_save_with_hash(payload, config.V3_KNN_MODEL_PATH)  # FIX 11
            log.debug("KNNClassifier.save(): done — classes=%s n=%d", self._classes, self._n)
        except Exception as exc:
            log.error("k-NN save failed: %s", exc)

    def load(self) -> bool:
        log.debug("KNNClassifier.load(): checking for model at '%s'", config.V3_KNN_MODEL_PATH)
        if not os.path.exists(config.V3_KNN_MODEL_PATH):
            log.debug("KNNClassifier.load(): no saved model found")
            return False
        try:
            d               = _model_load(config.V3_KNN_MODEL_PATH)   # FIX 11
            self._pipe      = d["pipe"]
            self._classes   = d["classes"]
            self._n         = d["n"]
            self._fitted    = True
            self._fitted_at = d.get("fitted_at", None)
            log.info("k-NN loaded: %d samples, %d classes", self._n, len(self._classes))
            log.debug("KNNClassifier.load(): classes=%s fitted_at=%s",
                      self._classes, self._fitted_at)
            return True
        except Exception as exc:
            log.warning("k-NN load failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._fitted

    @property
    def is_untrained(self) -> bool:
        """True when the model has never been fitted or failed to load."""
        return not self._fitted

    @property
    def is_stale(self) -> bool:
        """Deprecated alias for is_untrained.  Will be removed in v6."""
        log.warning("KNNClassifier.is_stale is deprecated — use is_untrained.")
        return self.is_untrained

    def status(self) -> dict:
        return {
            "fitted":    self._fitted,
            "classes":   self._classes,
            "n_samples": self._n,
            "fitted_at": self._fitted_at,
            "metric":    str(_cfg("V3_KNN_METRIC", "euclidean")),
        }


# ═════════════════════════════════════════════════════════════════════════════
#  EMA Voter  (v5 — replaces DemocracyVoter's sliding window)
# ═════════════════════════════════════════════════════════════════════════════

class EMAVoter:
    """
    Exponential-moving-average belief state for temporal fusion.

    Replaces DemocracyVoter (FIX 4 + FIX 7 + FIX 8).

    Config keys (all optional):
      V5_EMA_ALPHA          = 0.10    smoothing factor  (FIX-CLS-2: was hard-coded)
      V5_EMA_REST_DECAY     = 0.85    per-frame decay during REST
      V5_EMA_CONF_MIN       = 0.65    minimum belief-share to emit
      V5_EMA_MIN_BELIEF_ABS = 0.05    minimum absolute belief for winner (FIX-CLS-1)
      V5_EMA_CONF_HIGH      = 0.80    confidence above which cooldown shrinks
      V5_EMA_CD_SHRINK_RATE = 3.0     cooldown shrink rate above conf_high
      V3_COOLDOWN_SECONDS            base cooldown (shared with v3/v4)
      V5_EMA_COOLDOWN_MIN   = 0.8    minimum cooldown (seconds)
      V5_EMA_REST_MARGIN    = 0.05   margin for REST to win over non-REST
      V5_EMA_BELIEF_FLOOR   = 1e-4   minimum belief value after decay
      V5_EMA_MIN_WINDOWS    = 12     warmup guard (FIX 18, raised from 8)
    """

    def __init__(self, classes: list[str]):
        self._classes    = list(classes)
        # FIX-CLS-2: read alpha from config instead of hard-coding 0.1
        self._alpha      = float(_cfg("V5_EMA_ALPHA", 0.10))
        self._rest_decay = float(_cfg("V5_EMA_REST_DECAY",  0.85))
        self._belief_floor = float(_cfg("V5_EMA_BELIEF_FLOOR", 1e-4))  # FIX 13
        # Initialize to 0.0 to prevent bias from initial high confidence
        self._beliefs    = {c: 0.0 for c in self._classes}
        self._last_out   = 0.0
        self._unknown_labels_seen: set[str] = set()  # FIX 10
        # FIX 18: minimum-window warmup guard — prevents conf=1.0 on frame 1
        self._min_windows  = int(_cfg("V5_EMA_MIN_WINDOWS", 12))
        self._window_count = 0
        log.debug("EMAVoter.__init__(): classes=%s alpha=%.3f rest_decay=%.3f belief_floor=%.2e min_windows=%d",
                  self._classes, self._alpha, self._rest_decay, self._belief_floor, self._min_windows)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _non_rest_beliefs(self) -> dict[str, float]:
        return {c: v for c, v in self._beliefs.items() if c != "REST"}

    def _effective_cooldown(self, conf: float) -> float:
        """FIX 8: cooldown computed from the same belief snapshot as commit."""
        base      = float(config.V3_COOLDOWN_SECONDS)
        minimum   = float(_cfg("V5_EMA_COOLDOWN_MIN", 0.8))
        threshold = float(_cfg("V5_EMA_CONF_HIGH",    0.80))
        rate      = float(_cfg("V5_EMA_CD_SHRINK_RATE", 3.0))
        excess    = max(0.0, conf - threshold)
        cd        = max(minimum, base * (1.0 - rate * excess))
        log.debug("_effective_cooldown: conf=%.3f base=%.1f excess=%.3f -> cooldown=%.2fs",
                  conf, base, excess, cd)
        return cd

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, proba: dict[str, float]) -> tuple[Optional[str], float]:
        """
        Update belief state with one frame's class probabilities.

        Returns (winner, conf) to emit, or (None, conf) when suppressed.
        """
        # FIX 12: REST wins only when it strictly exceeds best non-REST by margin
        rest_p   = proba.get("REST", 0.0)
        best_nr  = max((v for k, v in proba.items() if k != "REST"), default=0.0)
        margin   = float(_cfg("V5_EMA_REST_MARGIN", 0.05))
        is_rest  = rest_p > best_nr + margin

        log.debug("EMAVoter.push(): rest_p=%.3f best_nr=%.3f margin=%.3f is_rest=%s",
                  rest_p, best_nr, margin, is_rest)

        if is_rest:
            # FIX 7: REST actively decays all beliefs
            for c in self._beliefs:
                self._beliefs[c] *= self._rest_decay
            # FIX 13: apply belief floor after decay
            for c in self._beliefs:
                if self._beliefs[c] < self._belief_floor:
                    self._beliefs[c] = self._belief_floor
            log.debug("EMAVoter.push(): REST decay applied — beliefs=%s",
                      {k: f"{v:.4f}" for k, v in self._beliefs.items()})
        else:
            for c in self._classes:
                p = proba.get(c, 0.0)
                old = self._beliefs[c]
                self._beliefs[c] += self._alpha * (p - old)
            log.debug("EMAVoter.push(): EMA update — beliefs=%s",
                      {k: f"{v:.4f}" for k, v in self._beliefs.items()})

        # ── commit decision ──────────────────────────────────────────
        # FIX 18: warmup guard — suppress output until enough frames have
        # accumulated so the belief-share ratio is meaningful.
        self._window_count += 1
        if self._window_count < self._min_windows:
            log.debug("EMAVoter.push(): suppressed — warming up (%d/%d)",
                      self._window_count, self._min_windows)
            return None, 0.0

        nr = self._non_rest_beliefs()
        if not nr:
            return None, 0.0

        total_nr = sum(nr.values())
        if total_nr <= 0.0:
            return None, 0.0

        winner = max(nr, key=nr.__getitem__)
        conf   = nr[winner] / total_nr
        log.debug("EMAVoter.push(): leader='%s' conf=%.3f total_nr=%.4f", winner, conf, total_nr)

        # Suppress if REST belief exceeds winner belief
        if self._beliefs.get("REST", 0.0) >= nr[winner]:
            log.debug("EMAVoter.push(): suppressed — REST belief (%.4f) >= winner belief (%.4f)",
                      self._beliefs.get("REST", 0.0), nr[winner])
            return None, conf

        # FIX-CLS-1: absolute belief gate — prevent phantom commits when all
        # beliefs are near-zero (conf ratio can be 1.0 with winner=0.001).
        min_abs = float(_cfg("V5_EMA_MIN_BELIEF_ABS", 0.05))
        if nr[winner] < min_abs:
            log.debug("EMAVoter.push(): suppressed — winner abs belief %.4f < min_abs %.4f",
                      nr[winner], min_abs)
            return None, conf

        conf_min = float(_cfg("V5_EMA_CONF_MIN", 0.65))
        if conf < conf_min:
            log.debug("EMAVoter.push(): suppressed — conf=%.3f below conf_min=%.3f", conf, conf_min)
            return None, conf

        # FIX 8: cooldown from same conf as commit check
        now     = time.monotonic()
        elapsed = now - self._last_out
        cd      = self._effective_cooldown(conf)
        if elapsed < cd:
            log.debug("EMAVoter.push(): suppressed — cooldown active (%.1fs remaining)",
                      cd - elapsed)
            return None, conf

        self._last_out = now
        log.info("EMAVoter.push(): OUTPUT '%s' conf=%.3f beliefs=%s",
                 winner, conf, {k: f"{v:.4f}" for k, v in self._beliefs.items()})
        return winner, conf

    def push_label(self, label: str) -> tuple[Optional[str], float]:
        """
        Convenience wrapper: push a hard label as a one-hot probability dict.

        FIX 10: Unknown labels are logged once then treated as REST.
        """
        log.debug("EMAVoter.push_label('%s'): converting to one-hot proba", label)
        proba = {c: 0.0 for c in self._classes}
        if label in proba:
            proba[label] = 1.0
        else:
            # FIX 10: log once per unknown label, then decay as REST
            if label not in self._unknown_labels_seen:
                self._unknown_labels_seen.add(label)
                log.warning(
                    "EMAVoter received unknown label %r — not in class list %s. "
                    "Treating as REST.  Check KNNClassifier.classes vs "
                    "EMAVoter.classes after retraining.",
                    label, self._classes,
                )
            proba["REST"] = 1.0
        return self.push(proba)

    def flush_rest(self, n: int = 5):
        """
        FIX-CLS-3: Inject *n* synthetic pure-REST frames to drain all class
        beliefs before the next utterance starts.

        Use this between words in the simulator (and in main.py between real
        utterances if there is a deliberate pause) so that residual YES/NO
        belief from the previous word does not contaminate the warmup phase
        of the next word.

        Unlike reset(), flush_rest() does NOT restart the warmup counter —
        it is designed to be called *during* an ongoing session where the
        voter is already warmed up and should stay warmed up.

        Example (simulator between words):
            voter.reset()          # restart warmup
            voter.flush_rest(6)    # drain any leftover belief
        """
        rest_proba = {c: 0.0 for c in self._classes}
        rest_proba["REST"] = 1.0
        for _ in range(n):
            # Bypass push() to avoid warmup counter interactions — apply REST
            # decay directly so flush_rest() has no side-effects on cooldown.
            for c in self._beliefs:
                self._beliefs[c] *= self._rest_decay
            for c in self._beliefs:
                if self._beliefs[c] < self._belief_floor:
                    self._beliefs[c] = self._belief_floor
        log.debug("EMAVoter.flush_rest(%d): beliefs after flush=%s", n,
                  {k: f"{v:.4f}" for k, v in self._beliefs.items()})

    def confidence_now(self) -> tuple[str, float]:
        """
        Current leader + belief-share without triggering output or cooldown.
        Returns ("REST", 0.0) if no non-REST belief has accumulated.
        """
        nr = self._non_rest_beliefs()
        if not nr:
            return "REST", 0.0
        total = sum(nr.values())
        if total <= 0.0:
            return "REST", 0.0
        winner = max(nr, key=nr.__getitem__)
        conf   = nr[winner] / total
        log.debug("EMAVoter.confidence_now(): leader='%s' conf=%.3f", winner, conf)
        return winner, conf

    def belief_state(self) -> dict[str, float]:
        """Return a copy of the full belief dict for diagnostics / UI."""
        return dict(self._beliefs)

    def reset(self, classes: Optional[list[str]] = None):
        """
        Reset all beliefs to zero and restart the warmup counter.
        Optionally update the class list — needed when k-NN is refitted
        with a different word set (FIX 17).

        FIX-CLS-4: also resets _last_out so the cooldown timer does not
        suppress the first commit of a new utterance when words arrive
        back-to-back.
        """
        if classes is not None:
            log.debug("EMAVoter.reset(): updating class list %s -> %s", self._classes, classes)
            self._classes = list(classes)
        self._beliefs          = {c: 0.0 for c in self._classes}
        self._last_out         = 0.0          # FIX-CLS-4: reset cooldown timer
        self._window_count     = 0            # FIX 19: fresh warmup after every reset
        self._unknown_labels_seen = set()     # FIX 10: fresh cache after reset
        log.debug("EMAVoter.reset(): beliefs zeroed, warmup restarted, classes=%s",
                  self._classes)

    # Legacy alias so callers that used DemocracyVoter.clear() still work.
    def clear(self):
        log.debug("EMAVoter.clear(): delegating to reset()")
        self.reset()


# ── Backward-compatibility shim ──────────────────────────────────────────────

class DemocracyVoter(EMAVoter):
    """
    Deprecated.  Replaced by EMAVoter in v5.
    This shim lets existing call sites (main.py, tests) keep working while
    emitting a one-time deprecation warning.

    DemocracyVoter.push(label) accepted a string label;
    EMAVoter.push(proba) accepts a probability dict.
    The shim routes string inputs through push_label() automatically.
    """

    _warned = False

    def __init__(self, *args, **kwargs):
        if not DemocracyVoter._warned:
            log.warning(
                "DemocracyVoter is deprecated and will be removed in v6. "
                "Use EMAVoter instead."
            )
            DemocracyVoter._warned = True
        classes = list(_cfg("V3_CLASSES", ["YES", "NO", "REST"]))
        if "REST" not in classes:
            classes.append("REST")
        log.debug("DemocracyVoter.__init__(): shim init with classes=%s", classes)
        super().__init__(classes=classes)

    def push(self, label_or_proba) -> tuple:  # type: ignore[override]
        if isinstance(label_or_proba, str):
            log.debug("DemocracyVoter.push(): string input '%s' — routing to push_label()", label_or_proba)
            return self.push_label(label_or_proba)
        return super().push(label_or_proba)

    # votes_breakdown kept for UI code that calls it
    def votes_breakdown(self) -> dict:
        return self.belief_state()


# ═════════════════════════════════════════════════════════════════════════════
#  Unified Classifier  —  single interface used by main.py
# ═════════════════════════════════════════════════════════════════════════════

class UnifiedClassifier:
    """
    Single entry point for main.py.

    mode = "v5" / "v4" / "v3"  (default "v5") : k-NN primary, LDA fallback
    mode = "v2"                                : LDA primary, RuleBased fallback

    Recommended usage (v5) — feed SOFT probabilities to EMAVoter:

        clf   = UnifiedClassifier(mode="v5")
        voter = EMAVoter(classes=["YES", "NO", "REST"])

        # soft proba path — preserves uncertainty
        proba        = clf.knn.predict_proba_dict(fv)
        winner, conf = voter.push(proba)
        if winner:
            speak(winner)

        # After retraining — keep voter class list in sync (FIX 17):
        if clf.knn.update(word_manager):
            voter.reset(classes=clf.knn.classes + ["REST"])

    Hard-label path (simpler, loses probability information):

        label, raw_conf = clf.predict(fv)
        winner, conf    = voter.push_label(label)
    """

    def __init__(self, mode: str = "v5"):
        self._mode = mode.lower()
        log.debug("UnifiedClassifier.__init__(): mode='%s'", self._mode)
        self._clf  = (
            KNNClassifier() if self._mode in ("v3", "v4", "v5")
            else LDAClassifier()
        )
        log.info("UnifiedClassifier: initialised in mode='%s', clf=%s",
                 self._mode, type(self._clf).__name__)

    def predict(self, fv: np.ndarray) -> tuple[str, float]:
        log.debug("UnifiedClassifier.predict(): mode='%s'", self._mode)
        if self._mode in ("v3", "v4", "v5"):
            return self._clf.predict_one(fv)
        return self._clf.predict(fv)

    @property
    def knn(self) -> Optional[KNNClassifier]:
        """Direct access to KNNClassifier (v3/v4/v5 only)."""
        return self._clf if self._mode in ("v3", "v4", "v5") else None

    @property
    def lda(self) -> Optional[LDAClassifier]:
        """Direct access to LDAClassifier."""
        if self._mode == "v2":
            return self._clf
        if self._mode in ("v3", "v4", "v5"):
            return self._clf._lda_fallback
        return None


# ═════════════════════════════════════════════════════════════════════════════
#  V2 CLI training entrypoint
# ═════════════════════════════════════════════════════════════════════════════

def _cli_train(use_rf: bool = False):
    data_path = "logs/training_data.npz"
    if not os.path.exists(data_path):
        print(f"Training data not found at {data_path}.")
        print("Run main.py --v2 --train <LABEL> to collect data first.")
        return
    try:
        data = np.load(data_path, allow_pickle=True)
        X, y = data["X"], data["y"]
    except Exception as exc:
        print(f"[ERROR] Failed to load training data: {exc}")
        return
    print(f"Loaded {len(X)} samples: {dict(zip(*np.unique(y, return_counts=True)))}")
    LDAClassifier().train(X, y, use_rf=use_rf)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="NeuroBand Classifier")
    ap.add_argument("--train", action="store_true", help="Train v2 LDA from logs")
    ap.add_argument("--rf",    action="store_true", help="Use Random Forest instead of LDA")
    ap.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging")
    args = ap.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        log.debug("Debug logging enabled")
    if args.train:
        _cli_train(use_rf=args.rf)