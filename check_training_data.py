"""
Drop this in your NeuroBand v3 folder and run:
    python check_training_data.py
It prints a full diagnosis of your k-NN training data.
"""
import os, sys
import numpy as np

# Try to load the saved k-NN model and inspect it
try:
    import config
except ImportError:
    print("[ERROR] Can't import config.py — run this from your NeuroBand v3 folder.")
    sys.exit(1)

knn_path = config.V3_KNN_MODEL_PATH
print(f"k-NN model path: {knn_path}")

if not os.path.exists(knn_path):
    print("[ERROR] Model file not found.")
    sys.exit(1)

try:
    import joblib
    d = joblib.load(knn_path)
except Exception:
    import pickle
    with open(knn_path, "rb") as f:
        d = pickle.load(f)

pipe     = d["pipe"]
classes  = d["classes"]
n        = d["n"]

print(f"\n=== k-NN Model Summary ===")
print(f"  Total raw training samples : {n}")
print(f"  Classes                    : {classes}")

# Get the fitted KNN step from the pipeline
knn_step = pipe.named_steps.get("knn") or pipe.steps[-1][1]
X_train  = knn_step._fit_X  # the transformed training data stored in sklearn KNN

# Reconstruct labels — sklearn KNN stores y in classes_
y_train  = knn_step._y      # integer labels

print(f"\n=== Class distribution in fitted (balanced) data ===")
unique, counts = np.unique(y_train, return_counts=True)
for idx, cnt in zip(unique, counts):
    label = pipe.classes_[idx] if hasattr(pipe, 'classes_') else knn_step.classes_[idx]
    print(f"  {label:<12} : {cnt:>4} samples")

print(f"\n=== Centroid distances (Euclidean in transformed space) ===")
from scipy.spatial.distance import cdist
centroids = {}
for idx in unique:
    mask = y_train == idx
    centroids[idx] = X_train[mask].mean(axis=0)

idxs = list(unique)
for i in range(len(idxs)):
    for j in range(i+1, len(idxs)):
        a, b = idxs[i], idxs[j]
        la = knn_step.classes_[a]
        lb = knn_step.classes_[b]
        dist = float(np.linalg.norm(centroids[a] - centroids[b]))
        print(f"  {la} <-> {lb} : {dist:.4f}")

print(f"\n=== Probe: what does k-NN output for a random feature vector? ===")
rng = np.random.default_rng(42)
# Generate 5 random feature vectors and check predicted class distribution
results = {str(c): 0 for c in knn_step.classes_}
for _ in range(200):
    fv = rng.standard_normal(X_train.shape[1])
    p  = pipe.predict_proba(fv.reshape(1, -1))[0]
    winner = str(knn_step.classes_[int(np.argmax(p))])
    results[winner] += 1
print("  Predicted class for 200 random inputs:")
for k, v in sorted(results.items(), key=lambda x: -x[1]):
    bar = "█" * (v // 5)
    print(f"  {k:<12} : {v:>3}  {bar}")

print()