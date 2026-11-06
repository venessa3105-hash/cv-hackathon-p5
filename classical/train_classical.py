# =============================================================================
# P5 - Classical CV baseline: CLAHE + HOG + LBP + HSV colour + geometric
#      features -> StandardScaler -> PCA -> Linear SVM (196 classes)
# Kaggle setup: CPU is fine (no GPU needed), same two datasets as the classifier
# Outputs (in /kaggle/working):
#   classical_model.joblib   - full sklearn pipeline (use in the app)
#   classical_metrics.json   - Top-1/Top-5 per feature group (ablation) + latency
# =============================================================================
import os, json, time
import numpy as np
import pandas as pd
import cv2
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from skimage.feature import hog, local_binary_pattern
from joblib import Parallel, delayed, dump

BASE, OUT = "/kaggle/input", "/kaggle/working"
SEED, VAL_FRAC = 42, 0.10        # SAME split as classifier & detector
SIZE = (128, 128)
PAD = 0.08

# ============================ FEATURES =======================================
# (copy this section into classical/features.py in the repo - the app reuses it)
# CLAHE object is created inside preprocess(): cv2.CLAHE cannot be pickled for parallel workers


def preprocess(img_bgr, box=None, pad=PAD, size=SIZE):
    """Crop to the (padded) box, resize, CLAHE on the L channel.
    Returns enhanced BGR crop and its grayscale version."""
    if box is not None:
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        x1, y1 = int(max(0, x1 - pad * bw)), int(max(0, y1 - pad * bh))
        x2, y2 = int(min(w, x2 + pad * bw)), int(min(h, y2 + pad * bh))
        img_bgr = img_bgr[y1:y2, x1:x2]
    img = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def feat_hog(gray):
    return hog(gray, orientations=9, pixels_per_cell=(16, 16),
               cells_per_block=(2, 2), feature_vector=True).astype(np.float32)


def feat_lbp(gray, grid=2, P=8, R=1):
    lbp = local_binary_pattern(gray, P, R, "uniform")
    h, w = lbp.shape
    out = []
    for i in range(grid):
        for j in range(grid):
            cell = lbp[i * h // grid:(i + 1) * h // grid, j * w // grid:(j + 1) * w // grid]
            hist, _ = np.histogram(cell, bins=P + 2, range=(0, P + 2), density=True)
            out.append(hist)
    return np.concatenate(out).astype(np.float32)


def feat_color(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    return (hist / (hist.sum() + 1e-6)).astype(np.float32)


def feat_geom(gray, box_wh):
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = gray.shape[0] * gray.shape[1]
    if contours:
        c = max(contours, key=cv2.contourArea)
        hu = cv2.HuMoments(cv2.moments(c)).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
        largest = cv2.contourArea(c) / area
        x, y, w, h = cv2.boundingRect(c)
        extent = (w * h) / area
    else:
        hu, largest, extent = np.zeros(7), 0.0, 0.0
    bw, bh = box_wh
    return np.concatenate([[bw / max(bh, 1), edges.mean() / 255, largest, extent,
                            len(contours) / 100], hu]).astype(np.float32)


def extract_all(img_bgr, box=None):
    bgr, gray = preprocess(img_bgr, box)
    wh = (box[2] - box[0], box[3] - box[1]) if box is not None else img_bgr.shape[1::-1]
    return {"hog": feat_hog(gray), "lbp": feat_lbp(gray),
            "color": feat_color(bgr), "geom": feat_geom(gray, wh)}
# ========================== END FEATURES =====================================


# ------------------------- discover files ------------------------------------
files, img_dirs = {}, {}
for root, dirs, fs in os.walk(BASE):
    for f in fs:
        files.setdefault(f, os.path.join(root, f))
    if sum(f.lower().endswith(".jpg") for f in fs) > 1000:
        r = root.lower()
        if "cars_train" in r: img_dirs["train"] = root
        elif "cars_test" in r: img_dirs["test"] = root
test_anno = next(p for f, p in files.items() if "withlabels" in f.lower() and f.lower().endswith(".mat"))


def load_annos(path):
    a = loadmat(path, squeeze_me=True)["annotations"]
    return pd.DataFrame({
        "fname": [str(x) for x in a["fname"]],
        "x1": a["bbox_x1"].astype(int), "y1": a["bbox_y1"].astype(int),
        "x2": a["bbox_x2"].astype(int), "y2": a["bbox_y2"].astype(int),
        "label": a["class"].astype(int) - 1,
    })


train_full = load_annos(files["cars_train_annos.mat"])
test_df = load_annos(test_anno)
train_df, val_df = train_test_split(train_full, test_size=VAL_FRAC,
                                    stratify=train_full["label"], random_state=SEED)
print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")


def _one(path, box):
    return extract_all(cv2.imread(path), box)


def featurize(df, img_dir):
    t0 = time.time()
    feats = Parallel(n_jobs=-1, batch_size=64)(
        delayed(_one)(os.path.join(img_dir, r.fname), (r.x1, r.y1, r.x2, r.y2))
        for r in df.itertuples())
    print(f"  {len(df)} images in {time.time()-t0:.0f}s")
    return {k: np.stack([f[k] for f in feats]) for k in feats[0]}


print("Extracting features...")
Ftr, Fva, Fte = featurize(train_df, img_dirs["train"]), featurize(val_df, img_dirs["train"]), \
    featurize(test_df, img_dirs["test"])
ytr, yva, yte = train_df.label.values, val_df.label.values, test_df.label.values
print({k: v.shape[1] for k, v in Ftr.items()})


def stack(F, groups):
    return np.hstack([F[g] for g in groups])


def make_model(dim, C):
    steps = [StandardScaler()]
    if dim > 256:
        steps.append(PCA(n_components=256, random_state=SEED))
    steps.append(LinearSVC(C=C, dual=False, max_iter=5000))
    return make_pipeline(*steps)


def topk(scores, y, k):
    return float((np.argsort(-scores, 1)[:, :k] == y[:, None]).any(1).mean())


# -------------------- choose C on validation (all features) ------------------
ALL = ["hog", "lbp", "color", "geom"]
best_C, best_val = None, -1
for C in [0.001, 0.01, 0.1]:
    m = make_model(stack(Ftr, ALL).shape[1], C).fit(stack(Ftr, ALL), ytr)
    v = topk(m.decision_function(stack(Fva, ALL)), yva, 1)
    print(f"C={C}: val_top1={v:.4f}")
    if v > best_val:
        best_C, best_val = C, v

# --------------------- ablation by feature group -----------------------------
results = {}
configs = {"hog_only": ["hog"], "lbp_only": ["lbp"], "color_only": ["color"],
           "geom_only": ["geom"], "all_minus_color": ["hog", "lbp", "geom"], "all": ALL}
final_model = None
for name, groups in configs.items():
    Xtr = stack(Ftr, groups)
    m = make_model(Xtr.shape[1], best_C).fit(Xtr, ytr)
    s_va, s_te = m.decision_function(stack(Fva, groups)), m.decision_function(stack(Fte, groups))
    results[name] = {"features": groups, "dim": int(Xtr.shape[1]),
                     "val_top1": topk(s_va, yva, 1), "test_top1": topk(s_te, yte, 1),
                     "test_top5": topk(s_te, yte, 5)}
    print(name, results[name])
    if name == "all":
        final_model = m

# --------------------------- latency (single image) --------------------------
sample = test_df.iloc[:50]
t0 = time.time()
for r in sample.itertuples():
    f = extract_all(cv2.imread(os.path.join(img_dirs["test"], r.fname)), (r.x1, r.y1, r.x2, r.y2))
    final_model.decision_function(np.hstack([f[g] for g in ALL])[None])
latency_ms = (time.time() - t0) / len(sample) * 1000

out = {"method": "CLAHE + HOG + LBP + HSV hist + geometric -> PCA(256) -> LinearSVC",
       "C": best_C, "ablation": results, "latency_ms_per_image_cpu": round(latency_ms, 1)}
json.dump(out, open(f"{OUT}/classical_metrics.json", "w"), indent=2)
dump({"model": final_model, "groups": ALL}, f"{OUT}/classical_model.joblib")
print(json.dumps(out, indent=2))
print("Done. Download classical_model.joblib and classical_metrics.json")
