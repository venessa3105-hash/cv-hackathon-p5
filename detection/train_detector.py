# =============================================================================
# P5 - Vehicle detector: YOLO11n fine-tuned on Stanford Cars bounding boxes
# Kaggle setup: Accelerator = GPU (T4), Internet = ON
# Inputs: same two datasets as the classifier notebook
#   eduardo4jesus/stanford-cars-dataset + abdelrahmant11/standford-cars-dataset-meta
# Outputs (in /kaggle/working):
#   det/weights/best.pt  - fine-tuned detector
#   det_metrics.json     - mAP50, mAP50-95, mean IoU (fine-tuned vs COCO zero-shot)
#   det_test_ious.csv    - per-image IoU (use for failure analysis)
# =============================================================================
import os, sys, json, subprocess
os.environ["WANDB_MODE"] = "disabled"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

BASE, OUT = "/kaggle/input", "/kaggle/working"
YOLO_DIR = f"{OUT}/yolo"
SEED, VAL_FRAC = 42, 0.10          # SAME split as the classifier
EPOCHS, IMGSZ, BATCH = 25, 640, 32

# ------------------------- discover files ------------------------------------
files, img_dirs = {}, {}
for root, dirs, fs in os.walk(BASE):
    for f in fs:
        files.setdefault(f, os.path.join(root, f))
    if sum(f.lower().endswith(".jpg") for f in fs) > 1000:
        r = root.lower()
        if "cars_train" in r: img_dirs["train"] = root
        elif "cars_test" in r: img_dirs["test"] = root
train_anno = files["cars_train_annos.mat"]
test_anno = next(p for f, p in files.items() if "withlabels" in f.lower() and f.lower().endswith(".mat"))


def load_annos(path):
    a = loadmat(path, squeeze_me=True)["annotations"]
    return pd.DataFrame({
        "fname": [str(x) for x in a["fname"]],
        "x1": a["bbox_x1"].astype(int), "y1": a["bbox_y1"].astype(int),
        "x2": a["bbox_x2"].astype(int), "y2": a["bbox_y2"].astype(int),
        "label": a["class"].astype(int) - 1,
    })


train_full, test_df = load_annos(train_anno), load_annos(test_anno)
train_df, val_df = train_test_split(train_full, test_size=VAL_FRAC,
                                    stratify=train_full["label"], random_state=SEED)
print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

# ------------------- build YOLO-format dataset (1 class: car) ----------------
def build(split, df, img_dir):
    idir, ldir = f"{YOLO_DIR}/images/{split}", f"{YOLO_DIR}/labels/{split}"
    os.makedirs(idir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
    for r in df.itertuples():
        src = os.path.join(img_dir, r.fname)
        dst = os.path.join(idir, r.fname)
        if not os.path.exists(dst):
            os.symlink(src, dst)
        w, h = Image.open(src).size
        x1, y1, x2, y2 = max(0, r.x1), max(0, r.y1), min(w, r.x2), min(h, r.y2)
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        with open(os.path.join(ldir, r.fname.rsplit(".", 1)[0] + ".txt"), "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


build("train", train_df, img_dirs["train"])
build("val", val_df, img_dirs["train"])
build("test", test_df, img_dirs["test"])

data_yaml = f"{YOLO_DIR}/data.yaml"
with open(data_yaml, "w") as f:
    f.write(f"path: {YOLO_DIR}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: car\n")

# ------------------------------- train ---------------------------------------
model = YOLO("yolo11n.pt")  # COCO-pretrained
model.train(data=data_yaml, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=0,
            workers=4, project=OUT, name="det", exist_ok=True, plots=True, seed=SEED)
best = YOLO(f"{OUT}/det/weights/best.pt")

# --------------------------- evaluation --------------------------------------
m = best.val(data=data_yaml, split="test", imgsz=IMGSZ, batch=BATCH, device=0, plots=True)


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def mean_iou(det_model, classes=None):
    """Top-confidence box per image vs ground-truth box."""
    paths = [os.path.join(img_dirs["test"], f) for f in test_df.fname]
    ious, confs = [], []
    for r, gt in zip(det_model.predict(paths, stream=True, imgsz=IMGSZ, conf=0.25, device=0,
                                       classes=classes, verbose=False, batch=BATCH),
                     test_df[["x1", "y1", "x2", "y2"]].values):
        if len(r.boxes) == 0:
            ious.append(0.0); confs.append(0.0); continue
        k = int(r.boxes.conf.argmax())
        ious.append(iou(r.boxes.xyxy[k].tolist(), gt)); confs.append(float(r.boxes.conf[k]))
    return np.array(ious), np.array(confs)


ft_iou, ft_conf = mean_iou(best)
# ablation: COCO zero-shot (car=2, bus=5, truck=7) without fine-tuning
zs_iou, _ = mean_iou(YOLO("yolo11n.pt"), classes=[2, 5, 7])

metrics = {
    "detector": "yolo11n fine-tuned (1 class)",
    "test_mAP50": float(m.box.map50), "test_mAP50_95": float(m.box.map),
    "test_mean_IoU": float(ft_iou.mean()), "test_IoU>=0.5_rate": float((ft_iou >= 0.5).mean()),
    "zero_shot_coco_mean_IoU": float(zs_iou.mean()),
    "zero_shot_coco_IoU>=0.5_rate": float((zs_iou >= 0.5).mean()),
    "model_size_MB": round(os.path.getsize(f"{OUT}/det/weights/best.pt") / 1e6, 2),
}
print(json.dumps(metrics, indent=2))
json.dump(metrics, open(f"{OUT}/det_metrics.json", "w"), indent=2)
pd.DataFrame({"fname": test_df.fname, "iou": ft_iou.round(4), "conf": ft_conf.round(4)}) \
    .sort_values("iou").to_csv(f"{OUT}/det_test_ious.csv", index=False)
print("Done. Download det/weights/best.pt, det_metrics.json, det_test_ious.csv")
