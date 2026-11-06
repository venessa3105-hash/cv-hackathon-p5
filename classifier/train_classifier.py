# =============================================================================
# P5 - Stanford Cars fine-grained classifier (ConvNeXt-Tiny, 196 classes)
# Kaggle setup: Accelerator = GPU T4 x2, Internet = ON
# Outputs (in /kaggle/working):
#   best_model.pth      - best weights (by validation accuracy)
#   config.json         - model name, image size, crop padding, class names
#   test_predictions.csv- per-image predictions (use for failure analysis)
#   metrics.json        - Top-1 / Top-5 on the official test set
# =============================================================================
import os, json, time, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.io import loadmat
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
import timm

# ----------------------------- config ----------------------------------------
BASE = "/kaggle/input"
OUT = "/kaggle/working"
MODEL_NAME = "convnext_tiny.fb_in22k_ft_in1k"
IMG = 384
BATCH = 32
EPOCHS = 20
LR = 2e-4
WD = 0.05
PAD = 0.08          # extra margin around the bounding box when cropping
VAL_FRAC = 0.10     # held-out validation split from the training set
SEED = 42
NUM_WORKERS = 4
DRY_RUN = False     # True = only check paths/annotations, no training
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


seed_all(SEED)

# --------------------- auto-discover dataset files ---------------------------
files, img_dirs = {}, {}
for root, dirs, fs in os.walk(BASE):
    for f in fs:
        files.setdefault(f, os.path.join(root, f))
    n_jpg = sum(f.lower().endswith(".jpg") for f in fs)
    if n_jpg > 1000:
        r = root.lower()
        if "cars_train" in r:
            img_dirs["train"] = root
        elif "cars_test" in r:
            img_dirs["test"] = root

meta_path = files.get("cars_meta.mat")
train_anno_path = files.get("cars_train_annos.mat")
test_anno_path = next((p for f, p in files.items()
                       if "withlabels" in f.lower() and f.lower().endswith(".mat")), None)

print("train images :", img_dirs.get("train"))
print("test images  :", img_dirs.get("test"))
print("meta         :", meta_path)
print("train annos  :", train_anno_path)
print("test annos   :", test_anno_path)
assert all([img_dirs.get("train"), img_dirs.get("test"), meta_path, train_anno_path, test_anno_path]), \
    "Missing files - paste the printout above to Claude."

class_names = [str(c) for c in loadmat(meta_path, squeeze_me=True)["class_names"]]


def load_annos(path):
    a = loadmat(path, squeeze_me=True)["annotations"]
    df = pd.DataFrame({
        "fname": [str(x) for x in a["fname"]],
        "x1": a["bbox_x1"].astype(int), "y1": a["bbox_y1"].astype(int),
        "x2": a["bbox_x2"].astype(int), "y2": a["bbox_y2"].astype(int),
        "label": a["class"].astype(int) - 1,   # 1-indexed in .mat -> 0-indexed
    })
    return df


train_full = load_annos(train_anno_path)
test_df = load_annos(test_anno_path)
train_df, val_df = train_test_split(train_full, test_size=VAL_FRAC,
                                    stratify=train_full["label"], random_state=SEED)
print(f"classes={len(class_names)} train={len(train_df)} val={len(val_df)} test={len(test_df)}")
print("example class:", class_names[train_full.label.iloc[0]])
assert os.path.exists(os.path.join(img_dirs["train"], train_df.fname.iloc[0]))
assert os.path.exists(os.path.join(img_dirs["test"], test_df.fname.iloc[0]))

# save splits so the classical baseline uses the SAME split
train_df.to_csv(f"{OUT}/split_train.csv", index=False)
val_df.to_csv(f"{OUT}/split_val.csv", index=False)
test_df.to_csv(f"{OUT}/split_test.csv", index=False)

if DRY_RUN:
    raise SystemExit("Dry run OK - set DRY_RUN = False to train.")

# ------------------------------ data -----------------------------------------
def crop_box(im, x1, y1, x2, y2, pad=PAD):
    w, h = im.size
    bw, bh = x2 - x1, y2 - y1
    return im.crop((max(0, x1 - pad * bw), max(0, y1 - pad * bh),
                    min(w, x2 + pad * bw), min(h, y2 + pad * bh)))


class CarDS(Dataset):
    def __init__(self, df, img_dir, tf):
        self.rows, self.img_dir, self.tf = df.to_dict("records"), img_dir, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        im = Image.open(os.path.join(self.img_dir, r["fname"])).convert("RGB")
        im = crop_box(im, r["x1"], r["y1"], r["x2"], r["y2"])
        return self.tf(im), r["label"]


train_tf = T.Compose([
    T.Resize((IMG, IMG)),
    T.RandomHorizontalFlip(),
    T.TrivialAugmentWide(),                 # illumination / colour robustness
    T.ToTensor(), T.Normalize(MEAN, STD),
    T.RandomErasing(p=0.25),                # simulates partial occlusion
])
eval_tf = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(), T.Normalize(MEAN, STD)])

dl_kw = dict(num_workers=NUM_WORKERS, pin_memory=True)
train_dl = DataLoader(CarDS(train_df, img_dirs["train"], train_tf), BATCH, shuffle=True, drop_last=True, **dl_kw)
val_dl = DataLoader(CarDS(val_df, img_dirs["train"], eval_tf), BATCH * 2, **dl_kw)
test_dl = DataLoader(CarDS(test_df, img_dirs["test"], eval_tf), BATCH * 2, **dl_kw)

# ------------------------------ model ----------------------------------------
device = "cuda"
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(class_names))
model = model.to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print("Using", torch.cuda.device_count(), "GPUs")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
total_steps = EPOCHS * len(train_dl)
warmup = len(train_dl)  # 1 epoch warmup


def lr_lambda(step):
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler("cuda")


@torch.no_grad()
def predict(dl, tta=False):
    model.eval()
    all_logits, all_labels = [], []
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x).float()
            if tta:
                logits = (logits + model(torch.flip(x, dims=[3])).float()) / 2
        all_logits.append(logits.cpu()); all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


def topk_acc(logits, labels, k):
    return (logits.topk(k, dim=1).indices == labels[:, None]).any(1).float().mean().item()


def raw_model():
    return model.module if isinstance(model, nn.DataParallel) else model


# ------------------------------ train ----------------------------------------
best_acc, history = 0.0, []
for epoch in range(EPOCHS):
    model.train(); t0 = time.time(); run_loss = 0.0
    for x, y in train_dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update(); scheduler.step()
        run_loss += loss.item()
    vl, vy = predict(val_dl)
    v1, v5 = topk_acc(vl, vy, 1), topk_acc(vl, vy, 5)
    history.append({"epoch": epoch + 1, "loss": run_loss / len(train_dl), "val_top1": v1, "val_top5": v5})
    print(f"epoch {epoch+1}/{EPOCHS} loss={run_loss/len(train_dl):.3f} val_top1={v1:.4f} "
          f"val_top5={v5:.4f} ({time.time()-t0:.0f}s)", flush=True)
    torch.save(raw_model().state_dict(), f"{OUT}/last_model.pth")
    if v1 > best_acc:
        best_acc = v1
        torch.save(raw_model().state_dict(), f"{OUT}/best_model.pth")

pd.DataFrame(history).to_csv(f"{OUT}/train_history.csv", index=False)

# ------------------------- test evaluation -----------------------------------
raw_model().load_state_dict(torch.load(f"{OUT}/best_model.pth", map_location=device))
tl, ty = predict(test_dl, tta=True)
probs = tl.softmax(1)
metrics = {
    "model": MODEL_NAME, "img_size": IMG,
    "val_top1_best": best_acc,
    "test_top1": topk_acc(tl, ty, 1), "test_top5": topk_acc(tl, ty, 5),
    "n_test": len(ty),
}
print(json.dumps(metrics, indent=2))
json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=2)

conf, pred = probs.max(1)
top5 = probs.topk(5, 1).indices
pd.DataFrame({
    "fname": test_df.fname.values,
    "true": [class_names[i] for i in ty.tolist()],
    "pred": [class_names[i] for i in pred.tolist()],
    "conf": conf.numpy().round(4),
    "correct": (pred == ty).numpy(),
    "top5": ["|".join(class_names[j] for j in row) for row in top5.tolist()],
}).to_csv(f"{OUT}/test_predictions.csv", index=False)

json.dump({"model_name": MODEL_NAME, "img_size": IMG, "pad": PAD, "mean": MEAN, "std": STD,
           "class_names": class_names}, open(f"{OUT}/config.json", "w"), indent=2)
print("Done. Download best_model.pth, config.json, metrics.json, test_predictions.csv")
