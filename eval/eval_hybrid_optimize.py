# =============================================================================
# P5 - Hybrid pipeline evaluation + robustness + confidence analysis + optimization
# Run AFTER det_best.pt is uploaded to the HF model repo.
# Kaggle setup: GPU T4, Internet ON, same 2 datasets, (optional) HF_TOKEN secret
# Outputs (in /kaggle/working, auto-uploaded to HF if HF_TOKEN secret is attached):
#   hybrid_metrics.json         - pipeline ablation, robustness, threshold analysis
#   hybrid_test_predictions.csv - per-image hybrid predictions (failure analysis)
#   optimization_metrics.json   - FP32 vs FP16 vs ONNX INT8: size, latency, accuracy
#   cars_int8.onnx              - quantized classifier
# =============================================================================
import os, sys, json, time, copy, subprocess
os.environ["WANDB_MODE"] = "disabled"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx", "onnxruntime"], check=True)

import numpy as np
import pandas as pd
import cv2
import torch
import timm
import psutil
from PIL import Image
from scipy.io import loadmat
from torchvision import transforms as T
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

REPO = "vene-05/p5-convnext-cars"
BASE, OUT = "/kaggle/input", "/kaggle/working"
CHUNK, ROBUST_N, OPT_N, SEED = 64, 2000, 500, 42
DARK = 80          # mean L below which adaptive CLAHE is applied
device = "cuda"
rng = np.random.default_rng(SEED)

# ------------------------------- data ----------------------------------------
files, img_dirs = {}, {}
for root, dirs, fs in os.walk(BASE):
    for f in fs:
        files.setdefault(f, os.path.join(root, f))
    if sum(f.lower().endswith(".jpg") for f in fs) > 1000 and "cars_test" in root.lower():
        img_dirs["test"] = root
test_anno = next(p for f, p in files.items() if "withlabels" in f.lower() and f.lower().endswith(".mat"))
a = loadmat(test_anno, squeeze_me=True)["annotations"]
test_df = pd.DataFrame({"fname": [str(x) for x in a["fname"]],
                        "x1": a["bbox_x1"].astype(int), "y1": a["bbox_y1"].astype(int),
                        "x2": a["bbox_x2"].astype(int), "y2": a["bbox_y2"].astype(int),
                        "label": a["class"].astype(int) - 1})
print("test images:", len(test_df))

# ------------------------------ models ---------------------------------------
cfg = json.load(open(hf_hub_download(REPO, "config.json")))
names, IMG, PAD = cfg["class_names"], cfg["img_size"], cfg["pad"]
model = timm.create_model(cfg["model_name"], pretrained=False, num_classes=len(names))
model.load_state_dict(torch.load(hf_hub_download(REPO, "best_model.pth"), map_location="cpu"))
model = model.to(device).eval()
tf = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])

try:
    det, DET_CLASSES, DET_NAME = YOLO(hf_hub_download(REPO, "det_best.pt")), None, "YOLO11n fine-tuned"
except Exception:
    det, DET_CLASSES, DET_NAME = YOLO("yolo11n.pt"), [2, 5, 7], "YOLO11n COCO zero-shot"
print("detector:", DET_NAME)


# ------------------------------ helpers --------------------------------------
def clahe(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def lum(bgr):
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0].mean())


def crop(bgr, box, pad):
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = int(max(0, x1 - pad * bw)), int(max(0, y1 - pad * bh))
    x2, y2 = int(min(w, x2 + pad * bw)), int(min(h, y2 + pad * bh))
    c = bgr[y1:y2, x1:x2]
    return c if c.size else bgr


@torch.inference_mode()
def classify(bgr_list, tta):
    x = torch.stack([tf(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in bgr_list]).to(device)
    with torch.autocast("cuda", dtype=torch.float16):
        p = model(x).float().softmax(1)
        if tta:
            p = (p + model(torch.flip(x, dims=[3])).float().softmax(1)) / 2
    return p.cpu().numpy()


def detect(bgrs):
    """Ultralytics expects BGR numpy arrays. Returns top-confidence box or None per image."""
    out = []
    for r in det.predict(bgrs, imgsz=640, conf=0.25, classes=DET_CLASSES, verbose=False, device=0):
        out.append(None if len(r.boxes) == 0 else r.boxes.xyxy[int(r.boxes.conf.argmax())].cpu().numpy())
    return out


# name -> (use_detector, pad, flip_TTA, adaptive_CLAHE)
CONFIGS = {
    "A_dl_only_full_image":        (False, 0.0, False, False),
    "B_det_crop":                  (True, 0.0, False, False),
    "C_det_crop_padded":           (True, PAD, False, False),
    "D_det_crop_padded_tta":       (True, PAD, True, False),
    "E_hybrid_full_adaptiveCLAHE": (True, PAD, True, True),
}


def run_configs(bgrs, boxes, which):
    out = {}
    for name in which:
        use_det, pad, tta, adapt = CONFIGS[name]
        crops = []
        for img, b in zip(bgrs, boxes):
            c = crop(img, b, pad) if (use_det and b is not None) else img
            if adapt and lum(c) < DARK:
                c = clahe(c)
            crops.append(c)
        out[name] = classify(crops, tta)
    return out


def topk(p, y, k):
    return float((np.argsort(-p, 1)[:, :k] == y[:, None]).any(1).mean())


def load(fname):
    return cv2.imread(os.path.join(img_dirs["test"], fname))


# ===================== 1. pipeline ablation (full test set) ==================
print("\n[1/3] Pipeline ablation on the full test set...")
probs = {k: [] for k in CONFIGS}
fallback, t0 = [], time.time()
for s in range(0, len(test_df), CHUNK):
    rows = test_df.iloc[s:s + CHUNK]
    bgrs = [load(f) for f in rows.fname]
    boxes = detect(bgrs)
    fallback += [b is None for b in boxes]
    for k, v in run_configs(bgrs, boxes, CONFIGS).items():
        probs[k].append(v)
    if s % (CHUNK * 20) == 0:
        print(f"  {s}/{len(test_df)} ({time.time()-t0:.0f}s)", flush=True)
y = test_df.label.values
probs = {k: np.concatenate(v) for k, v in probs.items()}
ablation = {k: {"top1": topk(p, y, 1), "top5": topk(p, y, 5)} for k, p in probs.items()}
for k, v in ablation.items():
    print(f"  {k:32s} top1={v['top1']:.4f} top5={v['top5']:.4f}")

# confidence threshold analysis on the full hybrid
pE = probs["E_hybrid_full_adaptiveCLAHE"]
conf, pred = pE.max(1), pE.argmax(1)
correct = pred == y
thresholds = {}
for t in [0.3, 0.5, 0.7, 0.9]:
    m = conf >= t
    thresholds[str(t)] = {"coverage": float(m.mean()),
                          "accuracy_on_accepted": float(correct[m].mean()) if m.any() else None,
                          "error_rate_on_rejected_would_be": float(1 - correct[~m].mean()) if (~m).any() else None}
print("  threshold analysis:", json.dumps(thresholds, indent=1))

pd.DataFrame({"fname": test_df.fname, "true": [names[i] for i in y], "pred": [names[i] for i in pred],
              "conf": conf.round(4), "correct": correct, "detector_missed": fallback}) \
    .to_csv(f"{OUT}/hybrid_test_predictions.csv", index=False)

# ===================== 2. robustness (subset, degraded inputs) ===============
print(f"\n[2/3] Robustness on {ROBUST_N} test images...")


def degrade(bgr, cond, gt_box):
    if cond == "normal":
        return bgr
    if cond == "low_light":
        return np.clip(((bgr / 255.0) ** 2.5) * 0.6 * 255, 0, 255).astype(np.uint8)
    if cond == "blur":
        return cv2.GaussianBlur(bgr, (15, 15), 0)
    if cond == "low_resolution":
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (max(8, w // 6), max(8, h // 6)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if cond == "noise":
        return np.clip(bgr + rng.normal(0, 25, bgr.shape), 0, 255).astype(np.uint8)
    if cond == "occlusion_30pct":
        out = bgr.copy()
        x1, y1, x2, y2 = gt_box
        bw, bh = x2 - x1, y2 - y1
        ow, oh = int(bw * 0.55), int(bh * 0.55)       # ~30% of box area
        ox = int(x1 + rng.uniform(0, max(1, bw - ow))); oy = int(y1 + rng.uniform(0, max(1, bh - oh)))
        out[oy:oy + oh, ox:ox + ow] = 127
        return out
    raise ValueError(cond)


sub = test_df.iloc[rng.choice(len(test_df), ROBUST_N, replace=False)]
ys = sub.label.values
which = ["A_dl_only_full_image", "D_det_crop_padded_tta", "E_hybrid_full_adaptiveCLAHE"]
robust = {}
for cond in ["normal", "low_light", "blur", "low_resolution", "noise", "occlusion_30pct"]:
    acc = {k: [] for k in which}
    for s in range(0, len(sub), CHUNK):
        rows = sub.iloc[s:s + CHUNK]
        bgrs = [degrade(load(r.fname), cond, (r.x1, r.y1, r.x2, r.y2)) for r in rows.itertuples()]
        for k, v in run_configs(bgrs, detect(bgrs), which).items():
            acc[k].append(v)
    robust[cond] = {k: topk(np.concatenate(v), ys, 1) for k, v in acc.items()}
    print(f"  {cond:16s}", {k[:1]: round(v, 4) for k, v in robust[cond].items()}, flush=True)

json.dump({"detector": DET_NAME, "n_test": len(test_df), "detector_miss_rate": float(np.mean(fallback)),
           "pipeline_ablation": ablation, "confidence_thresholds": thresholds,
           "robustness_top1": robust, "robustness_n": ROBUST_N},
          open(f"{OUT}/hybrid_metrics.json", "w"), indent=2)

# ===================== 3. model optimization =================================
print("\n[3/3] Optimization: FP32 vs FP16 vs ONNX INT8...")
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

opt_rows = test_df.iloc[rng.choice(len(test_df), OPT_N, replace=False)]
X = torch.stack([tf(Image.fromarray(cv2.cvtColor(crop(load(r.fname), (r.x1, r.y1, r.x2, r.y2), PAD),
                                                  cv2.COLOR_BGR2RGB))) for r in opt_rows.itertuples()])
yo = opt_rows.label.values
n_params = sum(p.numel() for p in model.parameters())


def acc_of(logits):
    return float((np.asarray(logits).argmax(1) == yo).mean())


def time_it(fn, n=20, warm=3):
    for _ in range(warm):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000


results = {"params_millions": round(n_params / 1e6, 2), "n_eval_images": OPT_N, "input_size": IMG}
x1 = X[:1]

# PyTorch FP32 / FP16 on GPU
for prec in ["fp32", "fp16"]:
    m = copy.deepcopy(model).to(device).eval()
    if prec == "fp16":
        m = m.half()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        logits = torch.cat([m(X[i:i + 32].to(device).to(next(m.parameters()).dtype)).float().cpu()
                            for i in range(0, OPT_N, 32)])
        xb = x1.to(device).to(next(m.parameters()).dtype)
        lat = time_it(lambda: (m(xb), torch.cuda.synchronize()))
    results[f"torch_{prec}_gpu"] = {"top1": acc_of(logits), "latency_ms_batch1": round(lat, 2),
                                    "size_MB": round(n_params * (4 if prec == "fp32" else 2) / 1e6, 1),
                                    "peak_gpu_mem_MB": round(torch.cuda.max_memory_allocated() / 1e6, 1)}
    del m

# ONNX export (CPU) + dynamic INT8 quantization
cpu_model = copy.deepcopy(model).cpu().float().eval()
fp32_path, int8_path = f"{OUT}/cars_fp32.onnx", f"{OUT}/cars_int8.onnx"
export_kw = dict(input_names=["input"], output_names=["logits"], opset_version=17,
                 dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
try:
    torch.onnx.export(cpu_model, x1, fp32_path, dynamo=False, **export_kw)
except TypeError:
    torch.onnx.export(cpu_model, x1, fp32_path, **export_kw)
quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)

with torch.inference_mode():
    results["torch_fp32_cpu"] = {"latency_ms_batch1": round(time_it(lambda: cpu_model(x1), n=10), 1)}

so = ort.SessionOptions()
so.intra_op_num_threads = os.cpu_count()
proc = psutil.Process()
for tag, path in [("onnx_fp32_cpu", fp32_path), ("onnx_int8_cpu", int8_path)]:
    rss0 = proc.memory_info().rss
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    logits = np.concatenate([sess.run(None, {"input": X[i:i + 16].numpy()})[0] for i in range(0, OPT_N, 16)])
    lat = time_it(lambda: sess.run(None, {"input": x1.numpy()}), n=10)
    results[tag] = {"top1": acc_of(logits), "latency_ms_batch1": round(lat, 1),
                    "size_MB": round(os.path.getsize(path) / 1e6, 1),
                    "session_memory_MB": round((proc.memory_info().rss - rss0) / 1e6, 1)}
    del sess
print(json.dumps(results, indent=2))
json.dump(results, open(f"{OUT}/optimization_metrics.json", "w"), indent=2)

# ------------------------- upload to HF (optional) ---------------------------
try:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import HfApi
    api = HfApi(token=UserSecretsClient().get_secret("HF_TOKEN"))
    for f in ["hybrid_metrics.json", "hybrid_test_predictions.csv", "optimization_metrics.json", "cars_int8.onnx"]:
        api.upload_file(path_or_fileobj=f"{OUT}/{f}", path_in_repo=f, repo_id=REPO)
        print("uploaded", f)
except Exception as e:
    print("HF upload skipped (attach HF_TOKEN secret to enable):", e)
print("Done.")
