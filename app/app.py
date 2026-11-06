"""P5 - Intelligent Fine-Grained Vehicle Recognition (Gradio app for HF Spaces).

Pipelines compared on every upload:
  * Hybrid  : YOLO detection -> padded ROI crop -> adaptive CLAHE (dark images)
              -> ConvNeXt-Tiny (flip TTA) -> low-confidence gating;
              falls back to classical edge/contour ROI when nothing is detected.
  * Deep learning only : ConvNeXt on the whole image (no detection, no preprocessing).
  * Classical only     : edge/contour ROI -> CLAHE -> HOG+LBP+HSV+geometric -> SVM.
"""
import os
import json
import time

import cv2
import numpy as np
import torch
import timm
import gradio as gr
from PIL import Image
from torchvision import transforms as T
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from features import (apply_clahe, crop_box, extract_all, classical_roi, hog_visualization)

REPO = os.environ.get("MODEL_REPO", "vene-05/p5-convnext-cars")
CONF_THRESHOLD = 0.50   # below this the prediction is flagged "uncertain"
DET_CONF = 0.25
DARK_THRESHOLD = 80     # mean L (0-255) below which CLAHE is applied in the hybrid path
MAX_CARS = 5
MAX_SIDE = 1280
torch.set_num_threads(max(1, os.cpu_count() or 2))


def try_download(name):
    try:
        return hf_hub_download(REPO, name)
    except Exception:
        return None


# ------------------------------------------------------------------ load models
cfg = json.load(open(hf_hub_download(REPO, "config.json")))
CLASS_NAMES = cfg["class_names"]
clf = timm.create_model(cfg["model_name"], pretrained=False, num_classes=len(CLASS_NAMES))
clf.load_state_dict(torch.load(hf_hub_download(REPO, "best_model.pth"), map_location="cpu"))
clf.eval()
tf = T.Compose([T.Resize((cfg["img_size"], cfg["img_size"])), T.ToTensor(),
                T.Normalize(cfg["mean"], cfg["std"])])

_det_path = try_download("det_best.pt")
if _det_path:
    detector, DET_CLASSES, DET_NAME = YOLO(_det_path), None, "YOLO11n fine-tuned on Stanford Cars"
else:
    detector, DET_CLASSES, DET_NAME = YOLO("yolo11n.pt"), [2, 5, 7], "YOLO11n COCO (car/bus/truck)"

classical = None
_cl_path = try_download("classical_model.joblib")
if _cl_path:
    try:
        import joblib
        classical = joblib.load(_cl_path)
    except Exception as e:  # sklearn version mismatch etc.
        print("Classical model not loaded:", e)


def load_metrics():
    out = {}
    for name in ["metrics.json", "classical_metrics.json", "det_metrics.json", "optimization_metrics.json"]:
        p = try_download(name)
        if p:
            out[name] = json.load(open(p))
    return out


METRICS = load_metrics()

# ---------------------------------------------------------------------- helpers
MULTI_WORD_MAKES = ("AM General", "Aston Martin", "Land Rover")


def parse_name(name):
    parts = name.split()
    year = parts[-1] if parts[-1].isdigit() else "?"
    rest = parts[:-1] if year != "?" else parts
    for mk in MULTI_WORD_MAKES:
        if name.startswith(mk):
            return mk, " ".join(rest[len(mk.split()):]), year
    return rest[0], " ".join(rest[1:]), year


@torch.inference_mode()
def classify(crops_rgb):
    """crops_rgb: list of HxWx3 uint8 RGB arrays -> (N, 196) probabilities (flip TTA)."""
    x = torch.stack([tf(Image.fromarray(c)) for c in crops_rgb])
    p = (clf(x).softmax(1) + clf(torch.flip(x, dims=[3])).softmax(1)) / 2
    return p.numpy()


def mean_luminance(bgr):
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0].mean())


def draw_label(img, box, text, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    ty = max(y1, th + 10)
    cv2.rectangle(img, (x1, ty - th - 10), (x1 + tw + 8, ty), color, -1)
    cv2.putText(img, text, (x1 + 4, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def resize_max(img, max_side=MAX_SIDE):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


# --------------------------------------------------------------------- pipeline
def run(image, adaptive_enh, conf_thr):
    if image is None:
        raise gr.Error("Please upload an image of a car first.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    rgb = resize_max(image[..., :3].copy())
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    H, W = rgb.shape[:2]
    timings = {}

    # ---------------- Hybrid ----------------
    t0 = time.perf_counter()
    det = detector.predict(bgr, conf=DET_CONF, classes=DET_CLASSES, verbose=False, imgsz=640)[0]
    boxes = det.boxes.xyxy.cpu().numpy() if len(det.boxes) else np.zeros((0, 4))
    dconfs = det.boxes.conf.cpu().numpy() if len(det.boxes) else np.zeros(0)
    order = np.argsort(-dconfs)[:MAX_CARS]
    boxes, dconfs = boxes[order], dconfs[order]
    used_fallback = False
    if len(boxes) == 0:  # classical ROI fallback
        roi_box, _, _ = classical_roi(bgr)
        boxes, dconfs, used_fallback = np.array([roi_box], dtype=float), np.array([0.0]), True

    crops, enhanced_flags = [], []
    for b in boxes:
        c = crop_box(bgr, b)
        dark = mean_luminance(c) < DARK_THRESHOLD
        if adaptive_enh and dark:
            c = apply_clahe(c)
        enhanced_flags.append(adaptive_enh and dark)
        crops.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
    probs = classify(crops)
    timings["Hybrid"] = (time.perf_counter() - t0) * 1000

    annotated = rgb.copy()
    rows = []
    for i, (b, p) in enumerate(zip(boxes, probs)):
        k = int(p.argmax())
        conf = float(p[k])
        make, model, year = parse_name(CLASS_NAMES[k])
        ok = conf >= conf_thr
        label = f"{make} {model} {year} {conf:.0%}" if ok else f"Uncertain ({conf:.0%})"
        draw_label(annotated, b, label, (0, 200, 0) if ok else (255, 165, 0))
        rows.append((i + 1, make, model, year, conf, float(dconfs[i]), ok, enhanced_flags[i]))

    top5 = {CLASS_NAMES[j]: float(probs[0][j]) for j in np.argsort(-probs[0])[:5]}

    # ---------------- DL only ----------------
    t0 = time.perf_counter()
    p_dl = classify([rgb])[0]
    timings["Deep learning only"] = (time.perf_counter() - t0) * 1000
    k_dl = int(p_dl.argmax())

    # ---------------- Classical only ----------------
    roi_box, edges, contour_vis = classical_roi(bgr)
    if classical is not None:
        t0 = time.perf_counter()
        f = extract_all(bgr, roi_box)
        scores = classical["model"].decision_function(np.hstack([f[g] for g in classical["groups"]])[None])[0]
        timings["Classical only"] = (time.perf_counter() - t0) * 1000
        e = np.exp(scores - scores.max())
        cl_pred, cl_conf = CLASS_NAMES[int(scores.argmax())], float((e / e.sum()).max())
    else:
        cl_pred, cl_conf = "model not uploaded yet", None

    # ---------------- text outputs ----------------
    md = [f"**Detector:** {DET_NAME}" + (" — *no vehicle detected, used classical edge/contour ROI*" if used_fallback else ""),
          "", "| # | Make | Model | Year | Confidence | Det. conf | Status |", "|---|---|---|---|---|---|---|"]
    for i, make, model, year, conf, dc, ok, enh in rows:
        status = ("✅ confident" if ok else "⚠️ low confidence — see top-5") + (" · CLAHE applied" if enh else "")
        md.append(f"| {i} | {make} | {model} | {year} | {conf:.1%} | {dc:.2f} | {status} |")
    md.append(f"\nThreshold: {conf_thr:.0%}. Predictions below it are shown as *Uncertain*.")

    cmp = ["| Approach | Prediction (main car) | Confidence | Latency (CPU) |", "|---|---|---|---|",
           f"| Classical only | {cl_pred} | {f'{cl_conf:.1%}' if cl_conf is not None else '–'} | "
           f"{timings.get('Classical only', 0):.0f} ms |",
           f"| Deep learning only | {CLASS_NAMES[k_dl]} | {p_dl[k_dl]:.1%} | {timings['Deep learning only']:.0f} ms |",
           f"| **Hybrid** | **{CLASS_NAMES[int(probs[0].argmax())]}** | **{probs[0].max():.1%}** | "
           f"**{timings['Hybrid']:.0f} ms** |"]

    # ---------------- intermediate visualizations ----------------
    main_crop_bgr = crop_box(bgr, boxes[0])
    gallery = [
        (cv2.cvtColor(apply_clahe(bgr), cv2.COLOR_BGR2RGB), "1. CLAHE (contrast-limited histogram equalization)"),
        (edges, "2. Canny edges"),
        (cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB), "3. Contours + classical ROI"),
        (crops[0], "4. Detector ROI crop fed to the CNN"),
        (hog_visualization(main_crop_bgr), "5. HOG gradient features"),
    ]
    return annotated, "\n".join(md), top5, "\n".join(cmp), gallery


# ------------------------------------------------------------------ metrics tab
def metrics_markdown():
    m = METRICS
    lines = ["### Test-set results (Stanford Cars, 8,041 unseen test images)", "",
             "| Approach | Top-1 | Top-5 |", "|---|---|---|"]
    cl = m.get("classical_metrics.json", {}).get("ablation", {}).get("all")
    lines.append(f"| Classical (HOG+LBP+colour+geometric → SVM) | {cl['test_top1']:.2%} | {cl['test_top5']:.2%} |"
                 if cl else "| Classical | pending | pending |")
    dl = m.get("metrics.json")
    if dl:
        lines.append(f"| Deep learning ({dl['model']}, GT crops) | {dl['test_top1']:.2%} | {dl['test_top5']:.2%} |")
    det = m.get("det_metrics.json")
    if det:
        lines += ["", "### Detection", "", "| Metric | Value |", "|---|---|",
                  f"| mAP@0.5 | {det['test_mAP50']:.3f} |", f"| mAP@0.5:0.95 | {det['test_mAP50_95']:.3f} |",
                  f"| Mean IoU (fine-tuned) | {det['test_mean_IoU']:.3f} |",
                  f"| Mean IoU (COCO zero-shot) | {det['zero_shot_coco_mean_IoU']:.3f} |"]
    if cl:
        lines += ["", "### Classical feature ablation", "", "| Features | Test Top-1 |", "|---|---|"]
        for k, v in m["classical_metrics.json"]["ablation"].items():
            lines.append(f"| {k} | {v['test_top1']:.2%} |")
    opt = m.get("optimization_metrics.json")
    if opt:
        lines += ["", "### Model optimization", "", "```json", json.dumps(opt, indent=2), "```"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- UI
with gr.Blocks(title="Fine-Grained Vehicle Recognition", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚗 Fine-Grained Vehicle Recognition\nUpload a car photo. The system detects each "
                "vehicle and predicts its **make, model and year** (196 Stanford Cars classes), comparing "
                "classical, deep-learning and hybrid pipelines.")
    with gr.Tab("Recognize"):
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="numpy", label="Upload a car image")
                enh = gr.Checkbox(value=True, label="Adaptive low-light enhancement (CLAHE when dark)")
                thr = gr.Slider(0.1, 0.9, value=CONF_THRESHOLD, step=0.05, label="Confidence threshold")
                btn = gr.Button("Recognize", variant="primary")
            with gr.Column(scale=1):
                out_img = gr.Image(label="Detections")
                out_top5 = gr.Label(num_top_classes=5, label="Top-5 (main car)")
        out_md = gr.Markdown()
        gr.Markdown("### Classical vs deep learning vs hybrid (this image)")
        out_cmp = gr.Markdown()
        out_gallery = gr.Gallery(label="Intermediate image-processing steps", columns=5, height="auto")
        btn.click(run, [inp, enh, thr], [out_img, out_md, out_top5, out_cmp, out_gallery])
    with gr.Tab("Evaluation"):
        gr.Markdown(metrics_markdown())
    with gr.Tab("About"):
        gr.Markdown(
            "**Limitations.** Trained on Stanford Cars (196 classes, mostly US-market cars up to 2012). "
            "Cars outside these classes will be mapped to the closest known class. That is why low-confidence "
            "predictions are flagged instead of shown as fact. Model years with near-identical styling are "
            "the hardest cases. No number-plate recognition or tracking is performed.")

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
