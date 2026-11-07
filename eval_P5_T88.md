1RUA24CSE0528, 1RUA24CSE0551, 1RUA24CSE0516, 1RUA24CSE0527

# Evaluation Results – P5: Intelligent Fine-Grained Vehicle Recognition

**Team:** T88 (Tech Titans) · **Dataset:** Stanford Cars (196 classes) · **Test set:** official split, 8,041 unseen images
**Train / validation split:** 7,329 / 815 (stratified 90/10 from the official 8,144 training images, seed 42; test set never used for training or model selection)

---

## 1. Classification performance (Top-1 / Top-5 accuracy)

| Approach | Pipeline | Top-1 | Top-5 |
|---|---|---|---|
| Classical only | CLAHE → HOG + LBP + HSV colour + geometric → PCA(256) → Linear SVM | 16.29% | 35.74% |
| Deep learning only | ConvNeXt-Tiny on the full image (no detection) | 91.92% | 98.53% |
| **Hybrid (final)** | YOLO11n detection → padded ROI → adaptive CLAHE → ConvNeXt-Tiny + flip TTA | **94.71%** | **99.28%** |
| Reference: classifier on ground-truth boxes | ConvNeXt-Tiny on annotated crops (upper bound for the detector) | 94.66% | 99.33% |

The hybrid pipeline with predicted boxes matches the ground-truth-box upper bound, so detection is not a bottleneck.

## 2. Object localization (IoU)

Top-confidence predicted box vs ground-truth box, 8,041 test images.

| Detector | Mean IoU | IoU ≥ 0.5 |
|---|---|---|
| YOLO11n COCO pretrained (car/bus/truck, zero-shot) | 86.81% | 91.18% |
| **YOLO11n fine-tuned on Stanford Cars** | **95.39%** | **99.81%** |

Fine-tuning improved mean IoU by +8.6 points and the IoU ≥ 0.5 rate by +8.6 points over the COCO-pretrained detector.

## 3. Object detection (mAP)

| Detector | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| YOLO11n fine-tuned (1 class: car), test set | **0.995** | **0.941** | 0.997 | 0.999 |

## 4. Model efficiency (model size and inference time)

| Component | Params | Size | Latency (batch 1) |
|---|---|---|---|
| YOLO11n detector | 2.60 M | 5.5 MB | 3.3 ms (T4 GPU) |
| ConvNeXt-Tiny – PyTorch FP32 | 27.97 M | 111.9 MB | 13.9 ms (T4 GPU) / 291.4 ms (CPU) |
| ConvNeXt-Tiny – PyTorch FP16 (reduced precision) | 27.97 M | 55.9 MB | 11.4 ms (T4 GPU) |
| ConvNeXt-Tiny – ONNX FP32 | 27.97 M | 112.0 MB | 145.1 ms (CPU) |
| ConvNeXt-Tiny – ONNX INT8 (dynamic quantization) | 27.97 M | 28.4 MB | 705.0 ms (CPU) |
| Classical pipeline (features + SVM) | – | 2.6 MB | 25.0 ms (CPU) |

**Optimization results (500 test crops):**

| Variant | Top-1 | Size reduction | Memory |
|---|---|---|---|
| PyTorch FP32 (baseline) | 96.0% | 1.0× | 1,593 MB peak GPU |
| PyTorch FP16 | 96.0% | 2.0× smaller | 889 MB peak GPU (−44%) |
| ONNX FP32 | 96.0% | 1.0× | 1,422 MB session |
| ONNX INT8 | 96.0% | **3.9× smaller** | 1,225 MB session |

- **FP16:** halves model size and GPU memory, 18% faster, no accuracy loss.
- **INT8:** 3.9× smaller with no accuracy loss, but slower on the evaluation CPU. Dynamic quantization adds quantize/dequantize overhead around ConvNeXt's linear layers, while the depthwise convolutions stay in FP32. The size gain suits storage- or bandwidth-limited deployment. For CPU latency, ONNX FP32 is the best variant (2.0× faster than PyTorch CPU).

## 5. Robustness (Top-1, 2,000 random test images under synthetic degradations)

| Condition | DL only (full image) | Hybrid without CLAHE | **Hybrid (final)** |
|---|---|---|---|
| Normal | 91.15% | 94.40% | **94.40%** |
| Low light (gamma 2.5, ×0.6) | 89.10% | 92.70% | **93.10%** |
| Gaussian blur (15×15) | 69.15% | 74.95% | **74.90%** |
| Low resolution (1/6 downscale) | 55.70% | 59.00% | **59.05%** |
| Gaussian noise (σ = 25) | 88.25% | 92.60% | **92.50%** |
| Occlusion (30% of the car box) | 82.35% | 90.00% | **89.90%** |

The hybrid pipeline outperforms DL-only under every condition, by up to +7.6 points under occlusion. Adaptive CLAHE helps in low light (+0.4) and is neutral elsewhere. Low resolution is the main weakness.

## 6. Confidence and threshold analysis (hybrid, full test set)

| Threshold | Coverage (accepted) | Accuracy on accepted | Error rate on rejected |
|---|---|---|---|
| 0.3 | 94.95% | 96.23% | 33.74% |
| **0.5 (deployed)** | **87.33%** | **97.41%** | 23.85% |
| 0.7 | 72.22% | 98.14% | 14.19% |
| 0.9 | 11.04% | 98.54% | 5.76% |

At the deployed threshold of 0.5, predictions below it are shown as "Uncertain" with the top-5 candidates instead of a single label.

---

## 7. Ablation study

**Hybrid pipeline (full test set, Top-1):**

| Step | Configuration | Top-1 | Δ |
|---|---|---|---|
| A | ConvNeXt on full image | 91.92% | – |
| B | + YOLO detection crop | 94.45% | **+2.53** |
| C | + 8% box padding | 94.70% | +0.25 |
| D | + horizontal-flip TTA | 94.71% | +0.01 |
| E | + adaptive CLAHE (final) | 94.71% | 0.00 (normal light) / +0.40 (low light) |

**Classical feature groups (test Top-1):**

| Features | Dim | Top-1 | Top-5 |
|---|---|---|---|
| Geometric only | 12 | 1.65% | 5.78% |
| LBP only | 40 | 3.02% | 9.97% |
| HSV colour only | 256 | 4.70% | 12.70% |
| HOG only | 1,764 | 13.95% | 31.51% |
| HOG + LBP + geometric | 1,816 | 14.21% | 31.84% |
| **All (HOG + LBP + colour + geometric)** | 2,072 | **16.29%** | **35.74%** |

**Detector:** fine-tuning vs COCO zero-shot: mean IoU 86.81% → 95.39%, IoU ≥ 0.5 rate 91.18% → 99.81%.

## 8. Failure analysis summary

429 of 8,041 test images were misclassified. See `docs/failure_analysis.md` for the full analysis and examples. The main failure types are visually near-identical model years, similar body styles from the same manufacturer, and difficult viewpoints, degraded images or occlusion.

## 9. Unseen internet images (live app)

14 internet images not in Stanford Cars, tested on the deployed app (FP32, laptop browser). Full table: [`eval/unseen_results.md`](eval/unseen_results.md).

| Group | Images | Result |
|---|---|---|
| Cars in the 196 classes | 6 | 6/6 correct Top-1 (5 confident, 1 rear view flagged Uncertain) |
| Near variant (F-150 SuperCrew) | 1 | Nearest class found, flagged Uncertain |
| Cars not in the dataset | 4 | 4/4 flagged Uncertain |
| Multiple cars | 2 | Only the main car boxed (failure) |
| No car (dog) | 1 | Detector false positive; classifier 3.5% → Uncertain |

Mean latency **4.80 s** per image (range 4.14–5.19 s), in the browser on a laptop CPU.

