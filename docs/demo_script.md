# Demo Script – P5 Fine-Grained Vehicle Recognition (Team T07)

**Live app:** https://huggingface.co/spaces/vene-05/p5-vehicle-recognition
**Length:** 3 minutes · **Device:** laptop (FP32 model), with a phone ready to show mobile support

## Before the demo (5 minutes earlier)

- Open the live app on the laptop and wait for **Ready**, so the models are already cached.
- Keep these files open in a folder: `eval/unseen_images/01_normal_bmw_m3.jpg`, `06_night_camry.jpg`, `13_new_tesla_y.jpg`, `11_multi_parking.jpg`.
- Have the GitHub repo and `eval_P5_T07.md` open in a second tab.
- Backup: if the internet fails, run the Gradio app locally (`cd app && python app.py`) or show the screenshots in `eval/unseen_images/screenshots/`.

## 3-minute plan

| Time | Show | Say |
|---|---|---|
| 0:00–0:20 | The live app's front page | "Our system detects cars and identifies make, model and year across the 196 Stanford Cars classes. It is a hybrid of classical computer vision and deep learning, and it runs entirely in the browser." |
| 0:20–1:00 | Upload `01_normal_bmw_m3.jpg` → Recognize | "YOLO11n finds the car. The padded crop goes to our fine-tuned ConvNeXt-Tiny, which predicts BMW M3 Coupe 2012. Here are the confidence and the top-5 candidates." |
| 1:00–1:30 | Scroll to the comparison table, latency and intermediate steps | "Classical only gets 16% on the test set, deep learning only 92%, and our hybrid 94.7%. These are the classical steps: CLAHE, Canny edges, contours with the ROI, and gradients. The whole pipeline takes about 5 seconds on a laptop." |
| 1:30–1:50 | Upload `06_night_camry.jpg` | "In low light the crop is dark, so adaptive CLAHE switches on (see the tag). In our robustness test it adds +0.4 points in low light." |
| 1:50–2:15 | Upload `13_new_tesla_y.jpg` | "This 2023 Tesla Model Y isn't one of the 196 classes. Instead of giving a confident wrong answer, the model's confidence is low, so it's flagged as **Uncertain** with the top-5 shown. At our 0.5 threshold, accepted predictions are 97.4% accurate." |
| 2:15–2:40 | **The judge's own image** | "Would you like to try your own image?" |
| 2:40–3:00 | The Evaluation tab | "Detection mAP@0.5 is 0.995 and IoU is 95.4%. The hybrid beats deep learning only under every condition, including +7.6 points under occlusion. INT8 quantization makes the classifier 3.9× smaller with no accuracy loss." |

## Who answers what in the viva

| Member (USN) | Own part | Be ready to explain |
|---|---|---|
| 1RUA24CSE0528 | Classifier | Why ConvNeXt-Tiny; transfer learning from ImageNet-22k; 384 px input; augmentation (TrivialAugment, RandomErasing); label smoothing; 94.66% Top-1 |
| 1RUA24CSE0551 | Detector, hybrid evaluation, optimization | YOLO11n fine-tuning; mAP vs IoU; ablation A→E (+2.53 from the crop); robustness table; why INT8 is smaller but slower on CPU; FP16 |
| 1RUA24CSE0516 | Failure analysis, diagrams, unseen testing | 3 failure types and their causes; the architecture diagram; unseen-image results; why confidence is moderate (label smoothing) |
| 1RUA24CSE0527 | Deployment and repository | Static HF Space; ONNX Runtime Web and OpenCV.js; privacy (the image never leaves the device); FP32 vs INT8 in the app; Docker / Gradio local run |

*(Each member: check your row and correct it if your part was different.)*

## Likely questions and short answers

- **Why is the confidence only around 50% when the prediction is correct?** We trained with label smoothing (0.1), which keeps confidence moderate and reduces overconfidence. The 0.5 threshold was chosen from the test-set analysis of this exact model.
- **What does the classical part contribute to the hybrid?** Adaptive CLAHE for dark crops, and the edge/contour ROI as a fallback when YOLO finds nothing. On its own, the classical pipeline reaches 16.3%, the baseline we compare against.
- **Why is the classical accuracy so low?** Hand-crafted features (HOG, LBP, colour, shape) cannot separate 196 near-identical classes. Deep features learn the fine details.
- **What does the ablation show?** The YOLO crop is the biggest gain (+2.53). Padding adds +0.25, TTA almost nothing, and CLAHE helps only in low light.
- **Biggest weakness?** Low resolution (59% Top-1) and model years with near-identical styling. Cars outside the 196 classes cannot be named, only flagged.
- **Why INT8 is slower:** dynamic quantization adds quantize/dequantize overhead, and ConvNeXt's depthwise convolutions stay in FP32. Its benefit is size (112 → 28 MB), which matters for phone downloads.
- **Ethics:** no plate reading or tracking; images are processed on-device; uncertainty is shown honestly; there is dataset bias toward US cars up to 2012.