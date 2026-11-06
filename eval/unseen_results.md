# Results on unseen internet images (live app)

**App:** https://huggingface.co/spaces/vene-05/p5-vehicle-recognition · **Threshold:** 0.5 · **Classifier:** FP32 ONNX · **Device:** Lenovo laptop, Chrome (CPU, in-browser WASM)

None of these images are from Stanford Cars. Sources are in [`unseen_images/sources.md`](unseen_images/sources.md).

| # | Image | True class / expected | Hybrid prediction | Conf. | Det. conf | DL-only prediction | Conf. | Result | Total time | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 01 | 01_normal_bmw_m3 | BMW M3 Coupe 2012 | BMW M3 Coupe 2012 | 87.2% | 0.95 | BMW M3 Coupe 2012 | 89.3% | ✅ Correct | 5.19 s | Clean shot; car fills the frame, so the crop adds little |
| 03 | 03_normal_ford_f150_supercrew | F-150 SuperCrew (nearest class: F-150 Regular Cab 2012) | Ford F-150 Regular Cab 2012 → Uncertain | 47.8% | 0.95 | Ford F-150 Regular Cab 2012 | 48.9% | ✅ Handled | 4.59 s | Nearest class found; cab variant not in the dataset, so confidence is just below threshold |
| 04 | 04_normal_tesla_s | Tesla Model S Sedan 2012 | Tesla Model S Sedan 2012 | 90.2% | 0.95 | Tesla Model S Sedan 2012 | 91.0% | ✅ Correct | 4.93 s | |
| 05 | 05_night_camaro | Chevrolet Camaro Convertible 2012 | Chevrolet Camaro Convertible 2012 | 85.1% | 0.94 | Chevrolet Camaro Convertible 2012 | 87.5% | ✅ Correct | 4.59 s | Dark scene: crop L = 75 → **adaptive CLAHE applied** |
| 06 | 06_night_camry | Toyota Camry Sedan 2012 | Toyota Camry Sedan 2012 | 62.8% | 0.93 | Toyota Camry Sedan 2012 | 49.1% | ✅ Correct | 5.07 s | Night street. **Hybrid confident, DL-only would have been Uncertain** |
| 07 | 07_rear_jeep | Jeep Wrangler SUV 2012 | Jeep Wrangler SUV 2012 → Uncertain | 38.7% | 0.94 | Jeep Wrangler SUV 2012 | 38.7% | ⚠️ Correct Top-1, but Uncertain | 4.91 s | Rear view: training images are mostly front/side views |
| 08 | 08_side_mustang | Ford Mustang Convertible 2007 | Ford Mustang Convertible 2007 | 68.0% | 0.92 | Ford Mustang Convertible 2007 | 81.3% | ✅ Correct | 4.81 s | Pure side profile; the very wide crop is squashed to a 384×384 square |
| 09 | 09_occluded_tree | 1990s Honda Civic (not in dataset) → Uncertain | Eagle Talon Hatchback 1998 → Uncertain | 44.7% | 0.76 | Eagle Talon Hatchback 1998 | 31.4% | ✅ Flagged | 4.88 s | Car still detected through branches (lower det. conf); nearest guess is a similar 1990s hatchback |
| 10 | 10_occluded_damaged | Crashed Renault Mégane (not in dataset) → Uncertain | Plymouth Neon Coupe 1999 → Uncertain | 15.1% | 0.72 | Acura Integra Type R 2001 | 11.8% | ✅ Flagged | 4.88 s | ⚠️ Box misses the crushed front end past the tree |
| 11 | 11_multi_parking | Several cars → several boxes | 1 box: Audi S5 Convertible 2012 → Uncertain | 19.1% | 0.92 | Audi S5 Convertible 2012 | 9.8% | ❌ Only 1 of ~5 cars boxed | 4.87 s | Make/model right but newer generation (2017+) than the 2012 class; CLAHE applied (L = 77) |
| 12 | 12_multi_street | Many vehicles → several boxes | 1 box: AM General Hummer SUV 2000 → Uncertain | 26.8% | 0.69 | McLaren MP4-12C Coupe 2012 | 6.7% | ❌ Only 1 vehicle boxed | 4.51 s | Pickup type not in dataset. DL-only on the cluttered scene gives a meaningless answer |
| 13 | 13_new_tesla_y | Tesla Model Y 2023 (not in dataset) → Uncertain | Porsche Panamera Sedan 2012 → Uncertain | 10.8% | 0.96 | Porsche Panamera Sedan 2012 | 11.2% | ✅ Flagged | 4.14 s | #2 candidate is Tesla Model S (closest relative in the dataset) |
| 14 | 14_new_hyundai_ioniq5 | Hyundai Ioniq 5 (not in dataset) → Uncertain | *not run* | | | | | – | | |
| 15 | 15_not_car | Dog, no vehicle → no confident label | Bentley Continental Supersports Conv. 2012 → Uncertain | 3.5% | 0.74 | same | 3.5% | ⚠️ Classifier handled; **detector false positive** | 4.94 s | Detector boxed the dog at 0.74; confidence gating prevented a wrong label |
| 16 | 16_ood_byd_qin | BYD Qin (Chinese market, not in dataset) → Uncertain | Hyundai Veloster Hatchback 2012 → Uncertain | 11.3% | 0.95 | Hyundai Veloster Hatchback 2012 | 9.0% | ✅ Flagged | 4.91 s | Shaded car: CLAHE applied (L = 71); neighbouring white car not boxed |

## Summary (14 images tested)

| Group | Images | Outcome |
|---|---|---|
| Cars in the 196 classes | 01, 04, 05, 06, 07, 08 | **6/6 correct Top-1**; 5/6 confident, 1 (rear view) correctly guessed but flagged Uncertain |
| Near variant | 03 | Nearest class found (F-150), flagged Uncertain at 47.8% |
| Cars not in the dataset | 09, 10, 13, 16 | **4/4 flagged Uncertain**; no confident wrong label |
| Multiple cars | 11, 12 | ❌ Only the main car was boxed in each image |
| No car | 15 | Detector false positive, but the classifier was not confident (3.5%) → Uncertain |

- **Low-confidence handling worked on every out-of-dataset input:** none of the 6 inputs outside the 196 classes (03, 09, 10, 13, 15, 16) received a confident label.
- **Hybrid vs DL-only:** the hybrid was more confident in 6 cases (06, 09, 10, 11, 12, 16). In **06 it turned an Uncertain DL-only result into a confident correct one**, and in **12 the DL-only answer on the full scene was meaningless (McLaren)**. DL-only was slightly more confident in 5 cases where the car already fills the frame (01, 03, 04, 05, 08).
- **Adaptive CLAHE** triggered automatically on the 4 dark crops (05, 11, 15, 16) and not on bright ones.
- **Latency:** 4.14–5.19 s per image, **mean 4.80 s**, on a laptop CPU in the browser, including the DL-only comparison run. That is within the ~10 s target.

## New failure cases found

1. **Only one car is detected per image (11, 12, 13, 16).** Every Stanford Cars training image has exactly one labelled car, so fine-tuning taught the detector to treat other cars in the picture as background. *Fix:* fine-tune on a multi-vehicle dataset, or combine it with the COCO-pretrained detector, which finds all cars.
2. **False positive on an image with no car (15).** The detector was trained on 8,041 images with **0 background images**, so it never learned to output "no car". The confidence threshold caught it. *Fix:* add background images without cars to detector training.
3. **Damaged or occluded car gets a partial box (10).** The crushed front end beyond the tree breaks the car's usual outline, so the box covers only the intact part.
4. **Rear view lowers confidence (07).** The prediction was correct, but confidence fell below the threshold, because the training data is dominated by front and side views.
