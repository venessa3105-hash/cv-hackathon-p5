# Failure Analysis – Stanford Cars Classification

## 1. Overview

The failure analysis was conducted on the Stanford Cars test dataset to understand the classification errors made by the trained ConvNeXt model.

### Overall Performance

| Metric | Result |
|---|---:|
| Total test images | 8,041 |
| Correct predictions | 7,612 |
| Incorrect predictions | 429 |
| Top-1 accuracy | 94.66% |
| True class present in Top-5 for incorrect predictions | 375 |

## 2. Error Distribution

| Error Category | Count |
|---|---:|
| Same model, different year | 39 |
| Same make, different model | 264 |
| Different make | 126 |
| True class still in Top-5 | 375 |

The largest error category is confusion between different models from the same manufacturer, accounting for 264 errors.

## 3. Top Confused Class Pairs

| Actual Class | Predicted Class | Errors |
|---|---|---:|
| Audi TTS Coupe 2012 | Audi TT Hatchback 2011 | 16 |
| Audi V8 Sedan 1994 | Audi 100 Sedan 1994 | 12 |
| Dodge Sprinter Cargo Van 2009 | Mercedes-Benz Sprinter Van 2012 | 12 |
| Audi S5 Coupe 2012 | Audi A5 Coupe 2012 | 11 |
| Chevrolet Express Van 2007 | Chevrolet Express Cargo Van 2007 | 11 |

## 4. Confidence Analysis

- Mean confidence for correct predictions: 75.82%.
- Mean confidence for incorrect predictions: 46.20%.
- High-confidence incorrect predictions (confidence ≥ 0.70): 104.

These results indicate that incorrect predictions generally have lower confidence than correct predictions, although some confident mistakes remain.

## 5. Detailed Failure Cases

The three cases below are taken from the test-set errors in [`failure_analysis/confusion_pairs.png`](failure_analysis/confusion_pairs.png) and [`failure_analysis/worst_errors.png`](failure_analysis/worst_errors.png). Each one is a different type of failure.

### Case 1 – Performance variant mistaken for the base model (same make, different model)

| True class | Predicted class | Errors | Confidence of mistake |
|---|---|---:|---:|
| Audi TTS Coupe 2012 | Audi TT Hatchback 2011 | 16 | up to 92% |
| Audi S5 Coupe 2012 | Audi A5 Coupe 2012 | 11 | up to 92% |

- **What happens:** the sporty version (TTS, S5) is predicted as the regular version (TT, A5) of the same car.
- **Why:** the two share the same body shell. The differences are small trim details: badges, grille inserts, wheel design, mirror caps and quad exhaust pipes. At 384 px these details cover only a few pixels, and in side or angled views the badges are often not visible at all.
- **How common:** this is the largest error type, **264 of 429 errors (62%)** are "same make, different model". Audi TTS Coupe 2012 (61.9% accuracy) and Audi S5 Coupe 2012 (69.0%) are among the 3 worst classes.
- **Possible fix:** higher input resolution, or part-based attention on the grille, badges and wheels.

### Case 2 – Same model, different year (facelift)

| True class | Predicted class | Errors | Confidence of mistake |
|---|---|---:|---:|
| Dodge Caliber Wagon 2012 | Dodge Caliber Wagon 2007 | 9 | 93% |
| Dodge Caliber Wagon 2007 | Dodge Caliber Wagon 2012 | 9 | 92% |
| Bentley Continental GT Coupe 2012 | Bentley Continental GT Coupe 2007 | 8 | – |

- **What happens:** the model gets the make and model right but picks the wrong model year. The errors go in both directions (2007 ↔ 2012), which shows the model genuinely cannot separate the two.
- **Why:** a facelift changes only small parts: headlight shape, grille pattern, bumper details. In the worst-errors grid the 2007 Caliber is photographed **from the rear**, where the two years look almost the same.
- **How common:** 39 errors. They are hard to avoid because the visual difference is tiny.
- **Possible fix:** focus on the parts that change in a facelift (lights, grille), plus more training views of each year.

### Case 3 – Near-identical vehicles sold under different names (different make or variant)

| True class | Predicted class | Errors |
|---|---|---:|
| Dodge Sprinter Cargo Van 2009 | Mercedes-Benz Sprinter Van 2012 | 12 |
| Chevrolet Express Van 2007 | Chevrolet Express Cargo Van 2007 | 11 |
| GMC Savana Van 2012 | Chevrolet Express Van 2007 | 8 |

- **What happens:** commercial vans are confused across brands and body variants.
- **Why:** these are **rebadged versions of the same vehicle**. The Dodge Sprinter *is* a Mercedes Sprinter with a different badge, and the GMC Savana and Chevrolet Express share one body. In a side view (like the Sprinter image) the only distinguishing feature, the front badge, is not visible. Passenger vs cargo vans differ only by the side windows.
- **How common:** 126 errors are "different make". Chevrolet Express Van 2007 is the worst class overall (48.6% accuracy).
- **Possible fix:** treat rebadged pairs as one class or report them together, and use front-view crops when available. For these cases the **Top-5 already contains the true class**: across all errors it does so for **375 of 429 (87%)**.

### Failures found on unseen internet images

Testing the live app on 14 internet images (see [`../eval/unseen_results.md`](../eval/unseen_results.md)) found further failures that come from the detector rather than the classifier:

- **Only one car is boxed per image**, because every Stanford Cars training image has exactly one labelled car.
- **False detection on a photo with no car** (a dog, 0.74 confidence), because detector training had 0 background images. The classifier's low confidence (3.5%) caught it, so the result was shown as Uncertain.
- **Partial box on a heavily damaged, occluded car.**

## 6. Possible Improvements

1. Increase input image resolution to capture finer vehicle details.
2. Apply attention mechanisms to focus on distinguishing features such as headlights, grilles and body shape.
3. Improve representation learning for visually similar vehicle models.
4. Include more diverse viewpoints and lighting conditions during training.
5. Use Top-5 predictions to help users identify visually similar alternatives.

## 7. Conclusion

The model achieved 94.66% Top-1 accuracy on 8,041 test images. The analysis shows that most classification errors occur between visually similar vehicle models from the same manufacturer. The findings highlight opportunities to improve fine-grained vehicle classification through better feature extraction, image resolution and viewpoint diversity.

### Supporting Visualizations

- [Confusion Pairs](failure_analysis/confusion_pairs.png)
- [Worst Errors](failure_analysis/worst_errors.png)
- [Confidence Histogram](failure_analysis/confidence_hist.png)
- [Failure Analysis Statistics](failure_analysis/failure_analysis.json)
