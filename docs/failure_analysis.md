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

*Three representative failure cases will be documented here using the generated visualizations, including their actual confidence scores, visual causes and possible improvements.*

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
