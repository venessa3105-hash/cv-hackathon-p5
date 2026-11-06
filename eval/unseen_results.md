# Results on unseen internet images (live app)

App: https://huggingface.co/spaces/vene-05/p5-vehicle-recognition · threshold 0.5 · classifier: FILL (FP32/INT8) · device: FILL

| Image | True class / expected | Hybrid prediction | Conf. | DL-only prediction | Conf. | Correct / handled? | Total time | Notes |
|---|---|---|---|---|---|---|---|---|
| 01_normal_bmw_m3 | BMW M3 Coupe 2012 | | | | | | | |
| 03_normal_ford_f150_supercrew | F-150 SuperCrew (nearest: F-150 Regular Cab 2012) | | | | | | | |
| 04_normal_tesla_s | Tesla Model S Sedan 2012 | | | | | | | |
| 05_night_camaro | Chevrolet Camaro Convertible 2012 | | | | | | | CLAHE applied? |
| 06_night_camry | Toyota Camry Sedan 2012 | | | | | | | CLAHE applied? |
| 07_rear_jeep | Jeep Wrangler SUV 2012 | | | | | | | |
| 08_side_mustang | Ford Mustang Convertible 2007 | | | | | | | |
| 09_occluded_tree | Not in dataset → Uncertain | | | | | | | |
| 10_occluded_damaged | Not in dataset → Uncertain | | | | | | | |
| 11_multi_parking | Multiple cars → several boxes | | | | | | | how many boxes? |
| 12_multi_street | Multiple vehicles → several boxes | | | | | | | how many boxes? |
| 13_new_tesla_y | Not in dataset → Uncertain | | | | | | | |
| 14_new_hyundai_ioniq5 | Not in dataset → Uncertain | | | | | | | |
| 15_not_car | No car → no confident label | | | | | | | |
| 16_ood_byd_qin | Not in dataset → Uncertain | | | | | | | |

## Summary
- In-dataset cars (01, 04–08): FILL / 6 correct Top-1, FILL / 6 in Top-5
- Out-of-dataset cars flagged Uncertain: FILL / 5
- Multi-car images: FILL
- Observations: FILL
