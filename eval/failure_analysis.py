# =============================================================================
# P5 - Failure analysis of the classifier on the test set
# Kaggle: CPU is fine, Internet ON, add the eduardo4jesus/stanford-cars-dataset input
# Outputs (in /kaggle/working):
#   failure_analysis.json     - most-confused pairs, worst classes, error types, confidence stats
#   confusion_pairs.png       - side-by-side examples of the top confused class pairs
#   worst_errors.png          - confident mistakes (high confidence but wrong)
#   confidence_hist.png       - confidence of correct vs wrong predictions
# =============================================================================
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from huggingface_hub import hf_hub_download

REPO, OUT = "vene-05/p5-convnext-cars", "/kaggle/working"
df = pd.read_csv(hf_hub_download(REPO, "test_predictions.csv"))

img_dir = None
for root, dirs, fs in os.walk("/kaggle/input"):
    if "cars_test" in root.lower() and sum(f.endswith(".jpg") for f in fs) > 1000:
        img_dir = root
print("test images:", img_dir, "| predictions:", len(df), "| accuracy:", round(df.correct.mean(), 4))


def split(name):
    """'BMW M3 Coupe 2012' -> make, model-without-year, year"""
    p = name.split()
    return p[0], " ".join(p[:-1]), p[-1]


errs = df[~df.correct].copy()
errs["same_make"] = [split(t)[0] == split(p)[0] for t, p in zip(errs.true, errs.pred)]
errs["same_model_diff_year"] = [split(t)[1] == split(p)[1] and split(t)[2] != split(p)[2]
                                for t, p in zip(errs.true, errs.pred)]
errs["in_top5"] = [t in str(t5).split("|") for t, t5 in zip(errs.true, errs.top5)]

pairs = errs.groupby(["true", "pred"]).size().sort_values(ascending=False)
per_class = df.groupby("true").correct.mean().sort_values()

summary = {
    "n_test": len(df), "n_errors": len(errs), "top1": float(df.correct.mean()),
    "error_types": {
        "same_model_different_year": int(errs.same_model_diff_year.sum()),
        "same_make_different_model": int((errs.same_make & ~errs.same_model_diff_year).sum()),
        "different_make": int((~errs.same_make).sum()),
        "true_class_still_in_top5": int(errs.in_top5.sum()),
    },
    "top10_confused_pairs": [{"true": t, "pred": p, "count": int(c)} for (t, p), c in pairs.head(10).items()],
    "worst10_classes": [{"class": c, "accuracy": round(float(a), 3)} for c, a in per_class.head(10).items()],
    "mean_conf_correct": float(df[df.correct].conf.mean()),
    "mean_conf_wrong": float(errs.conf.mean()),
    "confident_mistakes_conf>=0.7": int((errs.conf >= 0.7).sum()),
}
print(json.dumps(summary, indent=2))
json.dump(summary, open(f"{OUT}/failure_analysis.json", "w"), indent=2)

# ------------------------------ plots ----------------------------------------
plt.figure(figsize=(7, 4))
plt.hist(df[df.correct].conf, bins=40, alpha=0.7, label="correct", density=True)
plt.hist(errs.conf, bins=40, alpha=0.7, label="wrong", density=True)
plt.axvline(0.5, color="k", ls="--", lw=1, label="threshold 0.5")
plt.xlabel("prediction confidence"); plt.ylabel("density"); plt.legend()
plt.title("Confidence: correct vs wrong predictions"); plt.tight_layout()
plt.savefig(f"{OUT}/confidence_hist.png", dpi=130); plt.close()


def show(ax, fname, title):
    ax.imshow(Image.open(os.path.join(img_dir, fname)).convert("RGB")); ax.set_title(title, fontsize=7); ax.axis("off")


if img_dir:
    # top 4 confused pairs: wrong image vs an image that truly is the predicted class
    fig, axes = plt.subplots(4, 2, figsize=(9, 12))
    for row, ((t, p), c) in enumerate(pairs.head(4).items()):
        wrong = errs[(errs.true == t) & (errs.pred == p)].iloc[0].fname
        ref = df[(df.true == p) & df.correct]
        show(axes[row, 0], wrong, f"TRUE: {t}\npredicted: {p} ({c}x)")
        if len(ref):
            show(axes[row, 1], ref.iloc[0].fname, f"an actual {p}")
        else:
            axes[row, 1].axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/confusion_pairs.png", dpi=130); plt.close()

    worst = errs.sort_values("conf", ascending=False).head(8)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, r in zip(axes.flat, worst.itertuples()):
        show(ax, r.fname, f"true: {r.true}\npred: {r.pred} ({r.conf:.0%})")
    plt.suptitle("Confident mistakes"); plt.tight_layout()
    plt.savefig(f"{OUT}/worst_errors.png", dpi=130); plt.close()
print("Done. Download failure_analysis.json and the 3 PNGs into docs/ in the GitHub repo.")
