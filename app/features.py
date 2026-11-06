"""Classical computer-vision components (shared by training script and app).

CLAHE preprocessing, HOG / LBP / HSV-colour / geometric features,
edge + contour based region-of-interest extraction, and visualizations.
Keep this file identical to the FEATURES section of train_classical.py.
"""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage import exposure

SIZE = (128, 128)
PAD = 0.08


# ---------------------------------------------------------------- preprocessing
def apply_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def crop_box(img, box, pad=PAD):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = int(max(0, x1 - pad * bw)), int(max(0, y1 - pad * bh))
    x2, y2 = int(min(w, x2 + pad * bw)), int(min(h, y2 + pad * bh))
    return img[y1:y2, x1:x2]


def preprocess(img_bgr, box=None, pad=PAD, size=SIZE):
    """Crop to the (padded) box, resize, CLAHE on the L channel.
    Returns enhanced BGR crop and its grayscale version."""
    if box is not None:
        img_bgr = crop_box(img_bgr, box, pad)
    img = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    img = apply_clahe(img)
    return img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------- features
def feat_hog(gray):
    return hog(gray, orientations=9, pixels_per_cell=(16, 16),
               cells_per_block=(2, 2), feature_vector=True).astype(np.float32)


def feat_lbp(gray, grid=2, P=8, R=1):
    lbp = local_binary_pattern(gray, P, R, "uniform")
    h, w = lbp.shape
    out = []
    for i in range(grid):
        for j in range(grid):
            cell = lbp[i * h // grid:(i + 1) * h // grid, j * w // grid:(j + 1) * w // grid]
            hist, _ = np.histogram(cell, bins=P + 2, range=(0, P + 2), density=True)
            out.append(hist)
    return np.concatenate(out).astype(np.float32)


def feat_color(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    return (hist / (hist.sum() + 1e-6)).astype(np.float32)


def feat_geom(gray, box_wh):
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = gray.shape[0] * gray.shape[1]
    if contours:
        c = max(contours, key=cv2.contourArea)
        hu = cv2.HuMoments(cv2.moments(c)).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
        largest = cv2.contourArea(c) / area
        x, y, w, h = cv2.boundingRect(c)
        extent = (w * h) / area
    else:
        hu, largest, extent = np.zeros(7), 0.0, 0.0
    bw, bh = box_wh
    return np.concatenate([[bw / max(bh, 1), edges.mean() / 255, largest, extent,
                            len(contours) / 100], hu]).astype(np.float32)


def extract_all(img_bgr, box=None):
    bgr, gray = preprocess(img_bgr, box)
    wh = (box[2] - box[0], box[3] - box[1]) if box is not None else img_bgr.shape[1::-1]
    return {"hog": feat_hog(gray), "lbp": feat_lbp(gray),
            "color": feat_color(bgr), "geom": feat_geom(gray, wh)}


# ------------------------------------------------- classical ROI (no deep learning)
def classical_roi(img_bgr, min_frac=0.05):
    """Edge + contour based vehicle region proposal. Returns (box, edges, contour_vis)."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(apply_clahe(img_bgr), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    closed = cv2.morphologyEx(cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2),
                              cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = img_bgr.copy()
    cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)
    box = (0, 0, w, h)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh > min_frac * w * h:
            box = (x, y, x + bw, y + bh)
    cv2.rectangle(vis, box[:2], box[2:], (255, 0, 255), 3)
    return box, edges, vis


def hog_visualization(img_bgr):
    gray = cv2.cvtColor(cv2.resize(img_bgr, SIZE), cv2.COLOR_BGR2GRAY)
    _, hog_img = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2), visualize=True)
    hog_img = exposure.rescale_intensity(hog_img, out_range=(0, 255)).astype(np.uint8)
    return cv2.resize(hog_img, (256, 256), interpolation=cv2.INTER_NEAREST)
