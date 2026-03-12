import cv2
import numpy as np

IN_PATH = "BB (62).tif"
OUT_PATH = "BB (62)_pre_soax.tif"

# Load 16-bit TIFF (keep depth)
img = cv2.imread(IN_PATH, cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(f"Could not read {IN_PATH}")

# If it loads as 3-channel, convert to gray
if img.ndim == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1) Mild denoise (median is great for salt-and-pepper texture)
den = cv2.medianBlur(img, 3)

# 2) Background estimate via large Gaussian blur (acts like rolling-ball-ish)
bg = cv2.GaussianBlur(den, (0, 0), sigmaX=15, sigmaY=15)

# Subtract background (keep in signed space then clip)
sub = cv2.subtract(den, bg)

# 3) Contrast stretch: map 1st–99th percentile to full range
p1, p99 = np.percentile(sub, (1, 99))
sub = np.clip(sub, p1, p99)

# Normalize back to 16-bit
out = ((sub - p1) / (p99 - p1 + 1e-9) * 65535.0).astype(np.uint16)

cv2.imwrite(OUT_PATH, out)
print("Wrote:", OUT_PATH, "dtype:", out.dtype, "min/max:", int(out.min()), int(out.max()))