import re
import numpy as np
import cv2

TXT_PATH = "img20_snakes.txt"   # change if needed
OUT_BINARY = "snakes20_binary.png"
OUT_THICK  = "snakes20_thick.png"
OUT_INTENS = "snakes20_intensity.png"

def load_soax_snakes(txt_path):
    """
    Returns:
      snakes: dict[int, list[tuple[float,float]]]
              snake_id -> [(x,y), (x,y), ...] in point order
      fg:     dict[int, list[float]]
              snake_id -> [fg_int, fg_int, ...] aligned with points
    """
    snakes = {}
    fg = {}

    data_started = False

    with open(txt_path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # SOAX has a header; data begins after the column header line:
            # "s p x y z fg_int bg_int"
            if not data_started:
                if line.startswith("s") and "fg_int" in line and "bg_int" in line:
                    data_started = True
                continue

            # Lines like "#0" mark snake sections; we can ignore them
            if line.startswith("#"):
                continue

            # Parse numeric row: s p x y z fg_int bg_int
            parts = re.split(r"\s+", line)
            if len(parts) < 7:
                continue

            s = int(parts[0])          # snake index
            p = int(parts[1])          # point index (we don't strictly need it)
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])        # often 0 for 2D
            fg_int = float(parts[5])

            snakes.setdefault(s, []).append((x, y))
            fg.setdefault(s, []).append(fg_int)

    return snakes, fg

def infer_canvas_size(snakes, pad=5):
    xs, ys = [], []
    for pts in snakes.values():
        for x, y in pts:
            xs.append(x)
            ys.append(y)
    if not xs:
        raise ValueError("No snake points found in file.")

    max_x = int(np.ceil(max(xs))) + pad
    max_y = int(np.ceil(max(ys))) + pad
    return max_x, max_y  # width, height

def draw_snakes_binary(snakes, width, height, thickness=1):
    img = np.zeros((height, width), dtype=np.uint8)

    for s, pts in snakes.items():
        if len(pts) < 2:
            continue
        poly = np.array([(int(round(x)), int(round(y))) for x, y in pts], dtype=np.int32)
        poly = poly.reshape((-1, 1, 2))
        cv2.polylines(img, [poly], isClosed=False, color=255, thickness=thickness)

    return img

def draw_snakes_intensity(snakes, fg, width, height, thickness=1):
    """
    Makes a grayscale image where each snake segment brightness is based on fg_int.
    We normalize fg_int to 0..255 for display.
    """
    # Collect all fg_int for normalization
    all_fg = [v for arr in fg.values() for v in arr]
    if not all_fg:
        raise ValueError("No fg_int values found.")

    lo, hi = float(min(all_fg)), float(max(all_fg))
    if hi == lo:
        hi = lo + 1.0

    img = np.zeros((height, width), dtype=np.uint8)

    for s, pts in snakes.items():
        if len(pts) < 2:
            continue

        # Draw as small segments so we can vary intensity along the snake
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            val = fg[s][i]
            intensity = int(round(255.0 * (val - lo) / (hi - lo)))

            cv2.line(
                img,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                color=intensity,
                thickness=thickness,
            )

    return img

def main():
    snakes, fg = load_soax_snakes(TXT_PATH)

    width, height = infer_canvas_size(snakes, pad=5)

    # 1) clean 1-px skeleton-like mask
    binary = draw_snakes_binary(snakes, width, height, thickness=1)
    cv2.imwrite(OUT_BINARY, binary)

    # 2) thicker mask (easier to see / overlap with microscopy)
    thick = draw_snakes_binary(snakes, width, height, thickness=3)
    cv2.imwrite(OUT_THICK, thick)

    # 3) intensity-weighted visualization (optional)
    intens = draw_snakes_intensity(snakes, fg, width, height, thickness=2)
    cv2.imwrite(OUT_INTENS, intens)

    print("Wrote:", OUT_BINARY, OUT_THICK, OUT_INTENS)
    print("Canvas size (width,height):", width, height)
    print("Snakes:", len(snakes))

if __name__ == "__main__":
    main()