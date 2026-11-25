"""
Common pieces shared by every void-detector backend.

WHY A COMMON INTERFACE?
  The ROS node should not care HOW voids are found. Every backend takes one
  BGR image (OpenCV's default colour order) and returns a list of VoidBox.
  Swapping the classical OpenCV detector for the DINOv2 ShelfVoid model (or a
  future YOLO model) is then a one-line parameter change: backend:=...
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VoidBox:
    """One detected empty shelf region, in image pixels."""
    x1: int
    y1: int
    x2: int
    y2: int
    score: float          # 0..1 confidence
    label: str = "void"

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2.0

    @property
    def w(self):
        return self.x2 - self.x1

    @property
    def h(self):
        return self.y2 - self.y1


class VoidDetector:
    """Base class. Subclasses implement detect()."""

    name = "base"

    def detect(self, bgr: np.ndarray) -> list:
        raise NotImplementedError

    # Optional: a picture of what the detector "thinks" (mask / heat-map).
    # Published on a debug topic; handy for presentation slides.
    def debug_image(self):
        return None


def mask_to_boxes(mask: np.ndarray, min_area_px: int = 1500, score_map=None) -> list:
    """Turn a binary void mask (H x W, 0/1) into bounding boxes.

    Used by the ShelfVoid (segmentation) backend: the notebook model outputs a
    per-pixel mask, but the robot should report *boxes*. Each connected blob of
    void pixels becomes one box. Tiny blobs (noise) are dropped.
    """
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    boxes = []
    for i in range(1, n):                       # label 0 = background
        x, y, w, h, area = stats[i]
        if area < min_area_px:
            continue
        if score_map is not None:
            score = float(score_map[y:y + h, x:x + w][mask[y:y + h, x:x + w] > 0].mean())
        else:
            score = 1.0
        boxes.append(VoidBox(int(x), int(y), int(x + w), int(y + h), score))
    return boxes


def draw_boxes(bgr: np.ndarray, boxes: list, tags=None, banner: str = "") -> np.ndarray:
    """Draw the presentation overlay: translucent red fill + thick box + label."""
    out = bgr.copy()
    overlay = out.copy()
    for b in boxes:
        cv2.rectangle(overlay, (b.x1, b.y1), (b.x2, b.y2), (0, 0, 255), -1)
    out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
    for i, b in enumerate(boxes):
        cv2.rectangle(out, (b.x1, b.y1), (b.x2, b.y2), (0, 0, 255), 4)
        text = tags[i] if tags and i < len(tags) and tags[i] else f"VOID {b.score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ty = max(b.y1, th + 8)
        cv2.rectangle(out, (b.x1, ty - th - 8), (b.x1 + tw + 8, ty), (0, 0, 255), -1)
        cv2.putText(out, text, (b.x1 + 4, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
    if banner:
        cv2.rectangle(out, (0, out.shape[0] - 34), (out.shape[1], out.shape[0]), (30, 30, 30), -1)
        cv2.putText(out, banner, (10, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return out
