#!/usr/bin/env python3
"""
Run a void detector on saved images, without ROS or Gazebo.

Useful to (a) tune the detector on screenshots, (b) make figures for slides,
(c) test the ShelfVoid notebook model on a laptop/Kaggle.

  ros2 run my_robot_perception detect_voids_offline <image or folder> [-o out_dir]
  ros2 run my_robot_perception detect_voids_offline img.png --backend shelfvoid --weights shelfvoid_head.pt
"""

import argparse
import glob
import os

import cv2

from my_robot_perception.detectors import draw_boxes, make_detector


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="image files or folders")
    ap.add_argument("-o", "--out", default="void_results", help="output folder (default ./void_results)")
    ap.add_argument("--backend", default="classical", choices=["classical", "shelfvoid"])
    ap.add_argument("--weights", default="", help="shelfvoid_head.pt (backend shelfvoid)")
    ap.add_argument("--debug", action="store_true", help="also save the detector's debug view")
    a = ap.parse_args()

    if a.backend == "shelfvoid":
        det = make_detector("shelfvoid", weights_path=os.path.expanduser(a.weights))
    else:
        det = make_detector("classical")

    files = []
    for p in a.inputs:
        if os.path.isdir(p):
            for ext in ("png", "jpg", "jpeg"):
                files += glob.glob(os.path.join(p, f"*.{ext}"))
        else:
            files.append(p)
    os.makedirs(a.out, exist_ok=True)

    for f in sorted(files):
        img = cv2.imread(f)
        if img is None:
            print(f"skip (not an image): {f}")
            continue
        boxes = det.detect(img)
        name = os.path.splitext(os.path.basename(f))[0]
        cv2.imwrite(os.path.join(a.out, f"{name}_voids.png"),
                    draw_boxes(img, boxes, banner=f"[{det.name}] {len(boxes)} void(s)"))
        if a.debug and det.debug_image() is not None:
            cv2.imwrite(os.path.join(a.out, f"{name}_debug.png"), det.debug_image())
        desc = ", ".join(f"({b.x1},{b.y1})-({b.x2},{b.y2}) s={b.score:.2f}" for b in boxes) or "none"
        print(f"{os.path.basename(f):28s} {len(boxes)} void(s): {desc}")
    print(f"annotated images -> {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
