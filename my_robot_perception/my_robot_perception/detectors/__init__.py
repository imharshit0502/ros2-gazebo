"""Void-detector backends. Pick one with make_detector(name, ...)."""

from .base import VoidBox, VoidDetector, draw_boxes, mask_to_boxes  # noqa: F401


def make_detector(backend: str, **kwargs) -> VoidDetector:
    """
    backend = "classical"  -> OpenCV shelf-rail detector (no training, works out of the box)
              "shelfvoid"  -> DINOv2 ShelfVoid model from the EE-511 notebook
                              (needs PyTorch + weights_path=.../shelfvoid_head.pt)
    """
    if backend == "classical":
        from .shelf_rail_detector import ShelfRailVoidDetector
        return ShelfRailVoidDetector(**kwargs)
    if backend == "shelfvoid":
        from .shelfvoid_dinov2 import ShelfVoidDinoDetector
        return ShelfVoidDinoDetector(**kwargs)
    raise ValueError(f"unknown backend '{backend}' (use 'classical' or 'shelfvoid')")
