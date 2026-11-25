"""
ShelfVoid (DINOv2 ViT-S/14 + prototype head) backend.

This is a direct port of the inference part of the EE-511 notebook
`shelfvoid-1.ipynb`, unchanged so that a head trained on Kaggle works here:

  notebook cell 3  -> to_tensor()        (ImageNet normalisation, RGB, 448x448)
  notebook cell 4  -> frozen DINOv2 ViT-S/14 backbone, x_norm_patchtokens
  notebook cell 7  -> class ShelfVoid    (2-layer projection + prototypes p / v)
  notebook cell 9  -> predict_void_mask  (sigmoid -> bilinear upsample -> 0.5
                                           threshold -> 5x5 open/close)
  NEW              -> mask_to_boxes()    (blob -> bounding box, for the robot)

REQUIREMENTS (not installed on this machine yet - see README):
  * PyTorch  (CPU is fine, ~0.5-1.5 s per frame for ViT-S/14 at 448 px)
  * the trained head  `shelfvoid_head.pt`  saved by notebook cell 14
  * internet ONCE so torch.hub can download DINOv2 (cached in ~/.cache/torch/hub)

DOMAIN-GAP WARNING
  The head was trained on real SKU-110K photos (dense real packaging). Gazebo
  renders flat-coloured boxes, which look very different, so expect weaker results here
  than on Kaggle unless the head is re-trained on simulator frames. The
  notebook's own self-supervised recipe can do that (see README).
"""

import cv2
import numpy as np

from .base import VoidDetector, mask_to_boxes

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _build_head(nn, F, torch, embed_dim):
    """Same layers / parameter names as notebook cell 7 so load_state_dict() matches."""

    class ShelfVoid(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.p = nn.Parameter(torch.zeros(embed_dim))   # "product" prototype
            self.v = nn.Parameter(torch.zeros(embed_dim))   # "void" prototype
            self.temperature = 0.5                          # notebook value (not saved in the .pt)

        def forward(self, patch_tokens):
            f = self.proj(patch_tokens)
            fn = F.normalize(f, dim=-1)
            p = F.normalize(self.p, dim=0)
            v = F.normalize(self.v, dim=0)
            s_p = (fn * p).sum(-1)
            s_v = (fn * v).sum(-1)
            return (s_v - s_p) / self.temperature            # logit, higher = void

    return ShelfVoid()


class ShelfVoidDinoDetector(VoidDetector):
    name = "shelfvoid"

    def __init__(self, weights_path: str, threshold: float = 0.5, min_box_area_frac: float = 0.004,
                 device: str = "auto", hub_repo: str = "facebookresearch/dinov2", hub_source: str = "github"):
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError as e:
            raise RuntimeError(
                "backend 'shelfvoid' needs PyTorch, which is not installed. "
                "See my_robot_perception/README.md, or use backend:=classical") from e
        self.torch, self.F = torch, F

        ckpt = torch.load(weights_path, map_location="cpu")
        cfg = ckpt.get("config", {})
        self.img_size = int(cfg.get("img_size", 448))
        self.grid = int(cfg.get("grid", self.img_size // 14))
        embed_dim = int(cfg.get("embed_dim", 384))
        backbone_name = cfg.get("backbone", "dinov2_vits14")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # hub_source="local" + hub_repo=<path to a dinov2 git checkout> works offline
        self.backbone = torch.hub.load(hub_repo, backbone_name, source=hub_source, trust_repo=True)
        self.backbone = self.backbone.to(device).eval()
        for prm in self.backbone.parameters():
            prm.requires_grad = False

        self.head = _build_head(nn, F, torch, embed_dim)
        self.head.load_state_dict(ckpt["state_dict"])
        self.head = self.head.to(device).eval()

        self.threshold = float(threshold)
        self.min_box_area_frac = float(min_box_area_frac)
        self._heat = None

    def _to_tensor(self, rgb_u8):
        img = rgb_u8.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))
        return self.torch.from_numpy(img.astype(np.float32)).unsqueeze(0).to(self.device)

    def predict_void_mask(self, rgb_u8):
        """Notebook cell 9, unchanged except for the input resize."""
        torch, F = self.torch, self.F
        img = cv2.resize(rgb_u8, (self.img_size, self.img_size))
        with torch.no_grad():
            tokens = self.backbone.forward_features(self._to_tensor(img))["x_norm_patchtokens"]
            prob = torch.sigmoid(self.head(tokens))
            prob_grid = prob.view(1, 1, self.grid, self.grid)
            up = F.interpolate(prob_grid, size=(self.img_size, self.img_size),
                               mode="bilinear", align_corners=False)
        heat = up.squeeze().cpu().numpy()
        pred = (heat > self.threshold).astype(np.uint8)
        k = np.ones((5, 5), np.uint8)
        pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, k)
        pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, k)
        return pred, heat

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)          # notebook trained on RGB
        pred, heat = self.predict_void_mask(rgb)
        # back to camera resolution, then mask -> boxes
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
        self._heat = (bgr, heat)
        return mask_to_boxes(pred, min_area_px=int(self.min_box_area_frac * w * h), score_map=heat)

    def debug_image(self):
        """P(void) heat-map blended over the camera image (like notebook figure 3)."""
        if self._heat is None:
            return None
        bgr, heat = self._heat
        colour = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        return cv2.addWeighted(bgr, 0.5, colour, 0.5, 0)
