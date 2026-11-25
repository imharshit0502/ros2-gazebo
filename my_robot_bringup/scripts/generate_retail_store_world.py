#!/usr/bin/env python3
"""
Retail store world generator (Gazebo Harmonic / gz-sim 8, SDF 1.9)
===================================================================

WHY A GENERATOR?
  A realistic grocery store has thousands of product boxes. Writing those by
  hand in XML is error-prone, so this script *computes* the world instead.
  Edit the LAYOUT / VOIDS / PEDESTRIANS sections below, re-run the script and
  the .world file (plus a void ground-truth YAML for your CV pipeline) are
  regenerated deterministically (fixed random seed -> same store every time).

USAGE (from anywhere):
  python3 generate_retail_store_world.py
  -> writes ../worlds/retail_store.world
  -> writes ../config/shelf_voids.yaml   (ground truth of every empty spot)
  Then rebuild:  cd ~/ros2_rws && colcon build --packages-select my_robot_bringup

STORE LAYOUT (top-down, metres, x = East, y = North, origin = store centre)

  y=+9  ┌──────────────── north wall (high windows) ────────────────┐
        │                  back corridor (2.4 m)                    │
  y=6.6 │W ┌──┐   ┌──┐    ┌──┐   ┌──┐ E│
        │A │N1│   │N2│    │N3│   │N4│ A│   <- row N gondolas (3.6 m long)
        │L │  │   │  │    │  │   │  │ L│
  y=3.0 │L └──┘   └──┘    └──┘   └──┘ L│
        │ S  ===== cross aisle (2.4 m) =====   S│
  y=0.6 │H ┌──┐   ┌──┐    ┌──┐   ┌──┐ H│
        │E │S1│   │S2│    │S3│   │S4│ E│   <- row S gondolas
        │L │  │   │  │    │  │   │  │ L│
  y=-3  │F └──┘   └──┘    └──┘   └──┘ F│
        │        front promenade (2.1 m)                          │
        │                          [C1]   [C2]   [C3]  checkouts  │
  y=-9  └──win──[ ENTRANCE ]─────storefront glass─────[  EXIT  ]──┘
        x=-9    -6.5    -3.5                          4.5     7.5   x=+9

  Aisles (N-S, each 2.68 m wide, centre x):  A1=-7.16 A2=-3.58 A3=0 A4=+3.58 A5=+7.16
  Robot footprint is 0.6 x 0.4 m, so every aisle leaves > 1 m clearance per side.
  The world origin (0,0) is in the middle of aisle 3 -> safe default spawn.
"""

import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
WORLD_OUT = os.path.join(PKG, "worlds", "retail_store.world")
# Same store WITHOUT pedestrians. Used for SLAM mapping: moving people would be
# drawn into the map as "ghost" obstacles, so we map an empty store once and
# then navigate in the busy one (Nav2's costmaps handle people at run time).
MAPPING_WORLD_OUT = os.path.join(PKG, "worlds", "retail_store_mapping.world")
VOIDS_OUT = os.path.join(PKG, "config", "shelf_voids.yaml")

rng = random.Random(42)  # fixed seed -> reproducible planogram

# ════════════════════════════ LAYOUT CONSTANTS ════════════════════════════
STORE_X = 9.0          # inner wall faces at x = ±9
STORE_Y = 9.0          # inner wall faces at y = ±9
WALL_T = 0.2           # wall thickness
WALL_H = 3.0           # wall height (no roof!)

SHELF_H = 1.75                         # height of shelf back panel
LEVELS = [0.12, 0.52, 0.92, 1.32]      # top surface of each shelf level (0 = base deck)
LEVEL_CLEAR = 0.33                     # usable product height on each level
BAY_LEN = 1.2                          # one "bay" = section between two uprights

GONDOLA_DEPTH = 0.90                   # double-sided island shelf
SPINE_T = 0.04
GONDOLA_LEN = 3.6
ROW_Y = {"S": -1.2, "N": 4.8}          # centre y of each gondola row
GONDOLA_X = [-5.37, -1.79, 1.79, 5.37]

WALL_SHELF_DEPTH = 0.5                 # single-sided shelves along E/W walls
WALL_SHELF_Y = (-3.0, 6.6)

AISLE_X = {1: -7.16, 2: -3.58, 3: 0.0, 4: 3.58, 5: 7.16}

BACK_PANEL_RGB = (0.80, 0.81, 0.83)    # uniform light grey -> what a void looks like

# ═══════════════════════════ PRODUCT CATALOGUE ════════════════════════════
# name: (width along shelf, depth, height, shape)
PRODUCTS = {
    "cereal_box":   (0.19, 0.07, 0.29, "box"),
    "oats_tub":     (0.11, 0.11, 0.18, "cyl"),
    "chips_bag":    (0.17, 0.08, 0.26, "box"),
    "cracker_box":  (0.15, 0.06, 0.22, "box"),
    "cookie_pack":  (0.20, 0.06, 0.07, "box"),
    "soda_bottle":  (0.09, 0.09, 0.30, "cyl"),
    "water_bottle": (0.075, 0.075, 0.24, "cyl"),
    "soda_can":     (0.07, 0.07, 0.12, "cyl"),
    "juice_carton": (0.095, 0.095, 0.25, "box"),
    "tin_can":      (0.08, 0.08, 0.11, "cyl"),
    "glass_jar":    (0.09, 0.09, 0.14, "cyl"),
    "pasta_box":    (0.13, 0.06, 0.24, "box"),
    "rice_bag":     (0.20, 0.09, 0.30, "box"),
    "sauce_bottle": (0.07, 0.07, 0.21, "cyl"),
    "detergent":    (0.20, 0.12, 0.30, "box"),
    "paper_towel":  (0.13, 0.13, 0.28, "cyl"),
    "tissue_box":   (0.24, 0.12, 0.11, "box"),
    "shampoo":      (0.08, 0.05, 0.22, "box"),
    "soap_pack":    (0.12, 0.07, 0.06, "box"),
    "diaper_pack":  (0.30, 0.20, 0.30, "box"),
    "pet_food_bag": (0.28, 0.12, 0.32, "box"),
    "coffee_jar":   (0.10, 0.10, 0.16, "cyl"),
    "tea_box":      (0.15, 0.07, 0.08, "box"),
    "flour_bag":    (0.16, 0.09, 0.26, "box"),
}

# category: (topper sign colour, [product types])
CATEGORIES = {
    "breakfast":     ((0.95, 0.60, 0.10), ["cereal_box", "oats_tub", "cereal_box"]),
    "snacks":        ((0.90, 0.20, 0.15), ["chips_bag", "cracker_box", "cookie_pack"]),
    "beverages":     ((0.10, 0.45, 0.85), ["soda_bottle", "water_bottle", "soda_can", "juice_carton"]),
    "canned_food":   ((0.55, 0.35, 0.20), ["tin_can", "glass_jar"]),
    "pasta_rice":    ((0.85, 0.75, 0.15), ["pasta_box", "rice_bag", "sauce_bottle"]),
    "household":     ((0.15, 0.65, 0.65), ["detergent", "paper_towel", "tissue_box"]),
    "personal_care": ((0.75, 0.35, 0.65), ["shampoo", "soap_pack"]),
    "baby_pet":      ((0.45, 0.70, 0.30), ["diaper_pack", "pet_food_bag"]),
    "tea_coffee":    ((0.40, 0.25, 0.15), ["coffee_jar", "tea_box"]),
    "baking":        ((0.95, 0.85, 0.65), ["flour_bag", "glass_jar", "sauce_bottle"]),
}

# Saturated packaging colours. Deliberately NO near-white/grey products, so an
# empty spot (light-grey back panel) contrasts clearly with stocked product.
BRAND_COLOURS = [
    (0.82, 0.12, 0.12), (0.95, 0.78, 0.08), (0.10, 0.30, 0.78), (0.12, 0.58, 0.22),
    (0.95, 0.48, 0.08), (0.45, 0.18, 0.62), (0.45, 0.26, 0.12), (0.00, 0.55, 0.58),
    (0.90, 0.35, 0.58), (0.12, 0.12, 0.14), (0.35, 0.65, 0.95), (0.62, 0.78, 0.12),
    (0.70, 0.05, 0.30), (0.05, 0.20, 0.40),
]

# Which category is stocked on each shelf face.  Face "W" faces -x, "E" faces +x.
FACE_CATEGORY = {
    ("WALL_W", "E"): "beverages",
    ("S1", "W"): "beverages",     ("S1", "E"): "snacks",
    ("S2", "W"): "snacks",        ("S2", "E"): "breakfast",
    ("S3", "W"): "tea_coffee",    ("S3", "E"): "baking",
    ("S4", "W"): "pasta_rice",    ("S4", "E"): "canned_food",
    ("N1", "W"): "beverages",     ("N1", "E"): "canned_food",
    ("N2", "W"): "pasta_rice",    ("N2", "E"): "breakfast",
    ("N3", "W"): "snacks",        ("N3", "E"): "personal_care",
    ("N4", "W"): "baby_pet",      ("N4", "E"): "household",
    ("WALL_E", "W"): "household",
}

# ═════════════════════════════ SHELF VOIDS ════════════════════════════════
# (shelf, face, bay, level, start_fraction, end_fraction) of the bay left EMPTY.
# bay 0 = southern-most bay; level 0 = bottom deck (closest to robot camera).
VOIDS = [
    ("S2", "E", 1, 1, 0.0, 1.0),
    ("S3", "W", 0, 0, 0.3, 0.8),
    ("N2", "E", 2, 2, 0.0, 1.0),
    ("N3", "W", 1, 1, 0.0, 0.5),
    ("S1", "E", 2, 0, 0.0, 1.0),
    ("N2", "W", 0, 0, 0.0, 1.0),
    ("N1", "W", 0, 3, 0.4, 1.0),
    ("S3", "E", 2, 3, 0.0, 1.0),
    ("N4", "W", 2, 1, 0.2, 0.7),
    ("S4", "E", 1, 2, 0.0, 1.0),
    ("WALL_W", "E", 3, 1, 0.0, 1.0),
    ("WALL_E", "W", 5, 0, 0.5, 1.0),
]

# ═════════════════════ PEDESTRIANS (dynamic obstacles) ════════════════════
# Each walks a closed loop of (x, y) waypoints at a slow shopper pace.
WALK_SPEED = 0.45      # m/s  (slow stroll)
TURN_TIME = 1.2        # s to turn on the spot at a corner
# Shoppers "keep right" like real people do (0.6 m right of the aisle centre line,
# in their walking direction). Gazebo actors are scripted and blind - they never
# step around anything - so a shopper walking exactly down the robot's centre line
# (and faster than it) would walk straight through it from behind, which no robot
# could avoid. Keeping right leaves room for the robot to steer past them.
PEDESTRIANS = {
    # big loop: up aisle 2 (east side), back corridor (south side), down aisle 4 (west side),
    # front promenade (north side, only 0.3 m offset because the gondola ends are close)
    "shopper_1": [(-2.98, -3.4), (-2.98, 7.2), (2.98, 7.2), (2.98, -3.4)],
    # back and forth along the central cross aisle, crossing aisles 2-4 in front of the
    # robot. Turns around at x = +/-6.0 (edge of aisles 1 / 5) instead of inside them, so it
    # never walks into a robot that has stopped to let it pass.
    "shopper_2": [(-6.0, 1.8), (6.0, 1.8)],
    # entrance -> front promenade -> through checkout lane 1 -> back to entrance
    # (walks the south half of the promenade, y = -4.85, just clear of the checkout counters,
    #  so the robot's promenade lane further north stays free)
    "shopper_3": [(-5.0, -7.8), (-5.0, -4.85), (2.8, -4.85), (2.8, -7.8)],
}
ACTOR_MESH = "https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae"


# ═══════════════════════════ SDF HELPER FUNCTIONS ═════════════════════════
def f(v):
    return f"{v:.4f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def pose(x, y, z, yaw=0.0, roll=0.0, pitch=0.0):
    return f"<pose>{f(x)} {f(y)} {f(z)} {f(roll)} {f(pitch)} {f(yaw)}</pose>"


def material(rgb, alpha=1.0, spec=0.1, emissive=None):
    r, g, b = rgb
    em = f"<emissive>{f(emissive[0])} {f(emissive[1])} {f(emissive[2])} 1</emissive>" if emissive else ""
    return (f"<material><ambient>{f(r)} {f(g)} {f(b)} {f(alpha)}</ambient>"
            f"<diffuse>{f(r)} {f(g)} {f(b)} {f(alpha)}</diffuse>"
            f"<specular>{f(spec)} {f(spec)} {f(spec)} 1</specular>{em}</material>")


class Counter:
    n = 0

    @classmethod
    def next(cls, prefix):
        cls.n += 1
        return f"{prefix}_{cls.n}"


def geom_box(sx, sy, sz):
    return f"<geometry><box><size>{f(sx)} {f(sy)} {f(sz)}</size></box></geometry>"


def geom_cyl(r, length):
    return f"<geometry><cylinder><radius>{f(r)}</radius><length>{f(length)}</length></cylinder></geometry>"


def visual_box(x, y, z, sx, sy, sz, rgb, prefix="v", alpha=1.0, shadows=True, spec=0.1, emissive=None):
    tr = f"<transparency>{f(1.0 - alpha)}</transparency>" if alpha < 1.0 else ""
    sh = "" if shadows else "<cast_shadows>false</cast_shadows>"
    return (f'<visual name="{Counter.next(prefix)}">{pose(x, y, z)}{geom_box(sx, sy, sz)}'
            f"{material(rgb, alpha, spec, emissive)}{tr}{sh}</visual>")


def visual_cyl(x, y, z, r, length, rgb, prefix="v", shadows=True, spec=0.1):
    sh = "" if shadows else "<cast_shadows>false</cast_shadows>"
    return (f'<visual name="{Counter.next(prefix)}">{pose(x, y, z)}{geom_cyl(r, length)}'
            f"{material(rgb, 1.0, spec)}{sh}</visual>")


def collision_box(x, y, z, sx, sy, sz, prefix="c"):
    return f'<collision name="{Counter.next(prefix)}">{pose(x, y, z)}{geom_box(sx, sy, sz)}</collision>'


def static_model(name, x, y, body, comment=""):
    c = f"\n    <!-- {comment} -->" if comment else ""
    return (f'{c}\n    <model name="{name}">\n      <static>true</static>\n      {pose(x, y, 0)}\n'
            f'      <link name="link">\n        ' + "\n        ".join(body) +
            "\n      </link>\n    </model>\n")


def jitter(rgb, amt=0.05):
    return tuple(min(1.0, max(0.0, c + rng.uniform(-amt, amt))) for c in rgb)


# ═══════════════════════════════ SHELVES ══════════════════════════════════
void_records = []


def faces_for_aisle(face_x_front):
    """Return the aisle number whose centre is closest to a shelf's front edge."""
    return min(AISLE_X, key=lambda a: abs(AISLE_X[a] - face_x_front))


def stock_face(shelf, face, cx, cy, xb_local, s, y0_local, length, depth):
    """Populate one shelf face with products.
    xb_local: local x of the back panel surface; s: +1 faces +x, -1 faces -x.
    Products are pushed to the FRONT edge (as store staff "face" them)."""
    out = []
    cat = FACE_CATEGORY[(shelf, face)]
    types = CATEGORIES[cat][1]
    n_bays = int(round(length / BAY_LEN))
    front_x_world = cx + xb_local + s * depth
    aisle = faces_for_aisle(front_x_world)

    for bay in range(n_bays):
        bay_y0 = y0_local + bay * BAY_LEN
        for lvl, z0 in enumerate(LEVELS):
            # two brand blocks per bay-level (like a real planogram)
            blocks = []
            split = rng.choice([0.4, 0.5, 0.6])
            for (a, b) in ((0.0, split), (split, 1.0)):
                blocks.append((a, b, rng.choice(types), jitter(rng.choice(BRAND_COLOURS))))

            void = [v for v in VOIDS if v[0] == shelf and v[1] == face and v[2] == bay and v[3] == lvl]
            vlo = vhi = None
            if void:
                _, _, _, _, fa, fb = void[0]
                margin = 0.03
                vlo = bay_y0 + margin + fa * (BAY_LEN - 2 * margin)
                vhi = bay_y0 + margin + fb * (BAY_LEN - 2 * margin)
                xa, xb2 = sorted([cx + xb_local, front_x_world])
                void_records.append({
                    "id": f"V{len(void_records) + 1:02d}",
                    "shelf": shelf, "face": face, "category": cat,
                    "aisle": aisle, "bay": bay, "level": lvl,
                    "x_range": [round(xa, 3), round(xb2, 3)],
                    "y_range": [round(cy + vlo, 3), round(cy + vhi, 3)],
                    "z_range": [round(z0, 3), round(z0 + LEVEL_CLEAR, 3)],
                    "centre": [round(front_x_world, 3), round(cy + (vlo + vhi) / 2, 3) + 0.0, round(z0 + LEVEL_CLEAR / 2, 3)],
                })

            for (a, b, ptype, rgb) in blocks:
                w, d, h, shape = PRODUCTS[ptype]
                gap = 0.012
                ys = bay_y0 + 0.03 + a * (BAY_LEN - 0.06)
                ye = bay_y0 + 0.03 + b * (BAY_LEN - 0.06)
                n = int((ye - ys + gap) // (w + gap))
                if n <= 0:
                    continue
                used = n * w + (n - 1) * gap
                start = ys + (ye - ys - used) / 2 + w / 2
                stack = max(1, min(3, int(LEVEL_CLEAR // (h + 0.005))))
                for i in range(n):
                    py = start + i * (w + gap)
                    if vlo is not None and vlo - w / 2 < py < vhi + w / 2:
                        continue  # leave this spot EMPTY (void)
                    if shape == "box":
                        # one deep box = a full row of identical products behind each other
                        pd = min(depth - 0.03, d * max(1, int((depth - 0.03) // d)))
                        px = xb_local + s * (depth - 0.01 - pd / 2)
                        ph = stack * h + (stack - 1) * 0.004
                        out.append(visual_box(px, py, z0 + ph / 2 + 0.001, pd, w, ph,
                                              rgb, "p", shadows=False, spec=0.25))
                    else:
                        r = w / 2
                        px = xb_local + s * (depth - 0.01 - r)
                        ph = stack * h + (stack - 1) * 0.004
                        out.append(visual_cyl(px, py, z0 + ph / 2 + 0.001, r, ph,
                                              rgb, "p", shadows=False, spec=0.35))
    return out


def shelf_structure_face(xb_local, s, y0_local, length, depth, cat_rgb):
    """Shelf boards, base deck, price strips and category topper for one face."""
    out = []
    xc = xb_local + s * depth / 2
    yc = y0_local + length / 2
    xfront = xb_local + s * (depth - 0.006)
    # base deck / kick plate
    out.append(visual_box(xc, yc, LEVELS[0] / 2, depth, length, LEVELS[0], (0.25, 0.26, 0.28), "deck"))
    out.append(visual_box(xfront, yc, 0.07, 0.012, length, 0.04, (0.95, 0.85, 0.20), "price", shadows=False))
    for z in LEVELS[1:]:
        out.append(visual_box(xc, yc, z - 0.0125, depth, length, 0.025, (0.62, 0.64, 0.67), "board", spec=0.4))
        out.append(visual_box(xfront, yc, z - 0.02, 0.012, length, 0.04, (0.95, 0.85, 0.20), "price", shadows=False))
    # category topper sign above the shelf (visible from the aisle & from above)
    out.append(visual_box(xb_local + s * 0.03, yc, SHELF_H + 0.14, 0.04, length, 0.26, cat_rgb, "topper",
                          emissive=tuple(c * 0.25 for c in cat_rgb)))
    return out


def uprights(x_lo, x_hi, y0_local, length):
    out = []
    n = int(round(length / BAY_LEN))
    for k in range(n + 1):
        y = y0_local + k * BAY_LEN
        y = min(max(y, y0_local + 0.0125), y0_local + length - 0.0125)
        out.append(visual_box((x_lo + x_hi) / 2, y, SHELF_H / 2, x_hi - x_lo, 0.025, SHELF_H,
                              (0.35, 0.36, 0.38), "upright"))
    return out


def gondola(name, cx, cy):
    L, D = GONDOLA_LEN, (GONDOLA_DEPTH - SPINE_T) / 2
    y0 = -L / 2
    body = [
        # ONE simple collision box for the whole shelf -> clean LiDAR/costmap outline
        collision_box(0, 0, SHELF_H / 2, GONDOLA_DEPTH, L, SHELF_H),
        visual_box(0, 0, SHELF_H / 2, SPINE_T, L, SHELF_H, BACK_PANEL_RGB, "back"),
        visual_box(0, 0, SHELF_H + 0.005, GONDOLA_DEPTH, L, 0.01, (0.55, 0.57, 0.60), "cap"),
    ]
    body += uprights(-GONDOLA_DEPTH / 2, GONDOLA_DEPTH / 2, y0, L)
    for face, s in (("W", -1), ("E", +1)):
        xb = s * SPINE_T / 2
        cat_rgb = CATEGORIES[FACE_CATEGORY[(name, face)]][0]
        body += shelf_structure_face(xb, s, y0, L, D, cat_rgb)
        body += stock_face(name, face, cx, cy, xb, s, y0, L, D)
    return static_model(f"shelf_{name}", cx, cy, body,
                        f"Gondola {name}: W face={FACE_CATEGORY[(name, 'W')]}, E face={FACE_CATEGORY[(name, 'E')]}")


def wall_shelf(name, face, s):
    """Single-sided shelf against the west (s=+1, faces east) or east (s=-1) wall."""
    y_lo, y_hi = WALL_SHELF_Y
    L = y_hi - y_lo
    cx = -s * (STORE_X - WALL_SHELF_DEPTH / 2)
    cy = (y_lo + y_hi) / 2
    back_t = 0.03
    xb = -s * (WALL_SHELF_DEPTH / 2 - back_t)       # local x of back surface
    D = WALL_SHELF_DEPTH - back_t
    y0 = -L / 2
    cat_rgb = CATEGORIES[FACE_CATEGORY[(name, face)]][0]
    body = [
        collision_box(0, 0, SHELF_H / 2, WALL_SHELF_DEPTH, L, SHELF_H),
        visual_box(-s * (WALL_SHELF_DEPTH / 2 - back_t / 2), 0, SHELF_H / 2, back_t, L, SHELF_H, BACK_PANEL_RGB, "back"),
        visual_box(0, 0, SHELF_H + 0.005, WALL_SHELF_DEPTH, L, 0.01, (0.55, 0.57, 0.60), "cap"),
    ]
    body += uprights(-WALL_SHELF_DEPTH / 2, WALL_SHELF_DEPTH / 2, y0, L)
    body += shelf_structure_face(xb, s, y0, L, D, cat_rgb)
    body += stock_face(name, face, cx, cy, xb, s, y0, L, D)
    return static_model(f"shelf_{name}", cx, cy, body, f"Wall shelf {name}: {FACE_CATEGORY[(name, face)]}")


# ════════════════════════════ WALLS & WINDOWS ═════════════════════════════
WALL_RGB = (0.93, 0.91, 0.86)
TRIM_RGB = (0.20, 0.35, 0.55)
GLASS_RGB = (0.55, 0.75, 0.90)


def wall(axis, fixed, a0, a1, openings):
    """Build one straight wall from a0 to a1 along `axis` ('x' or 'y') at
    coordinate `fixed` on the other axis. openings = list of
    (start, end, bottom, top, kind) with kind in {'door', 'window'}.
    Solid parts get collisions; windows get collidable glass; doors are open."""
    out = []

    def piece(s0, s1, z0, z1, rgb, glass=False, collide=True, prefix="wall"):
        length, hgt = s1 - s0, z1 - z0
        if length <= 1e-3 or hgt <= 1e-3:
            return
        c = (s0 + s1) / 2
        x, y = (c, fixed) if axis == "x" else (fixed, c)
        sx, sy = (length, WALL_T) if axis == "x" else (WALL_T, length)
        if glass:
            sx, sy = (length, 0.03) if axis == "x" else (0.03, length)
            out.append(visual_box(x, y, (z0 + z1) / 2, sx, sy, hgt, GLASS_RGB, "glass", alpha=0.35, shadows=False, spec=0.9))
        else:
            out.append(visual_box(x, y, (z0 + z1) / 2, sx, sy, hgt, rgb, prefix))
        if collide:
            out.append(collision_box(x, y, (z0 + z1) / 2, sx, sy, hgt))

    cursor = a0
    for (s0, s1, b, t, kind) in sorted(openings):
        piece(cursor, s0, 0, WALL_H, WALL_RGB)
        piece(s0, s1, t, WALL_H, WALL_RGB)                      # header above
        if kind == "window":
            piece(s0, s1, 0, b, WALL_RGB)                       # sill wall below
            piece(s0, s1, b, t, None, glass=True)               # glass pane
            # window frame (sill + head trim) for a realistic look
            piece(s0, s1, b - 0.05, b, TRIM_RGB, collide=False, prefix="frame")
        cursor = s1
    piece(cursor, a1, 0, WALL_H, WALL_RGB)
    # coloured trim band around the top of the wall (looks nice from above)
    piece(a0, a1, WALL_H - 0.15, WALL_H, TRIM_RGB, collide=False, prefix="trim")
    return out


ENTRANCE = (-6.5, -3.5)
EXIT = (4.5, 7.5)


def walls():
    e = STORE_X + WALL_T          # outer extent incl. corners
    wy = STORE_Y + WALL_T / 2
    wx = STORE_X + WALL_T / 2
    body = []
    # South wall = storefront: entrance, exit and big glass windows
    body += wall("x", -wy, -e, e, [
        (-8.6, -7.0, 0.5, 2.4, "window"),
        ENTRANCE + (0, 2.5, "door"),
        (-2.8, 3.8, 0.5, 2.4, "window"),
        EXIT + (0, 2.5, "door"),
    ])
    # North wall: high clerestory windows above the back corridor
    body += wall("x", wy, -e, e, [
        (-7.0, -4.0, 1.9, 2.7, "window"),
        (-1.5, 1.5, 1.9, 2.7, "window"),
        (4.0, 7.0, 1.9, 2.7, "window"),
    ])
    # West/East walls: normal windows in the front zone, high windows above shelves
    side = [
        (-8.2, -6.2, 0.9, 2.3, "window"),
        (-5.4, -3.6, 0.9, 2.3, "window"),
        (-2.6, -0.6, 2.0, 2.7, "window"),
        (0.8, 2.8, 2.0, 2.7, "window"),
        (4.0, 6.0, 2.0, 2.7, "window"),
    ]
    body += wall("y", -wx, -STORE_Y, STORE_Y, side)
    body += wall("y", wx, -STORE_Y, STORE_Y, side)
    # "Staff only" back door (closed, visual only) in the north wall
    body.append(visual_box(8.0, STORE_Y - 0.01, 1.05, 1.0, 0.03, 2.1, (0.45, 0.30, 0.18), "staff_door"))
    return static_model("store_walls", 0, 0, body, "Outer walls with windows and door openings (NO ROOF)")


def gate(name, x0, x1, sign_rgb):
    """Automatic sliding doors (shown open) + anti-theft (EAS) pedestals + sign."""
    wy = -(STORE_Y + WALL_T / 2)
    w = x1 - x0
    body = []
    # glass door leaves slid open, sitting just outside the wall
    for xc in (x0 - w / 4 + 0.05, x1 + w / 4 - 0.05):
        body.append(visual_box(xc, wy - 0.16, 1.25, w / 2, 0.04, 2.45, GLASS_RGB, "door_glass", alpha=0.35, shadows=False, spec=0.9))
        body.append(visual_box(xc, wy - 0.16, 2.49, w / 2, 0.06, 0.04, (0.3, 0.3, 0.32), "door_rail"))
    # door frame posts
    for xp in (x0 + 0.05, x1 - 0.05):
        body.append(visual_box(xp, wy, 1.25, 0.1, WALL_T + 0.04, 2.5, (0.30, 0.30, 0.32), "frame"))
    # EAS security pedestals forming 2 walking lanes (each ~1.4 m wide)
    for xp in (x0 + 0.15, (x0 + x1) / 2, x1 - 0.15):
        body.append(visual_box(xp, -STORE_Y + 0.55, 0.75, 0.08, 0.5, 1.5, (0.85, 0.88, 0.92), "eas", alpha=0.7))
        body.append(collision_box(xp, -STORE_Y + 0.55, 0.75, 0.08, 0.5, 1.5))
        body.append(visual_box(xp, -STORE_Y + 0.55, 0.03, 0.2, 0.6, 0.06, (0.2, 0.2, 0.22), "eas_base"))
    # illuminated sign above the doorway (green = entrance, red = exit)
    body.append(visual_box((x0 + x1) / 2, -STORE_Y + 0.02, 2.75, w * 0.6, 0.05, 0.35, sign_rgb, "sign",
                           emissive=tuple(c * 0.8 for c in sign_rgb)))
    # entrance floor mat
    body.append(visual_box((x0 + x1) / 2, -STORE_Y + 0.9, 0.003, w * 0.8, 0.9, 0.006, (0.22, 0.27, 0.35), "mat", shadows=False))
    return static_model(name, 0, 0, body, f"{name}: sliding doors + EAS anti-theft gates")


# ═══════════════════════════════ CHECKOUTS ════════════════════════════════
CHECKOUT_X = [1.6, 4.0, 6.4]
CHECKOUT_Y = -6.0


def checkout(i, cx, cy):
    L, W, H = 1.8, 0.7, 0.9
    body = [
        collision_box(0, 0, H / 2, W, L, H),
        visual_box(0, 0, H / 2, W, L, H, (0.85, 0.83, 0.78), "counter"),
        visual_box(0, 0, 0.05, W + 0.02, L + 0.02, 0.1, (0.25, 0.25, 0.28), "kick"),
        # conveyor belt (customer side, north half)
        visual_box(0, 0.35, H + 0.01, W - 0.2, L / 2 + 0.1, 0.02, (0.08, 0.08, 0.08), "belt"),
        # scanner window
        visual_box(0, -0.15, H + 0.01, W - 0.3, 0.25, 0.02, (0.2, 0.5, 0.3), "scanner"),
        # cash register + monitor facing the cashier (east side)
        visual_box(0.18, -0.55, H + 0.08, 0.3, 0.35, 0.16, (0.15, 0.15, 0.17), "register"),
        visual_box(0.18, -0.55, H + 0.32, 0.04, 0.3, 0.22, (0.1, 0.1, 0.12), "monitor"),
        # bagging area (south end)
        visual_box(0, -L / 2 + 0.05, H + 0.1, W, 0.1, 0.2, (0.7, 0.7, 0.72), "bag_rail"),
        # cashier stool (east of the counter)
        visual_box(0.7, -0.45, 0.33, 0.35, 0.35, 0.05, (0.2, 0.3, 0.6), "stool"),
        collision_box(0.7, -0.45, 0.3, 0.35, 0.35, 0.6),
        visual_cyl(0.7, -0.45, 0.16, 0.03, 0.3, (0.4, 0.4, 0.42), "stool_leg"),
        # lane light pole with numbered colour light on top
        visual_cyl(-0.3, 0.85, 0.95, 0.025, 1.9, (0.4, 0.4, 0.42), "pole"),
        visual_box(-0.3, 0.85, 1.95, 0.25, 0.25, 0.18, (0.1, 0.8, 0.2), "lane_light",
                   emissive=(0.05, 0.6, 0.1)),
    ]
    return static_model(f"checkout_{i}", cx, cy, body, f"Checkout counter {i} (cashier on east side)")


# ═══════════════════════════ DECOR / EXTRAS ═══════════════════════════════
def basket_stack():
    body = []
    for k in range(6):
        body.append(visual_box(0, 0, 0.12 + k * 0.06, 0.45, 0.32, 0.06, (0.8, 0.1, 0.1), "basket"))
    body.append(collision_box(0, 0, 0.25, 0.45, 0.32, 0.5))
    return static_model("basket_stack", -8.2, -8.2, body, "Stack of shopping baskets near entrance")


def promo_island():
    """Low promotional bin in the front promenade-west (fruit display)."""
    body = [collision_box(0, 0, 0.35, 1.2, 0.8, 0.7),
            visual_box(0, 0, 0.35, 1.2, 0.8, 0.7, (0.55, 0.38, 0.22), "crate")]
    colours = [(0.9, 0.1, 0.1), (0.95, 0.75, 0.1), (0.2, 0.65, 0.15), (0.98, 0.5, 0.05)]
    for ix in range(4):
        for iy in range(3):
            c = colours[(ix + iy) % 4]
            body.append(f'<visual name="{Counter.next("fruit")}">{pose(-0.42 + ix * 0.28, -0.25 + iy * 0.25, 0.75)}'
                        f"<geometry><sphere><radius>0.11</radius></sphere></geometry>{material(c, 1.0, 0.4)}"
                        f"<cast_shadows>false</cast_shadows></visual>")
    return static_model("produce_display", -7.4, -5.0, body, "Fruit display bin (static obstacle)")


def floor():
    body = [
        # single infinite collision plane for everything
        '<collision name="ground_col"><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>'
        "<surface><friction><ode><mu>0.8</mu><mu2>0.8</mu2></ode></friction></surface></collision>",
        # outdoor parking lot (asphalt) slightly below the store floor
        f'<visual name="parking">{pose(0, 0, -0.02)}<geometry><plane><normal>0 0 1</normal><size>60 60</size></plane></geometry>'
        f"{material((0.28, 0.29, 0.30), 1.0, 0.05)}</visual>",
        # sidewalk in front of the store
        visual_box(0, -STORE_Y - 2.2, -0.01, 2 * STORE_X + 6, 4.0, 0.02, (0.70, 0.69, 0.66), "sidewalk"),
        # store floor (light tiles)
        visual_box(0, 0, -0.004, 2 * STORE_X + 2 * WALL_T, 2 * STORE_Y + 2 * WALL_T, 0.01, (0.78, 0.77, 0.73), "store_floor", spec=0.15),
    ]
    # tile grid lines every 1.2 m  -> gives the floor a tiled look and helps
    # judge distances in the top-down view
    for k in range(-7, 8):
        body.append(visual_box(k * 1.2, 0, 0.0015, 0.012, 2 * STORE_Y, 0.002, (0.66, 0.65, 0.62), "tile", shadows=False))
        body.append(visual_box(0, k * 1.2, 0.0015, 2 * STORE_X, 0.012, 0.002, (0.66, 0.65, 0.62), "tile", shadows=False))
    # parking bay lines
    for k in range(-6, 7):
        body.append(visual_box(k * 2.6, -STORE_Y - 7.5, -0.008, 0.1, 5.0, 0.01, (0.95, 0.95, 0.95), "park_line", shadows=False))
    return static_model("ground", 0, 0, body, "Ground: parking lot, sidewalk, tiled store floor")


# ═════════════════════════════ PEDESTRIANS ════════════════════════════════
def actor(name, pts):
    """gz-sim <actor>: a skinned, animated person following a scripted path.
    NOTE: actors are visible to cameras / gpu_lidar but have NO physics
    collision -> perfect as 'dynamic obstacles' the robot must perceive & avoid."""
    loop = pts + [pts[0]]
    wps = []
    t = 0.0
    yaw_prev = None
    for i in range(len(loop) - 1):
        (x0, y0), (x1, y1) = loop[i], loop[i + 1]
        yaw = math.atan2(y1 - y0, x1 - x0)
        if yaw_prev is not None:
            # turn on the spot at the corner
            wps.append((t, x0, y0, yaw_prev))
            t += TURN_TIME
        wps.append((t, x0, y0, yaw))
        t += math.hypot(x1 - x0, y1 - y0) / WALK_SPEED
        yaw_prev = yaw
    # arrive back at the start, then turn to the initial heading to close the loop
    wps.append((t, loop[-1][0], loop[-1][1], yaw_prev))
    t += TURN_TIME
    wps.append((t, loop[0][0], loop[0][1], wps[0][3]))
    w_xml = "\n            ".join(
        f"<waypoint><time>{t_:.2f}</time>{pose(x, y, 1.0, yaw)}</waypoint>" for (t_, x, y, yaw) in wps)
    return f"""
    <!-- Pedestrian {name}: loop {pts} at {WALK_SPEED} m/s -->
    <actor name="{name}">
      <skin>
        <filename>{ACTOR_MESH}</filename>
        <scale>1.0</scale>
      </skin>
      <animation name="walking">
        <filename>{ACTOR_MESH}</filename>
        <interpolate_x>true</interpolate_x>
      </animation>
      <script>
        <loop>true</loop>
        <auto_start>true</auto_start>
        <trajectory id="0" type="walking" tension="0">
            {w_xml}
        </trajectory>
      </script>
    </actor>
"""


# ═══════════════════════════════ WORLD ════════════════════════════════════
def lights():
    out = ["""
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 20 0 0 0</pose>
      <diffuse>0.75 0.75 0.72 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.35 0.25 -0.9</direction>
    </light>"""]
    # 4 overhead store lights (no shadows -> cheap on integrated GPUs)
    for i, (x, y) in enumerate([(-4.5, -1.2), (4.5, -1.2), (-4.5, 4.8), (4.5, 4.8)]):
        out.append(f"""
    <light type="point" name="store_light_{i + 1}">
      {pose(x, y, 2.9)}
      <diffuse>0.35 0.35 0.33 1</diffuse>
      <specular>0.05 0.05 0.05 1</specular>
      <attenuation><range>14</range><constant>0.6</constant><linear>0.05</linear><quadratic>0.005</quadratic></attenuation>
      <cast_shadows>false</cast_shadows>
    </light>""")
    return "".join(out)


GUI = """
    <!-- GUI: same panels as the gz-sim default, but the camera starts
         TOP-DOWN above the open-roof store for a clear presentation view. -->
    <gui fullscreen="0">
      <plugin filename="MinimalScene" name="3D View">
        <gz-gui><title>3D View</title><property type="bool" key="showTitleBar">false</property><property type="string" key="state">docked</property></gz-gui>
        <engine>ogre2</engine>
        <scene>scene</scene>
        <ambient_light>0.4 0.4 0.4</ambient_light>
        <background_color>0.62 0.78 0.92</background_color>
        <camera_pose>0 -14 24 0 1.05 1.5708</camera_pose>
        <camera_clip><near>0.25</near><far>200</far></camera_clip>
      </plugin>
      <plugin filename="EntityContextMenuPlugin" name="Entity context menu">
        <gz-gui><property key="state" type="string">floating</property><property key="width" type="double">5</property><property key="height" type="double">5</property><property key="showTitleBar" type="bool">false</property></gz-gui>
      </plugin>
      <plugin filename="GzSceneManager" name="Scene Manager">
        <gz-gui><property key="resizable" type="bool">false</property><property key="width" type="double">5</property><property key="height" type="double">5</property><property key="state" type="string">floating</property><property key="showTitleBar" type="bool">false</property></gz-gui>
      </plugin>
      <plugin filename="InteractiveViewControl" name="Interactive view control">
        <gz-gui><property key="resizable" type="bool">false</property><property key="width" type="double">5</property><property key="height" type="double">5</property><property key="state" type="string">floating</property><property key="showTitleBar" type="bool">false</property></gz-gui>
      </plugin>
      <plugin filename="CameraTracking" name="Camera Tracking">
        <gz-gui><property key="resizable" type="bool">false</property><property key="width" type="double">5</property><property key="height" type="double">5</property><property key="state" type="string">floating</property><property key="showTitleBar" type="bool">false</property></gz-gui>
      </plugin>
      <plugin filename="MarkerManager" name="Marker manager">
        <gz-gui><property key="resizable" type="bool">false</property><property key="width" type="double">5</property><property key="height" type="double">5</property><property key="state" type="string">floating</property><property key="showTitleBar" type="bool">false</property></gz-gui>
      </plugin>
      <plugin filename="SelectEntities" name="Select Entities">
        <gz-gui><property key="resizable" type="bool">false</property><property key="width" type="double">5</property><property key="height" type="double">5</property><property key="state" type="string">floating</property><property key="showTitleBar" type="bool">false</property></gz-gui>
      </plugin>
      <plugin filename="Spawn" name="Spawn Entities">
        <gz-gui><property key="resizable" type="bool">false</property><property key="width" type="double">5</property><property key="height" type="double">5</property><property key="state" type="string">floating</property><property key="showTitleBar" type="bool">false</property></gz-gui>
      </plugin>
      <plugin filename="VisualizationCapabilities" name="Visualization Capabilities">
        <gz-gui><property key="resizable" type="bool">false</property><property key="width" type="double">5</property><property key="height" type="double">5</property><property key="state" type="string">floating</property><property key="showTitleBar" type="bool">false</property></gz-gui>
      </plugin>
      <plugin filename="WorldControl" name="World control">
        <gz-gui><title>World control</title><property type="bool" key="showTitleBar">false</property><property type="bool" key="resizable">false</property><property type="double" key="height">72</property><property type="double" key="z">1</property><property type="string" key="state">floating</property>
          <anchors target="3D View"><line own="left" target="left"/><line own="bottom" target="bottom"/></anchors></gz-gui>
        <play_pause>true</play_pause><step>true</step><start_paused>false</start_paused><use_event>true</use_event>
      </plugin>
      <plugin filename="WorldStats" name="World stats">
        <gz-gui><title>World stats</title><property type="bool" key="showTitleBar">false</property><property type="bool" key="resizable">false</property><property type="double" key="height">110</property><property type="double" key="width">290</property><property type="double" key="z">1</property><property type="string" key="state">floating</property>
          <anchors target="3D View"><line own="right" target="right"/><line own="bottom" target="bottom"/></anchors></gz-gui>
        <sim_time>true</sim_time><real_time>true</real_time><real_time_factor>true</real_time_factor><iterations>true</iterations>
      </plugin>
      <plugin filename="Screenshot" name="Screenshot">
        <gz-gui><property key="resizable" type="bool">false</property><property key="x" type="double">0</property><property key="y" type="double">0</property><property key="width" type="double">50</property><property key="height" type="double">50</property><property key="state" type="string">floating</property><property key="showTitleBar" type="bool">false</property><property key="cardBackground" type="string">#777777</property></gz-gui>
      </plugin>
      <plugin filename="ComponentInspector" name="Component inspector">
        <gz-gui><property type="bool" key="showTitleBar">false</property><property type="string" key="state">docked</property></gz-gui>
      </plugin>
      <plugin filename="EntityTree" name="Entity tree">
        <gz-gui><property type="bool" key="showTitleBar">false</property><property type="string" key="state">docked</property></gz-gui>
      </plugin>
    </gui>
"""


def build_world(with_pedestrians=True):
    parts = []
    parts.append(floor())
    parts.append(walls())
    parts.append(gate("entrance_gate", *ENTRANCE, (0.1, 0.75, 0.2)))
    parts.append(gate("exit_gate", *EXIT, (0.85, 0.1, 0.1)))
    for i, x in enumerate(CHECKOUT_X):
        parts.append(checkout(i + 1, x, CHECKOUT_Y))
    for row, cy in ROW_Y.items():
        for i, cx in enumerate(GONDOLA_X):
            parts.append(gondola(f"{row}{i + 1}", cx, cy))
    parts.append(wall_shelf("WALL_W", "E", +1))
    parts.append(wall_shelf("WALL_E", "W", -1))
    parts.append(basket_stack())
    parts.append(promo_island())
    pedestrians = PEDESTRIANS if with_pedestrians else {}
    for name, pts in pedestrians.items():
        parts.append(actor(name, pts))

    header = f"""<?xml version="1.0" ?>
<!--
  ======================================================================
  RETAIL STORE WORLD  (Gazebo Harmonic / gz-sim 8)
  AUTO-GENERATED by scripts/generate_retail_store_world.py - DO NOT EDIT
  BY HAND. Change the script and re-run it instead.
  ======================================================================
  * 10 shelves: 8 double-sided gondolas (rows S1-S4, N1-N4) + 2 wall shelves
  * 5 north-south aisles, 2.68 m wide (centres x = -7.16, -3.58, 0, 3.58, 7.16)
  * cross aisle at y = 0.6..3.0, back corridor y = 6.6..9, front zone y < -3
  * {len(void_records)} deliberate shelf voids -> see config/shelf_voids.yaml
  * {len(pedestrians)} slow pedestrians (actors) as dynamic obstacles
  * NO ROOF: top-down GUI camera shows the whole store
  * Safe robot spawn: (0, 0) facing north (aisle 3), or (-5, -6.5) near the entrance
  ======================================================================
-->
<sdf version="1.9">
  <world name="retail_store">

    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <!-- gz-sim system plugins (these replace Gazebo Classic's libgazebo_ros_*):
         Physics          = rigid-body dynamics (robot wheels, collisions)
         UserCommands     = lets ros_gz_sim 'create' spawn the robot
         SceneBroadcaster = sends the scene to the GUI
         Sensors          = renders camera / gpu_lidar sensors (ogre2) -->
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <scene>
      <ambient>0.45 0.45 0.45 1</ambient>
      <background>0.62 0.78 0.92 1</background>
      <shadows>true</shadows>
      <grid>false</grid>
    </scene>

    <gravity>0 0 -9.81</gravity>
{GUI}
{lights()}
"""
    return header + "".join(parts) + "\n  </world>\n</sdf>\n"


def write_voids_yaml():
    lines = [
        "# AUTO-GENERATED by scripts/generate_retail_store_world.py",
        "# Ground truth of every deliberately EMPTY shelf spot in retail_store.world.",
        "# Use it to (a) plan where the robot should look and (b) score your CV",
        "# void detector (precision / recall) during the demo.",
        "# Coordinates are in the Gazebo world frame (metres). level 0 = bottom deck.",
        "voids:",
    ]
    for v in void_records:
        lines.append(f"  - id: {v['id']}")
        for k in ("shelf", "face", "category", "aisle", "bay", "level", "x_range", "y_range", "z_range", "centre"):
            lines.append(f"    {k}: {v[k]}")
    with open(VOIDS_OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sdf = build_world()
    with open(WORLD_OUT, "w") as fh:
        fh.write(sdf)
    write_voids_yaml()
    # Reset all global generator state so the mapping world gets the exact same
    # products / voids as the main world (only the pedestrians differ).
    rng.seed(42)
    Counter.n = 0
    void_records.clear()
    mapping_sdf = build_world(with_pedestrians=False)
    with open(MAPPING_WORLD_OUT, "w") as fh:
        fh.write(mapping_sdf)
    print(f"Wrote {MAPPING_WORLD_OUT}  (no pedestrians, for SLAM mapping)")
    n_vis = sdf.count("<visual ")
    print(f"Wrote {WORLD_OUT}  ({n_vis} visuals, {sdf.count('<collision ')} collisions, "
          f"{sdf.count('<actor ')} actors)")
    print(f"Wrote {VOIDS_OUT}  ({len(void_records)} voids)")
