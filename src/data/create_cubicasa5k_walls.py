"""
create_cubicasa5k_walls.py

Extract wall segmentation masks from CubicasA5k SVG floor plans.

Expected layout (script sits alongside the dataset dirs):
    <root>/
        create_cubicasa5k_walls.py
        colorful/                   <-- input
            5617/
                F1_scaled.png
                model.svg
            ...
        high_quality_architectural/ <-- input
        high_quality/               <-- input
        cubicasa5k/                 <-- output (created automatically)
            colorful/
                images/
                masks/
                    walls/
                    walls_check/
            high_quality_architectural/
            high_quality/

Usage:
    python create_cubicasa5k_walls.py                          # all 3 dirs
    python create_cubicasa5k_walls.py --dirname colorful       # full name
    python create_cubicasa5k_walls.py --dirname c              # abbreviation
    python create_cubicasa5k_walls.py --dirname hqa
    python create_cubicasa5k_walls.py --dirname hq
"""

import argparse
import collections
import io
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from PIL import Image
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

try:
    import cairosvg as _cairosvg
    _HAS_CAIROSVG = True
except (ImportError, OSError):
    _HAS_CAIROSVG = False



# ── Directory / prefix mappings ───────────────────────────────────────────────

DIRS = {
    'colorful':                   'c',
    'high_quality_architectural':  'hqa',
    'high_quality':                'hq',
}

# All valid keys a user may pass to --dirname (full name or abbreviation)
_ABBREV_TO_FULL = {v: k for k, v in DIRS.items()}
_ABBREV_TO_FULL.update({k: k for k in DIRS})   # full names also accepted


def resolve_dirname(name: str) -> str:
    """Return the canonical directory name for a full name or abbreviation."""
    if name in _ABBREV_TO_FULL:
        return _ABBREV_TO_FULL[name]
    valid = list(DIRS.keys()) + list(DIRS.values())
    raise ValueError(
        f"Unknown --dirname {name!r}. "
        f"Valid values: {', '.join(sorted(set(valid)))}"
    )


# ── Core extraction logic ─────────────────────────────────────────────────────

def _parse_polygon(elem, sx, sy, offset):
    """Parse an SVG <polygon> element into scaled pixel coordinates."""
    points_str = elem.get('points', '').strip()
    if not points_str:
        return None
    pts = []
    for token in points_str.split():
        if ',' in token:
            try:
                x, y = token.split(',', 1)
                pts.append((float(x) * sx + offset, float(y) * sy + offset))
            except ValueError:
                continue
    return pts if len(pts) >= 3 else None


def _to_polygon(pts):
    """Return a valid ShapelyPolygon, repairing self-intersections via buffer(0)."""
    poly = ShapelyPolygon(pts)
    return poly if poly.is_valid else poly.buffer(0)


def _build_geometry(root, sx, sy, offset):
    """
    Walk the SVG tree and collect:
      wall_shapes   – polygons to fill white (Wall, Column)
      cutout_shapes – polygons to subtract black (Threshold = door, Glass = window)
    """
    wall_shapes, cutout_shapes = [], []

    balcony_shapes = []

    for elem in root.iter():
        eid        = elem.get('id', '')
        cls_tokens = elem.get('class', '').split()

        # SVG element hierarchy:
        #   Wall   -> direct <polygon>  ← wall material
        #   Column -> direct <polygon>  ← structural column (below floor plan)
        #   <any class containing "Balcony"> -> direct <polygon>  ← balcony border
        #     Door -> Threshold -> <polygon>  ← door opening (cut)
        #     Window -> Glass  -> <polygon>   ← window opening (cut)
        if any('Balcony' in t or 'Porch' in t for t in cls_tokens):
            for child in elem:
                if child.tag.split('}')[-1] == 'polygon':
                    pts = _parse_polygon(child, sx, sy, offset)
                    if pts:
                        balcony_shapes.append(_to_polygon(pts))

        elif eid in ('Wall', 'Column'):
            for child in elem:
                if child.tag.split('}')[-1] == 'polygon':
                    pts = _parse_polygon(child, sx, sy, offset)
                    if pts:
                        wall_shapes.append(_to_polygon(pts))

        elif eid in ('Threshold', 'Glass'):
            for child in elem:
                if child.tag.split('}')[-1] == 'polygon':
                    pts = _parse_polygon(child, sx, sy, offset)
                    if pts:
                        cutout_shapes.append(_to_polygon(pts))

    return wall_shapes, cutout_shapes, balcony_shapes


def _geometry_to_patch(geom, color, border_only=False, linewidth=0):
    """Convert a shapely (Multi)Polygon to a matplotlib PathPatch."""
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        polys = [g for g in geom.geoms
                 if isinstance(g, ShapelyPolygon) and not g.is_empty]
    elif isinstance(geom, ShapelyPolygon) and not geom.is_empty:
        polys = [geom]
    else:
        polys = []
    verts, codes = [], []
    for poly in polys:
        exterior = poly.exterior
        if exterior is None:
            continue
        for ring in [exterior] + list(poly.interiors):
            coords = np.array(ring.coords)
            if len(coords) < 2:
                continue
            verts.append(coords[0]);   codes.append(Path.MOVETO)
            for pt in coords[1:-1]:
                verts.append(pt);      codes.append(Path.LINETO)
            verts.append(coords[-1]);  codes.append(Path.CLOSEPOLY)
    if not verts:
        return mpatches.PathPatch(
            Path(np.zeros((1, 2)), [Path.MOVETO]),
            facecolor='none', edgecolor='none', linewidth=0,
        )
    return mpatches.PathPatch(
        Path(np.array(verts), codes),
        facecolor='none' if border_only else color,
        edgecolor=color,
        linewidth=linewidth,
    )


def extract_walls(svg_path: str, reference_png_path: str):
    """
    Extract wall mask from *svg_path* sized to match *reference_png_path*.

    Returns
    -------
    walls_arr      : np.ndarray  (H, W, 3) uint8 – white walls on black
    walls_check_arr: np.ndarray  (H, W, 3) uint8 – red overlay on reference
    Both are None if no wall polygons were found.
    """
    ref = Image.open(reference_png_path).convert('RGB')
    target_w, target_h = ref.size

    tree = ET.parse(svg_path)
    root = tree.getroot()

    # F1_scaled.png is a 2.79× upscale of a 263×237 render where
    # 1 SVG unit ≈ 1 pixel.  The upscaling anti-aliasing shifts every
    # visible wall edge 1 px inward, so we apply a −1 px alignment offset.
    sx, sy = 1.0, 1.0
    OFFSET = -1.0

    wall_shapes, cutout_shapes, balcony_shapes = _build_geometry(root, sx, sy, OFFSET)
    if not wall_shapes and not balcony_shapes:
        return None, None, None

    wall_union    = unary_union(wall_shapes)    if wall_shapes    else None
    balcony_union = unary_union(balcony_shapes) if balcony_shapes else None

    if wall_union and cutout_shapes:
        wall_union = wall_union.difference(unary_union(cutout_shapes))

    # ── Crop box: bbox of all geometry + margin, clamped to image ────────
    MARGIN = 50
    bbox_geoms = [g for g in [wall_union, balcony_union] if g is not None]
    minx, miny, maxx, maxy = unary_union(bbox_geoms).bounds
    cx0 = max(0,        int(minx)     - MARGIN)
    cy0 = max(0,        int(miny)     - MARGIN)
    cx1 = min(target_w, int(maxx) + 1 + MARGIN)
    cy1 = min(target_h, int(maxy) + 1 + MARGIN)
    crop_box = (cx0, cy0, cx1, cy1)

    # ── Render to in-memory PNG ───────────────────────────────────────────
    dpi = 100
    fig, ax = plt.subplots(figsize=(target_w / dpi, target_h / dpi), dpi=dpi)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    if wall_union:
        ax.add_patch(_geometry_to_patch(wall_union, 'white'))
    if balcony_union:
        ax.add_patch(_geometry_to_patch(balcony_union, 'white', border_only=True, linewidth=3))
    ax.set_xlim(0, target_w)
    ax.set_ylim(target_h, 0)   # invert Y: SVG origin is top-left
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='black',
                edgecolor='none', pad_inches=0, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    walls_img = Image.open(buf).convert('RGB')
    if walls_img.size != (target_w, target_h):
        walls_img = walls_img.resize((target_w, target_h), Image.NEAREST)
    walls_arr = np.array(walls_img)

    # ── Alignment-check overlay (walls in red on reference) ───────────────
    check_arr = _make_check_overlay(np.array(ref), walls_arr)

    return walls_arr, check_arr, crop_box


def _make_check_overlay(base_arr, walls_arr):
    """Overlay the wall mask (white pixels) in red on a base RGB array."""
    wall_mask = walls_arr.mean(axis=2) > 128
    check     = base_arr.astype(np.float32)
    check[wall_mask, 0]  = 255
    check[wall_mask, 1] *= 0.25
    check[wall_mask, 2] *= 0.25
    return check.astype(np.uint8)


# ── Dry-run stats collection ──────────────────────────────────────────────────

_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE,
)


def _collect_svg_id_stats(svg_path: str):
    """Return (id_counts, cls_counts) dicts, skipping UUID-like values."""
    id_counts, cls_counts = {}, {}
    for elem in ET.parse(svg_path).getroot().iter():
        eid = elem.get('id', '')
        if eid and not _UUID_RE.match(eid):
            id_counts[eid] = id_counts.get(eid, 0) + 1
        for token in elem.get('class', '').split():
            if not _UUID_RE.match(token):
                cls_counts[token] = cls_counts.get(token, 0) + 1
    return id_counts, cls_counts


def _print_stats_table(label: str, total: dict, samples: dict) -> None:
    print(f"\n  -- {label} --")
    if not total:
        print("  (none found)")
        return
    w = max(len(k) for k in total)
    w = max(w, len(label)) + 2
    print(f"  {label:<{w}} {'samples':>8}   {'total occurrences':>18}")
    print(f"  {'-' * w} {'-' * 8}   {'-' * 18}")
    for k in sorted(total, key=lambda x: -total[x]):
        print(f"  {k:<{w}} {samples[k]:>8}   {total[k]:>18}")


def _print_dry_run_stats(n_scanned: int, n_skipped: int,
                         id_total: dict, id_samples: dict,
                         cls_total: dict, cls_samples: dict) -> None:
    print(f"  Samples scanned : {n_scanned}  |  skipped: {n_skipped}")
    _print_stats_table('IDs',     id_total,  id_samples)
    _print_stats_table('classes', cls_total, cls_samples)


# ── SVG → PIL renderer ───────────────────────────────────────────────────────

def _render_svg(svg_path: str, target_w: int, target_h: int) -> Image.Image:
    """
    Render an SVG to a PIL RGB Image (white background) using the SAME
    coordinate mapping as the wall mask: 1 SVG unit = 1 pixel, placed at the
    SVG origin and clipped/padded to a *target_w* x *target_h* canvas.

    Rendering at the viewBox's intrinsic scale (scale=1) instead of letting
    cairosvg rescale the viewBox to the target size is what keeps the model
    image aligned with walls_arr (matplotlib uses an identity svg->pixel map).
    """
    if not _HAS_CAIROSVG:
        raise ImportError(
            "cairosvg is required to render model SVG images. "
            "Install it with: conda install -c conda-forge cairosvg"
        )
    from pathlib import Path as _Path
    raw  = _cairosvg.svg2png(
        url=_Path(os.path.abspath(svg_path)).as_uri(),
        scale=1.0,
    )
    rgba   = Image.open(io.BytesIO(raw)).convert('RGBA')
    canvas = Image.new('RGB', (target_w, target_h), (255, 255, 255))
    canvas.paste(rgba, (0, 0), mask=rgba.split()[3])
    return canvas


# ── Output directory helpers ──────────────────────────────────────────────────

def _output_paths(out_root: str, dirname: str) -> dict:
    """Create the output subdirectory tree and return a dict of leaf paths."""
    base = os.path.join(out_root, dirname)
    paths = {
        'images':      os.path.join(base, 'images'),
        'walls':       os.path.join(base, 'masks', 'walls'),
        'walls_check': os.path.join(base, 'masks', 'walls_check'),
        'model':       os.path.join(base, 'masks', 'model'),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


# ── Per-sample worker (top-level so it is picklable) ─────────────────────────

def _process_sample(svg_path, img_path, out_paths, prefix, sample_id):
    """Process one sample. Runs in a worker process."""
    fname       = f"cubicasa_{prefix}_{sample_id:05d}.png"
    fname_svg   = f"cubicasa_{prefix}_{sample_id:05d}.svg"
    fname_model = f"cubicasa_{prefix}_{sample_id:05d}_model.png"

    with Image.open(img_path) as _ref:
        target_w, target_h = _ref.size

    walls_arr, check_arr, crop_box = extract_walls(svg_path, img_path)
    if walls_arr is None:
        return

    cx0, cy0, cx1, cy1 = crop_box
    walls_cropped = Image.fromarray(walls_arr[cy0:cy1, cx0:cx1])
    check_full    = Image.fromarray(check_arr)

    Image.open(img_path).crop(crop_box).save(
        os.path.join(out_paths['images'], fname))
    walls_cropped.save(os.path.join(out_paths['walls'],       fname))
    check_full.save(   os.path.join(out_paths['walls_check'], fname))
    shutil.copy2(svg_path, os.path.join(out_paths['model'],   fname_svg))

    try:
        model_img  = _render_svg(svg_path, target_w, target_h)
        model_check = _make_check_overlay(np.array(model_img), walls_arr)
        model_img.crop(crop_box).save(
            os.path.join(out_paths['images'],     fname_model))
        walls_cropped.save(os.path.join(out_paths['walls'],       fname_model))
        Image.fromarray(model_check).save(
            os.path.join(out_paths['walls_check'], fname_model))
    except Exception:
        pass


# ── Per-directory processing ──────────────────────────────────────────────────

def process_dirname(input_root_dir: str, output_root_dir: str, dirname: str,
                    dry_run: bool = False, sample: str = None,
                    workers: int = None) -> None:
    prefix     = DIRS[dirname]
    input_dir  = os.path.join(input_root_dir, dirname)
    out_root   = os.path.join(output_root_dir, 'cubicasa5k')

    if not os.path.isdir(input_dir):
        print(f"  [SKIP] input dir not found: {input_dir}")
        return

    subdirs = sorted(
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d)) and d.isdigit()
    )
    if sample is not None:
        subdirs = [d for d in subdirs if d == sample]
    if not subdirs:
        print(f"  [SKIP] no numeric subdirs in {input_dir}")
        return

    n_workers = workers or os.cpu_count() or 1

    if dry_run:
        # ── Build task list ───────────────────────────────────────────────
        svg_tasks, n_skipped = [], 0
        for subdir in subdirs:
            subdir_path = os.path.join(input_dir, subdir)
            svg_path    = os.path.join(subdir_path, 'model.svg')
            img_path    = os.path.join(subdir_path, 'F1_scaled.png')
            if os.path.exists(svg_path) and os.path.exists(img_path):
                svg_tasks.append((subdir, svg_path))
            else:
                n_skipped += 1

        id_total, id_samples   = collections.Counter(), collections.Counter()
        cls_total, cls_samples = collections.Counter(), collections.Counter()
        n_scanned = 0

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            future_to_sub = {
                pool.submit(_collect_svg_id_stats, svg_path): subdir
                for subdir, svg_path in svg_tasks
            }
            with tqdm(total=len(future_to_sub), desc=dirname,
                      unit='sample', leave=True) as pbar:
                for future in as_completed(future_to_sub):
                    subdir = future_to_sub[future]
                    try:
                        id_counts, cls_counts = future.result()
                        for k, v in id_counts.items():
                            id_total[k] += v;  id_samples[k] += 1
                        for k, v in cls_counts.items():
                            cls_total[k] += v; cls_samples[k] += 1
                        n_scanned += 1
                    except Exception as exc:
                        pbar.write(f"  [ERROR] {subdir}: {exc}")
                        n_skipped += 1
                    pbar.update(1)

        _print_dry_run_stats(n_scanned, n_skipped,
                             dict(id_total), dict(id_samples),
                             dict(cls_total), dict(cls_samples))
        return

    out_paths = _output_paths(out_root, dirname)

    # ── Build task list ───────────────────────────────────────────────────
    tasks = []
    for subdir in subdirs:
        sample_id   = int(subdir)
        subdir_path = os.path.join(input_dir, subdir)
        svg_path    = os.path.join(subdir_path, 'model.svg')
        img_path    = os.path.join(subdir_path, 'F1_scaled.png')
        if not os.path.exists(svg_path) or not os.path.exists(img_path):
            tqdm.write(f"  [SKIP] {subdir}: missing model.svg or F1_scaled.png")
            continue
        tasks.append((subdir, svg_path, img_path, sample_id))

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        future_to_sub = {
            pool.submit(_process_sample, svg, img, out_paths, prefix, sid): sub
            for sub, svg, img, sid in tasks
        }
        with tqdm(total=len(future_to_sub), desc=dirname,
                  unit='sample', leave=True) as pbar:
            for future in as_completed(future_to_sub):
                subdir = future_to_sub[future]
                try:
                    future.result()
                except Exception as exc:
                    pbar.write(f"  {subdir}: ERROR: {exc}")
                pbar.update(1)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract wall segmentation masks from CubicasA5k floor plans.'
    )
    parser.add_argument(
        '--dirname', '-d',
        default=None,
        metavar='NAME',
        help=(
            'Dataset subdirectory to process. '
            'Accepts full name or abbreviation: '
            'colorful/c, high_quality_architectural/hqa, high_quality/hq. '
            'Omit to process all three.'
        ),
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help=(
            'Do not create any output files or directories. '
            'Instead, scan every SVG and print a table of distinct element IDs '
            'with their sample count and total occurrence count per dataset dir.'
        ),
    )
    parser.add_argument(
        '--sample', '-s',
        default=None,
        metavar='ID',
        help=(
            'Process only the sample subdirectory with this numeric ID (e.g. 10000). '
            'Applies to every dirname that is processed.'
        ),
    )
    parser.add_argument(
        '--dry-path', '-p',
        default=None,
        metavar='PATH',
        help=(
            'Restrict dry-run stats to a single sample directory '
            '(must contain model.svg). Implies --dry-run; --dirname is ignored.'
        ),
    )
    parser.add_argument(
        '--input-root', '-i',
        default=None,
        metavar='DIR',
        help=(
            'Root directory holding the dataset dirs (colorful/, high_quality/, ...). '
            'Defaults to the directory where this script resides.'
        ),
    )
    parser.add_argument(
        '--output-root', '-o',
        default=None,
        metavar='DIR',
        help=(
            "Root directory under which the 'cubicasa5k/' output tree is created. "
            "Defaults to ../../data/raw relative to this script."
        ),
    )
    args = parser.parse_args()

    script_dir      = os.path.dirname(os.path.abspath(__file__))
    input_root_dir  = os.path.abspath(args.input_root) if args.input_root \
        else script_dir
    output_root_dir = os.path.abspath(args.output_root) if args.output_root \
        else os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'raw'))

    # ── Single-path dry run ───────────────────────────────────────────────────
    if args.dry_path:
        dry_path = args.dry_path
        if not os.path.isabs(dry_path):
            dry_path = os.path.join(input_root_dir, dry_path)
        svg_path = os.path.join(dry_path, 'model.svg')
        if not os.path.exists(svg_path):
            parser.error(f"model.svg not found in: {dry_path}")
        print(f"[DRY RUN] {dry_path}\n")
        try:
            id_counts, cls_counts = _collect_svg_id_stats(svg_path)
            _print_dry_run_stats(
                1, 0,
                id_counts,  {k: 1 for k in id_counts},
                cls_counts, {k: 1 for k in cls_counts},
            )
        except Exception as exc:
            print(f"ERROR: {exc}")
        print("\nDone.")
        return

    # ── Full-dataset dry run or normal run ────────────────────────────────────
    if args.dirname is None:
        dirnames = list(DIRS.keys())
    else:
        try:
            dirnames = [resolve_dirname(args.dirname)]
        except ValueError as e:
            parser.error(str(e))

    if args.dry_run:
        print("[DRY RUN] No files will be written.\n")

    for dirname in dirnames:
        print(f"\n=== {dirname} ({DIRS[dirname]}) ===")
        process_dirname(input_root_dir, output_root_dir, dirname,
                        dry_run=args.dry_run, sample=args.sample)

    print("\nDone.")


if __name__ == '__main__':
    main()
