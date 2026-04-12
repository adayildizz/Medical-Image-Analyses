"""
COMP 448/548 - Medical Image Analysis
Homework #1 - Full Solution
Parts 1-5
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import ndimage
from skimage import measure, morphology, segmentation
from skimage.morphology import (
    binary_dilation, binary_erosion, binary_opening, binary_closing,
    disk, skeletonize
)
from skimage.measure import label, regionprops
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG 
# ─────────────────────────────────────────────
DATA_ROOT = Path("/Users/adayildiz/PROJECTS/comp448_project1/data")
OUT_ROOT  = Path("/Users/adayildiz/PROJECTS/comp448_project1/results")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "cell-detection",
    "c-elegans-worm-segmentation",
    "corneal-nerve-segmentation",
    "gland-segmentation",
    "liver-segmentation",
    "oct-fluid-segmentation",
]

# Actual file prefix for each dataset folder
DATASET_PREFIX = {
    "cell-detection": "BRCA",
    "c-elegans-worm-segmentation": "Elegans",
    "corneal-nerve-segmentation": "CORN",
    "gland-segmentation": "Gland",
    "liver-segmentation": "liver",
    "oct-fluid-segmentation": "oct",
}

# ─────────────────────────────────────────────
# UTILITIES: load files
# ─────────────────────────────────────────────
def load_dataset(ds_name):
    """Load all prob + gold pairs for a dataset."""
    from PIL import Image
    ds_path = DATA_ROOT / ds_name
    prefix = DATASET_PREFIX.get(ds_name, ds_name)
    prob_files = sorted(ds_path.glob(f"{prefix}_prob_*.npy"))
    pairs = []
    for pf in prob_files:
        stem = pf.stem  # e.g. BRCA_prob_1
        suffix = stem.split("_prob_")[-1]  # e.g. "1"
        candidates = list(ds_path.glob(f"{prefix}_gold_{suffix}.*"))
        if not candidates:
            print(f"  [WARN] No gold for {pf.name}, skipping")
            continue
        gold_file = candidates[0]
        prob = np.load(str(pf)).astype(np.float32)
        gold = np.array(Image.open(str(gold_file)))
        pairs.append((suffix, prob, gold))
    return pairs

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def dice_score(pred_bin, gold_bin):
    """Overlap-based: Dice coefficient."""
    inter = np.logical_and(pred_bin, gold_bin).sum()
    denom = pred_bin.sum() + gold_bin.sum()
    return 2 * inter / denom if denom > 0 else 1.0

def iou_score(pred_bin, gold_bin):
    """Overlap-based: Intersection over Union (Jaccard)."""
    inter = np.logical_and(pred_bin, gold_bin).sum()
    union = np.logical_or(pred_bin, gold_bin).sum()
    return inter / union if union > 0 else 1.0

def pixel_accuracy(pred_bin, gold_bin):
    """Overlap-based: pixel accuracy."""
    return np.mean(pred_bin == gold_bin)

# ── Match-based ──────────────────────────────

def object_f1(pred_labeled, gold_labeled, iou_threshold=0.5):
    """
    Match-based: object-level F1.
    Each predicted object is matched to a gold object if IoU >= threshold.
    """
    pred_ids = np.unique(pred_labeled); pred_ids = pred_ids[pred_ids > 0]
    gold_ids = np.unique(gold_labeled); gold_ids = gold_ids[gold_ids > 0]
    tp = 0
    matched_gold = set()
    for pid in pred_ids:
        pmask = pred_labeled == pid
        best_iou = 0; best_gid = None
        for gid in gold_ids:
            if gid in matched_gold:
                continue
            gmask = gold_labeled == gid
            i = np.logical_and(pmask, gmask).sum()
            u = np.logical_or(pmask, gmask).sum()
            iou = i / u if u > 0 else 0
            if iou > best_iou:
                best_iou = iou; best_gid = gid
        if best_iou >= iou_threshold and best_gid is not None:
            tp += 1
            matched_gold.add(best_gid)
    fp = len(pred_ids) - tp
    fn = len(gold_ids) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

def panoptic_quality(pred_labeled, gold_labeled, iou_threshold=0.5):
    """
    Match-based (extra, not in slides): Panoptic Quality.
    PQ = SQ * RQ where SQ = avg IoU of matched pairs, RQ = F1 of matching.
    """
    pred_ids = np.unique(pred_labeled); pred_ids = pred_ids[pred_ids > 0]
    gold_ids = np.unique(gold_labeled); gold_ids = gold_ids[gold_ids > 0]
    tp = 0; sum_iou = 0
    matched_gold = set()
    for pid in pred_ids:
        pmask = pred_labeled == pid
        best_iou = 0; best_gid = None
        for gid in gold_ids:
            if gid in matched_gold:
                continue
            gmask = gold_labeled == gid
            i = np.logical_and(pmask, gmask).sum()
            u = np.logical_or(pmask, gmask).sum()
            iou = i / u if u > 0 else 0
            if iou > best_iou:
                best_iou = iou; best_gid = gid
        if best_iou >= iou_threshold and best_gid is not None:
            tp += 1; sum_iou += best_iou
            matched_gold.add(best_gid)
    fp = len(pred_ids) - tp
    fn = len(gold_ids) - tp
    sq = sum_iou / tp if tp > 0 else 0
    rq = tp / (tp + 0.5*fp + 0.5*fn) if (tp + 0.5*fp + 0.5*fn) > 0 else 0
    pq = sq * rq
    return pq

# ── Distance-based ───────────────────────────

def hausdorff_distance(pred_bin, gold_bin):
    """Distance-based: Hausdorff distance (95th percentile)."""
    from scipy.ndimage import distance_transform_edt
    pred_bin = pred_bin.astype(bool)
    gold_bin = gold_bin.astype(bool)
    if not pred_bin.any() or not gold_bin.any():
        return np.nan
    d_pred = distance_transform_edt(~pred_bin)
    d_gold = distance_transform_edt(~gold_bin)
    h1 = d_gold[pred_bin]
    h2 = d_pred[gold_bin]
    return max(np.percentile(h1, 95), np.percentile(h2, 95))

def average_surface_distance(pred_bin, gold_bin):
    """
    Distance-based (extra, not in slides): Average Symmetric Surface Distance (ASSD).
    Mean of distances from pred surface to gold surface and vice versa.
    """
    from scipy.ndimage import distance_transform_edt
    pred_bin = pred_bin.astype(bool)
    gold_bin = gold_bin.astype(bool)
    if not pred_bin.any() or not gold_bin.any():
        return np.nan
    pred_border = pred_bin ^ binary_erosion(pred_bin)
    gold_border = gold_bin ^ binary_erosion(gold_bin)
    d_pred = distance_transform_edt(~pred_bin)
    d_gold = distance_transform_edt(~gold_bin)
    d1 = d_gold[pred_border].mean() if pred_border.any() else 0
    d2 = d_pred[gold_border].mean() if gold_border.any() else 0
    return (d1 + d2) / 2

# ── Topology / Structure-based ───────────────

def clDice(pred_bin, gold_bin):
    """
    Topology-based: centerline Dice (clDice).
    Compares skeletons — important for thin structures like vessels/nerves.
    """
    pred_bin = pred_bin.astype(bool)
    gold_bin = gold_bin.astype(bool)
    if not pred_bin.any() or not gold_bin.any():
        return 0.0
    skel_pred = skeletonize(pred_bin)
    skel_gold = skeletonize(gold_bin)
    tprec = np.logical_and(skel_pred, gold_bin).sum() / skel_pred.sum() if skel_pred.sum() > 0 else 0
    tsens = np.logical_and(skel_gold, pred_bin).sum() / skel_gold.sum() if skel_gold.sum() > 0 else 0
    return 2*tprec*tsens/(tprec+tsens) if (tprec+tsens) > 0 else 0

def betti_error(pred_bin, gold_bin):
    """
    Topology-based (extra, not in slides): Betti number error.
    Counts difference in connected components (beta0) and holes (beta1).
    Lower is better (0 = same topology).
    """
    pred_bin = pred_bin.astype(bool)
    gold_bin = gold_bin.astype(bool)
    # beta0: number of connected components
    beta0_pred = label(pred_bin).max()
    beta0_gold = label(gold_bin).max()
    # beta1: number of holes (Euler characteristic based approximation)
    # euler_number = components - holes  =>  holes = components - euler
    euler_pred = measure.euler_number(pred_bin, connectivity=1)
    euler_gold = measure.euler_number(gold_bin, connectivity=1)
    beta1_pred = beta0_pred - euler_pred
    beta1_gold = beta0_gold - euler_gold
    return abs(beta0_pred - beta0_gold) + abs(beta1_pred - beta1_gold)

def compute_all_metrics(pred_bin, pred_labeled, gold_bin, gold_labeled):
    """Compute all metrics for a single image."""
    results = {}
    # Overlap
    results["dice"]     = dice_score(pred_bin, gold_bin)
    results["iou"]      = iou_score(pred_bin, gold_bin)
    # Match
    obj = object_f1(pred_labeled, gold_labeled)
    results["obj_f1"]   = obj["f1"]
    results["obj_prec"] = obj["precision"]
    results["obj_rec"]  = obj["recall"]
    results["pq"]       = panoptic_quality(pred_labeled, gold_labeled)
    # Distance
    results["hd95"]     = hausdorff_distance(pred_bin, gold_bin)
    results["assd"]     = average_surface_distance(pred_bin, gold_bin)
    # Topology
    results["cldice"]   = clDice(pred_bin, gold_bin)
    results["betti_err"]= betti_error(pred_bin, gold_bin)
    return results

# ─────────────────────────────────────────────
# PART 2: BASELINES
# ─────────────────────────────────────────────

def threshold_and_label(prob, threshold=0.5):
    """Threshold prob map → binary mask → labeled components."""
    binary = prob >= threshold
    labeled = label(binary)
    return binary, labeled

def remove_small_objects(binary, labeled, area_threshold):
    """Baseline 2: remove connected components smaller than area_threshold."""
    new_binary = binary.copy()
    for region in regionprops(labeled):
        if region.area < area_threshold:
            new_binary[labeled == region.label] = False
    new_labeled = label(new_binary)
    return new_binary, new_labeled

def run_part2(pairs, ds_name, prob_threshold=0.5, area_thresholds=None):
    """Run baselines for one dataset."""
    if area_thresholds is None:
        area_thresholds = [0, 5, 10, 20, 50, 100, 200, 500]

    print(f"\n{'='*60}")
    print(f"PART 2: {ds_name}")
    print(f"{'='*60}")

    ds_out = OUT_ROOT / "part2" / ds_name
    ds_out.mkdir(parents=True, exist_ok=True)

    for suffix, prob, gold in pairs:
        gold_bin    = gold > 0
        gold_labeled = gold.astype(int)  # gold already has integer labels

        print(f"\n  Image: {suffix}")

        # Baseline 1: no postprocessing
        b1_bin, b1_lab = threshold_and_label(prob, prob_threshold)
        m1 = compute_all_metrics(b1_bin, b1_lab, gold_bin, gold_labeled)
        print(f"  Baseline1 → Dice:{m1['dice']:.3f}  IoU:{m1['iou']:.3f}  "
              f"ObjF1:{m1['obj_f1']:.3f}  HD95:{m1['hd95']:.1f}  "
              f"clDice:{m1['cldice']:.3f}  Betti:{m1['betti_err']:.0f}")

        # Baseline 2: sweep area thresholds
        metric_curves = {k: [] for k in ["dice","iou","obj_f1","hd95","cldice"]}
        for at in area_thresholds:
            b2_bin, b2_lab = remove_small_objects(b1_bin, b1_lab, at)
            m2 = compute_all_metrics(b2_bin, b2_lab, gold_bin, gold_labeled)
            for k in metric_curves:
                metric_curves[k].append(m2[k])

        # Plot metric vs area threshold
        fig, axes = plt.subplots(1, 5, figsize=(20, 3))
        metric_names = ["dice","iou","obj_f1","hd95","cldice"]
        for ax, mk in zip(axes, metric_names):
            ax.plot(area_thresholds, metric_curves[mk], 'o-')
            ax.set_title(mk)
            ax.set_xlabel("Area Threshold")
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"{ds_name} | {suffix} — Metric vs Area Threshold (Baseline 2)")
        plt.tight_layout()
        plt.savefig(ds_out / f"{suffix}_area_sweep.png", dpi=100)
        plt.close()

        # Best area threshold by dice
        best_idx = np.argmax(metric_curves["dice"])
        best_at  = area_thresholds[best_idx]
        b2_bin, b2_lab = remove_small_objects(b1_bin, b1_lab, best_at)
        m2 = compute_all_metrics(b2_bin, b2_lab, gold_bin, gold_labeled)
        print(f"  Baseline2 (best AT={best_at}) → Dice:{m2['dice']:.3f}  "
              f"IoU:{m2['iou']:.3f}  ObjF1:{m2['obj_f1']:.3f}  "
              f"HD95:{m2['hd95']:.1f}  clDice:{m2['cldice']:.3f}  Betti:{m2['betti_err']:.0f}")

        # Visualize
        _save_comparison(prob, gold, b1_bin, b2_bin,
                         ds_out / f"{suffix}_baselines.png",
                         titles=["Prob Map","Gold","Baseline1","Baseline2"])

    return best_at  # return last best_at for use in later parts

# ─────────────────────────────────────────────
# PART 3: MORPHOLOGICAL OPERATORS
# ─────────────────────────────────────────────

def run_part3(pairs, ds_name, prob_threshold=0.5, area_threshold=50):
    """Apply morphological operators starting from Baseline 2."""
    print(f"\n{'='*60}")
    print(f"PART 3: {ds_name}")
    print(f"{'='*60}")

    ds_out = OUT_ROOT / "part3" / ds_name
    ds_out.mkdir(parents=True, exist_ok=True)

    for suffix, prob, gold in pairs:
        gold_bin     = gold > 0
        gold_labeled = gold.astype(int)

        # Start from baseline 2
        b1_bin, b1_lab = threshold_and_label(prob, prob_threshold)
        b2_bin, b2_lab = remove_small_objects(b1_bin, b1_lab, area_threshold)

        # Config A: closing (disk r=3) — fills small holes, connects nearby blobs
        selem_a = disk(3)
        config_a = binary_closing(b2_bin, selem_a)
        config_a_lab = label(config_a)

        # Config B: opening (disk r=2) then dilation (disk r=2)
        # opening removes thin protrusions, dilation slightly expands boundaries
        selem_b = disk(2)
        config_b = binary_opening(b2_bin, selem_b)
        config_b = binary_dilation(config_b, selem_b)
        config_b_lab = label(config_b)

        m_b2 = compute_all_metrics(b2_bin, b2_lab, gold_bin, gold_labeled)
        m_ca = compute_all_metrics(config_a, config_a_lab, gold_bin, gold_labeled)
        m_cb = compute_all_metrics(config_b, config_b_lab, gold_bin, gold_labeled)

        print(f"\n  Image: {suffix}")
        print(f"  Baseline2  → Dice:{m_b2['dice']:.3f}  IoU:{m_b2['iou']:.3f}  "
              f"ObjF1:{m_b2['obj_f1']:.3f}  HD95:{m_b2['hd95']:.1f}  clDice:{m_b2['cldice']:.3f}")
        print(f"  ConfigA(closing r=3) → Dice:{m_ca['dice']:.3f}  IoU:{m_ca['iou']:.3f}  "
              f"ObjF1:{m_ca['obj_f1']:.3f}  HD95:{m_ca['hd95']:.1f}  clDice:{m_ca['cldice']:.3f}")
        print(f"  ConfigB(open+dilate) → Dice:{m_cb['dice']:.3f}  IoU:{m_cb['iou']:.3f}  "
              f"ObjF1:{m_cb['obj_f1']:.3f}  HD95:{m_cb['hd95']:.1f}  clDice:{m_cb['cldice']:.3f}")

        _save_comparison(prob, gold, config_a, config_b,
                         ds_out / f"{suffix}_morphology.png",
                         titles=["Prob Map","Gold","Closing (A)","Open+Dilate (B)"])

# ─────────────────────────────────────────────
# PART 4: SEED-CONTROLLED REGION GROWING
# ─────────────────────────────────────────────

def region_growing(prob, pt=0.7, at=20):
    """
    Seed-controlled region growing.
    PT: probability threshold for seeds
    AT: area threshold to remove small seeds
    """
    # Foreground seeds: prob >= PT
    fg_seed_bin = prob >= pt
    fg_seed_lab = label(fg_seed_bin)
    # Remove small fg seeds
    for r in regionprops(fg_seed_lab):
        if r.area < at:
            fg_seed_bin[fg_seed_lab == r.label] = False
    fg_seed_lab = label(fg_seed_bin)

    # Background seeds: prob < (1 - PT)
    bg_seed_bin = prob < (1 - pt)
    bg_seed_lab = label(bg_seed_bin)
    for r in regionprops(bg_seed_lab):
        if r.area < at:
            bg_seed_bin[bg_seed_lab == r.label] = False

    # Label map: 0=unassigned, -1=background, >0=foreground id
    assignment = np.zeros(prob.shape, dtype=np.int32)
    assignment[bg_seed_bin] = -1
    for r in regionprops(label(fg_seed_bin)):
        assignment[label(fg_seed_bin) == r.label] = r.label

    # Priority queue based growing (using heap via confidence)
    import heapq
    heap = []
    h, w = prob.shape

    # Add border pixels of seeds to heap
    def add_neighbors(y, x, current_label):
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < h and 0 <= nx < w and assignment[ny, nx] == 0:
                conf = prob[ny, nx] if current_label > 0 else (1 - prob[ny, nx])
                heapq.heappush(heap, (-conf, ny, nx, current_label))

    # Initialize heap from all seed borders
    for yy in range(h):
        for xx in range(w):
            if assignment[yy, xx] != 0:
                add_neighbors(yy, xx, assignment[yy, xx])

    # Grow
    while heap:
        neg_conf, y, x, lbl = heapq.heappop(heap)
        if assignment[y, x] != 0:
            continue
        assignment[y, x] = lbl
        add_neighbors(y, x, lbl)

    result_bin = assignment > 0
    result_lab = label(result_bin)
    return result_bin, result_lab


def run_part4(pairs, ds_name, prob_threshold=0.5, area_threshold=50):
    """Run region growing for one dataset."""
    print(f"\n{'='*60}")
    print(f"PART 4: {ds_name}")
    print(f"{'='*60}")

    ds_out = OUT_ROOT / "part4" / ds_name
    ds_out.mkdir(parents=True, exist_ok=True)

    for suffix, prob, gold in pairs:
        gold_bin     = gold > 0
        gold_labeled = gold.astype(int)

        # Baseline 2 for comparison
        b1_bin, b1_lab = threshold_and_label(prob, prob_threshold)
        b2_bin, b2_lab = remove_small_objects(b1_bin, b1_lab, area_threshold)

        print(f"\n  Image: {suffix} — running region growing (may take a moment)...")
        rg_bin, rg_lab = region_growing(prob, pt=0.7, at=area_threshold)

        m_b2 = compute_all_metrics(b2_bin, b2_lab, gold_bin, gold_labeled)
        m_rg = compute_all_metrics(rg_bin, rg_lab, gold_bin, gold_labeled)

        print(f"  Baseline2 → Dice:{m_b2['dice']:.3f}  IoU:{m_b2['iou']:.3f}  "
              f"ObjF1:{m_b2['obj_f1']:.3f}  HD95:{m_b2['hd95']:.1f}  clDice:{m_b2['cldice']:.3f}")
        print(f"  RegionGrow → Dice:{m_rg['dice']:.3f}  IoU:{m_rg['iou']:.3f}  "
              f"ObjF1:{m_rg['obj_f1']:.3f}  HD95:{m_rg['hd95']:.1f}  clDice:{m_rg['cldice']:.3f}")

        _save_comparison(prob, gold, b2_bin, rg_bin,
                         ds_out / f"{suffix}_regiongrowing.png",
                         titles=["Prob Map","Gold","Baseline2","Region Growing"])

# ─────────────────────────────────────────────
# PART 5: CUSTOM PIPELINE
# ─────────────────────────────────────────────

def custom_pipeline(prob, area_threshold=50, prob_threshold=0.5):
    """
    Custom postprocessing pipeline:
    1. Threshold at 0.5 → binary mask
    2. Remove small objects (area < AT)
    3. Fill holes (binary_fill_holes) — recover interior pixels missed by network
    4. Morphological closing (disk 5) — smooth contours and connect nearby regions
    5. Watershed separation using distance transform — re-separate merged objects
    
    Rationale: filling holes + closing recovers missed interior pixels,
    then watershed re-separates objects that got wrongly merged during closing.
    """
    from scipy.ndimage import binary_fill_holes, distance_transform_edt
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    # Step 1+2: baseline
    b1_bin, b1_lab = threshold_and_label(prob, prob_threshold)
    b2_bin, _ = remove_small_objects(b1_bin, b1_lab, area_threshold)

    # Step 3: fill holes
    filled = binary_fill_holes(b2_bin)

    # Step 4: closing to smooth
    closed = binary_closing(filled, disk(5))

    # Step 5: watershed to re-separate merged objects
    dist = distance_transform_edt(closed)
    # find local maxima as seeds
    coords = peak_local_max(dist, min_distance=10, labels=closed)
    mask_peaks = np.zeros(dist.shape, dtype=bool)
    mask_peaks[tuple(coords.T)] = True
    markers = label(mask_peaks)
    ws = watershed(-dist, markers, mask=closed)

    result_bin = ws > 0
    result_lab = ws
    return result_bin, result_lab


def run_part5(pairs, ds_name, prob_threshold=0.5, area_threshold=50):
    """Run custom pipeline for one dataset."""
    print(f"\n{'='*60}")
    print(f"PART 5: {ds_name}")
    print(f"{'='*60}")

    ds_out = OUT_ROOT / "part5" / ds_name
    ds_out.mkdir(parents=True, exist_ok=True)

    for suffix, prob, gold in pairs:
        gold_bin     = gold > 0
        gold_labeled = gold.astype(int)

        b1_bin, b1_lab = threshold_and_label(prob, prob_threshold)
        b2_bin, b2_lab = remove_small_objects(b1_bin, b1_lab, area_threshold)

        print(f"\n  Image: {suffix} — running custom pipeline...")
        cp_bin, cp_lab = custom_pipeline(prob, area_threshold, prob_threshold)

        m_b2 = compute_all_metrics(b2_bin, b2_lab, gold_bin, gold_labeled)
        m_cp = compute_all_metrics(cp_bin, cp_lab, gold_bin, gold_labeled)

        print(f"  Baseline2  → Dice:{m_b2['dice']:.3f}  IoU:{m_b2['iou']:.3f}  "
              f"ObjF1:{m_b2['obj_f1']:.3f}  HD95:{m_b2['hd95']:.1f}  clDice:{m_b2['cldice']:.3f}")
        print(f"  CustomPipe → Dice:{m_cp['dice']:.3f}  IoU:{m_cp['iou']:.3f}  "
              f"ObjF1:{m_cp['obj_f1']:.3f}  HD95:{m_cp['hd95']:.1f}  clDice:{m_cp['cldice']:.3f}")

        _save_comparison(prob, gold, b2_bin, cp_bin,
                         ds_out / f"{suffix}_custom.png",
                         titles=["Prob Map","Gold","Baseline2","Custom Pipeline"])

# ─────────────────────────────────────────────
# VISUALIZATION HELPER
# ─────────────────────────────────────────────

def _save_comparison(prob, gold, result1, result2, path, titles=None):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    imgs = [prob, gold, result1.astype(float), result2.astype(float)]
    if titles is None:
        titles = ["Prob","Gold","Result1","Result2"]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img, cmap='nipy_spectral' if title=="Gold" else 'gray')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(str(path), dpi=100, bbox_inches='tight')
    plt.close()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    from PIL import Image  # ensure available

    print("COMP 448/548 — HW1 Full Pipeline")
    print(f"Data root: {DATA_ROOT}")
    print(f"Output:    {OUT_ROOT}\n")

    # Area threshold to use for parts 3-5 (you can tune per dataset)
    AREA_THRESH = {
        "cell-detection": 10,
        "c-elegans-worm-segmentation": 50,
        "corneal-nerve-segmentation": 20,
        "gland-segmentation": 200,
        "liver-segmentation": 500,
        "oct-fluid-segmentation": 100,
    }

    for ds in DATASETS:
        ds_path = DATA_ROOT / ds
        if not ds_path.exists():
            print(f"[SKIP] {ds} not found")
            continue
        print(f"\n{'#'*60}")
        print(f"# DATASET: {ds}")
        print(f"{'#'*60}")
        pairs = load_dataset(ds)
        if not pairs:
            print(f"  No valid pairs found, skipping.")
            continue
        at = AREA_THRESH.get(ds, 50)
        run_part2(pairs, ds, area_thresholds=[0,5,10,20,50,100,200,500,1000])
        run_part3(pairs, ds, area_threshold=at)
        run_part4(pairs, ds, area_threshold=at)
        run_part5(pairs, ds, area_threshold=at)

    print(f"\n\nDone! Results saved to {OUT_ROOT}")

if __name__ == "__main__":
    main()
