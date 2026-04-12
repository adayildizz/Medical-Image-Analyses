# HW1 - Medical Image Segmentation Postprocessing

Post-processing pipeline evaluation on 6 biomedical image segmentation datasets.

## Datasets
| Dataset | Task |
|---------|------|
| BRCA (cell-detection) | Cell detection |
| Elegans (c-elegans) | Worm segmentation |
| CORN (corneal-nerve) | Nerve segmentation |
| Gland (gland-segmentation) | Gland segmentation |
| Liver (liver-segmentation) | Liver segmentation |
| OCT (oct-fluid) | Fluid segmentation |

## Pipeline Parts
- **Part 2 — Baselines**: Simple thresholding + small object removal (area sweep)
- **Part 3 — Morphological Operators**: Closing, opening+dilation
- **Part 4 — Region Growing**: Seed-controlled priority queue region growing
- **Part 5 — Custom Pipeline**: Hole filling → closing → watershed separation

## Metrics
- **Overlap**: Dice, IoU
- **Match-based**: Object F1, Panoptic Quality
- **Distance**: Hausdorff (HD95), ASSD
- **Topology**: clDice, Betti error

## Usage
```bash
pip install numpy scipy scikit-image matplotlib pillow
python hw1_solution.py
```

Results saved to `results/` per part per dataset.