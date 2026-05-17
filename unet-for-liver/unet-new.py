"""
COMP 448/548 - Homework 4
Multi-Task U-Net for Liver Segmentation
Parts 1, 2 (nnU-Net instructions separate), and 3 (custom loss)

ARCHITECTURE OVERVIEW
─────────────────────
Baseline UNet: same as HW3 (encoder → bottleneck → segmentation decoder)

Multi-Task UNet:
    Input
      ↓
  [Shared Encoder]   ← same 4-level encoder as HW3
      ↓
  [Bottleneck]
      ↓ (splits into two parallel paths)
  [Seg Decoder]      [Reg Decoder]
      ↓                   ↓
  Seg Output          Reg Output
  (sigmoid, 0-1)      (raw values — image or distance map)

WHY TWO DECODERS?
  Multi-task learning forces the shared encoder to learn more general
  features. The regression task (predict image OR distance map) acts as
  a regularizer, helping the encoder not overfit to just segmentation.

WHY DISTANCE MAP?
  The distance transform assigns each foreground pixel a value = how far
  it is from the nearest boundary. This gives the network spatial structure
  information (center vs edge of organ). Learning to predict this helps
  the encoder understand object shape.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy import ndimage


# ══════════════════════════════════════════════════════════════════
# REUSED FROM HW3 (unchanged)
# ══════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(p=dropout_p))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, dropout_p)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout_p)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """Baseline UNet — same as HW3, used for Part 1 baseline."""
    def __init__(self, in_channels=1, out_channels=1, init_features=16, dropout_p=0.0):
        super().__init__()
        f = init_features
        self.enc1 = EncoderBlock(in_channels, f,    dropout_p)
        self.enc2 = EncoderBlock(f,           f*2,  dropout_p)
        self.enc3 = EncoderBlock(f*2,         f*4,  dropout_p)
        self.enc4 = EncoderBlock(f*4,         f*8,  dropout_p)
        self.bottleneck = DoubleConv(f*8, f*16, dropout_p)
        self.dec4 = DecoderBlock(f*16, f*8,  dropout_p)
        self.dec3 = DecoderBlock(f*8,  f*4,  dropout_p)
        self.dec2 = DecoderBlock(f*4,  f*2,  dropout_p)
        self.dec1 = DecoderBlock(f*2,  f,    dropout_p)
        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        return self.sigmoid(self.final_conv(x))


# ══════════════════════════════════════════════════════════════════
# PART 1: MULTI-TASK U-NET
# Key change: one encoder, two separate decoders
# ══════════════════════════════════════════════════════════════════

class MTUNet(nn.Module):
    """
    Multi-Task UNet: shared encoder + two independent decoders.

    Segmentation decoder: predicts binary liver mask (sigmoid output)
    Regression decoder:   predicts a continuous target (no sigmoid)
                          — either the input image (experiment 1)
                          — or the distance map (experiment 2)

    IMPORTANT: Both decoders share the SAME skip connections from the encoder.
    This is fine — skip connections are just feature maps being read, not modified.
    """
    def __init__(self, in_channels=1, init_features=16):
        super().__init__()
        f = init_features

        # ── SHARED ENCODER (identical to baseline UNet) ──
        self.enc1 = EncoderBlock(in_channels, f)    # out: f channels
        self.enc2 = EncoderBlock(f,           f*2)  # out: f*2 channels
        self.enc3 = EncoderBlock(f*2,         f*4)  # out: f*4 channels
        self.enc4 = EncoderBlock(f*4,         f*8)  # out: f*8 channels
        self.bottleneck = DoubleConv(f*8, f*16)     # out: f*16 channels

        # ── SEGMENTATION DECODER (same structure as HW3 decoder) ──
        self.seg_dec4 = DecoderBlock(f*16, f*8)
        self.seg_dec3 = DecoderBlock(f*8,  f*4)
        self.seg_dec2 = DecoderBlock(f*4,  f*2)
        self.seg_dec1 = DecoderBlock(f*2,  f)
        self.seg_head = nn.Conv2d(f, 1, kernel_size=1)   # → 1 channel
        self.sigmoid  = nn.Sigmoid()                      # → probabilities [0,1]

        # ── REGRESSION DECODER (same structure, different output head) ──
        # Exactly mirroring the segmentation decoder — same blocks, separate weights
        self.reg_dec4 = DecoderBlock(f*16, f*8)
        self.reg_dec3 = DecoderBlock(f*8,  f*4)
        self.reg_dec2 = DecoderBlock(f*4,  f*2)
        self.reg_dec1 = DecoderBlock(f*2,  f)
        self.reg_head = nn.Conv2d(f, 1, kernel_size=1)   # → 1 channel
        # NO sigmoid here — we want raw continuous values for regression

    def forward(self, x):
        # ── Shared encoder: runs once, produces skip connections ──
        skip1, x = self.enc1(x)   # skip1: (B, f,    H,   W  )
        skip2, x = self.enc2(x)   # skip2: (B, f*2,  H/2, W/2)
        skip3, x = self.enc3(x)   # skip3: (B, f*4,  H/4, W/4)
        skip4, x = self.enc4(x)   # skip4: (B, f*8,  H/8, W/8)
        x = self.bottleneck(x)    # x:     (B, f*16, H/16,W/16)

        # ── Segmentation decoder: upsamples + uses skip connections ──
        seg = self.seg_dec4(x, skip4)
        seg = self.seg_dec3(seg, skip3)
        seg = self.seg_dec2(seg, skip2)
        seg = self.seg_dec1(seg, skip1)
        seg_out = self.sigmoid(self.seg_head(seg))  # shape: (B,1,H,W), range [0,1]

        # ── Regression decoder: same structure, same skips, separate weights ──
        reg = self.reg_dec4(x, skip4)   # x is still the bottleneck output
        reg = self.reg_dec3(reg, skip3)
        reg = self.reg_dec2(reg, skip2)
        reg = self.reg_dec1(reg, skip1)
        reg_out = self.reg_head(reg)    # shape: (B,1,H,W), unbounded continuous

        return seg_out, reg_out


# ══════════════════════════════════════════════════════════════════
# DISTANCE MAP GENERATION
# Used for the second regression experiment
# ══════════════════════════════════════════════════════════════════

def compute_distance_map(binary_mask_numpy):
    """
    Given a binary mask (0/1 numpy array, shape H×W),
    returns the Euclidean distance transform.

    For each foreground pixel (=1): value = distance to nearest boundary.
    For background pixels (=0): value = 0.

    This is the "outer distance map" — it encodes shape/thickness of the organ.
    The center of the liver gets the largest value; edges get values near 0.

    Example:
        mask = [[0,0,0],         distance = [[0,0,0],
                [0,1,0],    →               [0,1,0],
                [0,0,0]]                    [0,0,0]]

        mask = [[0,0,0,0,0],     distance = [[0,0,0,0,0],
                [0,1,1,1,0],  →             [0,1,1,1,0],
                [0,1,1,1,0],                [0,1,2,1,0],
                [0,0,0,0,0]]                [0,0,0,0,0]]
    """
    # distance_transform_edt computes Euclidean distance from each 1-pixel to nearest 0-pixel
    distance = ndimage.distance_transform_edt(binary_mask_numpy)
    return distance.astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# DATASET — Liver version
# Key changes from HW3:
#   1. Filenames: "volume-X_Y_0000.png" format
#   2. Validation split: volumes 81-90 (by filename)
#   3. Optional z-normalization
#   4. Returns regression target (image or distance map) alongside mask
# ══════════════════════════════════════════════════════════════════

class LiverDataset(Dataset):
    """
    reg_target: 'image'    → regression target is the input image itself
                'distance' → regression target is the distance map of the mask
    z_norm:     True       → apply z-normalization to input images (and distance map if used)
    """
    def __init__(self, image_dir, label_dir, reg_target='image',
                 z_norm=True, img_size=(256, 256)):
        self.image_dir  = image_dir
        self.label_dir  = label_dir
        self.reg_target = reg_target
        self.z_norm     = z_norm
        self.img_size   = img_size
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]

        # ── Load image ──
        img = Image.open(os.path.join(self.image_dir, fname)).convert("L")
        img = img.resize(self.img_size)
        img = np.array(img, dtype=np.float32) / 255.0  # normalize to [0,1]

        # ── Z-normalization ──
        # z-norm = (x - mean) / std
        # WHY: makes the input distribution zero-centered with unit variance,
        # which helps gradient flow and convergence speed.
        if self.z_norm:
            img = (img - img.mean()) / (img.std() + 1e-8)

        # ── Load label (binary mask) ──
        label_fname = fname.replace("_0000", "")  # labels don't have _0000 suffix
        lbl = Image.open(os.path.join(self.label_dir, label_fname)).convert("L")
        lbl = lbl.resize(self.img_size)
        lbl = np.array(lbl, dtype=np.float32) / 255.0
        lbl = (lbl > 0.5).astype(np.float32)  # binarize

        # ── Regression target ──
        if self.reg_target == 'image':
            # Experiment 1: predict the input image itself
            # This forces the decoder to learn a general reconstruction of anatomy
            reg = img.copy()

        elif self.reg_target == 'distance':
            # Experiment 2: predict the distance map
            # This forces the decoder to understand organ shape/thickness
            reg = compute_distance_map(lbl)

            # Z-normalize the distance map too if requested
            # (important: distance values can be large; normalization stabilizes training)
            if self.z_norm and reg.max() > 0:
                reg = (reg - reg.mean()) / (reg.std() + 1e-8)

        # ── Convert to tensors (add channel dimension) ──
        img_t = torch.tensor(img).unsqueeze(0)   # (1, H, W)
        lbl_t = torch.tensor(lbl).unsqueeze(0)   # (1, H, W)
        reg_t = torch.tensor(reg).unsqueeze(0)   # (1, H, W)

        return img_t, lbl_t, reg_t


def get_volume_number(filename):
    """Extract the volume number X from 'volume-X_Y_0000.png'."""
    # filename like: volume-27_3_0000.png → volume number = 27
    part = filename.split("_")[0]       # "volume-27"
    return int(part.split("-")[1])      # 27


def split_train_val(image_dir, val_volumes=range(81, 91)):
    """
    Split filenames into train and validation sets.
    Validation: volumes 81–90 inclusive (as specified in hw).
    """
    all_files = sorted(os.listdir(image_dir))
    train_files = [f for f in all_files if get_volume_number(f) not in val_volumes]
    val_files   = [f for f in all_files if get_volume_number(f) in val_volumes]
    return train_files, val_files


# ══════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def dice_loss(pred, target, smooth=1e-8):
    """
    Dice loss for segmentation.
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice  (so 0 is perfect, 1 is worst)

    WHY Dice instead of plain BCE?
    Liver scans have mostly background pixels — BCE would be dominated
    by background. Dice focuses on the overlap of foreground regions.

    For Q1: "What is your loss for the base UNet?"
    Answer: Dice loss (or BCE + Dice combination).
    """
    numerator   = 2 * (pred * target).sum()
    denominator = pred.sum() + target.sum() + smooth
    return 1 - numerator / denominator


def mse_loss(pred, target):
    """
    MSE loss for the regression task.
    MSE = mean((pred - target)^2)

    WHY MSE for regression?
    The regression output is a continuous value (image pixel or distance).
    MSE penalizes large deviations quadratically — appropriate for continuous targets.
    BCE/Dice only make sense for binary targets.

    For Q2: "What is your loss for the regression task?"
    Answer: Mean Squared Error (MSE).
    """
    return ((pred - target) ** 2).mean()


# ══════════════════════════════════════════════════════════════════
# METRICS (same as HW3, per-image averaging)
# ══════════════════════════════════════════════════════════════════

def compute_metrics(pred, target, threshold=0.5):
    """
    Computes pixel-level precision, recall, and F-score (Dice) per image.
    The HW requires averaging metrics computed for each image separately
    (not pooling all pixels together first).
    """
    pred_binary = (pred > threshold).float()
    TP = (pred_binary * target).sum(dim=(1, 2, 3))          # per image in batch
    FP = (pred_binary * (1 - target)).sum(dim=(1, 2, 3))
    FN = ((1 - pred_binary) * target).sum(dim=(1, 2, 3))

    precision = (TP / (TP + FP + 1e-8)).mean().item()
    recall    = (TP / (TP + FN + 1e-8)).mean().item()
    f_score   = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f_score


# ══════════════════════════════════════════════════════════════════
# TRAINING LOOP — Baseline UNet (segmentation only)
# ══════════════════════════════════════════════════════════════════

def train_baseline(model, train_loader, val_loader, num_epochs=50, lr=1e-3, device='cuda'):
    """
    Train the baseline UNet (segmentation only, Dice loss).
    Same approach as HW3.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # ReduceLROnPlateau: if val loss doesn't improve for 5 epochs, halve the LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(num_epochs):
        # ── Train ──
        model.train()
        train_loss = 0
        for imgs, masks, _ in train_loader:   # ignore reg target for baseline
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = dice_loss(pred, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ──
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                pred = model(imgs)
                val_loss += dice_loss(pred, masks).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Train Loss: {train_loss/len(train_loader):.4f}"
                  f" | Val Loss: {val_loss:.4f}")

    model.load_state_dict(best_state)
    return model


# ══════════════════════════════════════════════════════════════════
# TRAINING LOOP — Multi-Task UNet
# ══════════════════════════════════════════════════════════════════

def train_mt(model, train_loader, val_loader, alpha=1.0, num_epochs=50,
             lr=1e-3, device='cuda'):
    """
    Train the MT-UNet with combined segmentation + regression loss.

    Total loss = Loss_segm + alpha * Loss_reg

    UNDERSTANDING ALPHA:
      alpha controls how much the regression task influences training.
      - Large alpha (100, 10): regression dominates → encoder focuses on
        reconstructing image/distance map, may hurt segmentation
      - Small alpha (0.01): segmentation dominates → regression barely helps
      - Middle ground (1.0, 0.1): both tasks contribute meaningfully

    You need to try: alpha ∈ {100, 10, 1, 0.1, 0.01} and report best.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(num_epochs):
        # ── Train ──
        model.train()
        train_loss = 0
        for imgs, masks, reg_targets in train_loader:
            imgs       = imgs.to(device)
            masks      = masks.to(device)
            reg_targets = reg_targets.to(device)

            optimizer.zero_grad()

            # Forward pass: two outputs from two decoders
            seg_pred, reg_pred = model(imgs)

            # Segmentation loss: Dice (same as baseline)
            loss_seg = dice_loss(seg_pred, masks)

            # Regression loss: MSE between predicted and actual target
            loss_reg = mse_loss(reg_pred, reg_targets)

            # Combined loss: segmentation + scaled regression
            # alpha weights how much the regression task matters
            total_loss = loss_seg + alpha * loss_reg

            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

        # ── Validate ──
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks, reg_targets in val_loader:
                imgs        = imgs.to(device)
                masks       = masks.to(device)
                reg_targets = reg_targets.to(device)
                seg_pred, reg_pred = model(imgs)
                loss = dice_loss(seg_pred, masks) + alpha * mse_loss(reg_pred, reg_targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {train_loss/len(train_loader):.4f}"
                  f" | Val: {val_loss:.4f}")

    model.load_state_dict(best_state)
    return model


# ══════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════

def evaluate(model, test_loader, device='cuda', multi_task=False):
    """
    Compute per-image precision, recall, F-score then average.

    Per-image averaging (required by HW):
      Compute metrics for image 1, image 2, ..., image N separately,
      then take the mean. This is different from pooling all pixels first.

    Why does this matter?
      If you pool all pixels, images with more pixels dominate the metric.
      Per-image averaging treats each image equally regardless of size.
    """
    model.eval()
    model.to(device)
    all_prec, all_rec, all_f = [], [], []

    with torch.no_grad():
        for imgs, masks, _ in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            if multi_task:
                seg_pred, _ = model(imgs)  # only use segmentation output
            else:
                seg_pred = model(imgs)

            # Compute per-image (one image at a time)
            for i in range(imgs.shape[0]):
                p, r, f = compute_metrics(seg_pred[i:i+1], masks[i:i+1])
                all_prec.append(p)
                all_rec.append(r)
                all_f.append(f)

    print(f"  Precision: {np.mean(all_prec):.4f} | "
          f"Recall: {np.mean(all_rec):.4f} | "
          f"Dice/F: {np.mean(all_f):.4f}")
    return np.mean(all_prec), np.mean(all_rec), np.mean(all_f)


# ══════════════════════════════════════════════════════════════════
# PART 3: CUSTOM LOSS FUNCTION — FIXED VERSION
#
# THE BUG EXPLAINED:
#   Original code computes ce_loss and dice_loss over the WHOLE BATCH
#   (single scalar each), then computes a single scalar coefficient.
#   So: coeff * base_loss = scalar * scalar → 1 number.
#
# THE FIX:
#   We need: for each sample i in batch → compute base_loss[i] and coeff[i]
#   Then: weighted_loss[i] = coeff[i] * base_loss[i]   ← element-wise
#   Final: mean(weighted_loss)                           ← scalar
#
# WHAT CHANGES:
#   1. BCEWithLogitsLoss(reduction='none') → gives per-pixel losses
#      then we average over spatial dims → per-sample CE (shape: B)
#   2. dice_loss: already per-sample (sum over H,W dims), remove .mean()
#   3. fp_norm, fn_norm: already per-sample; remove .mean() from coefficient
#   4. coeff shape: (B,) instead of scalar
#   5. Final: (coeff * base_loss).mean() → scalar
# ══════════════════════════════════════════════════════════════════

class FPFNWeightedLoss(nn.Module):
    """
    Custom loss that penalizes FP and FN explicitly.
    Multiplies the base loss (CE + Dice) by a coefficient that grows
    with the number of false positives/negatives.

    alpha: weight between FP and FN sensitivity
      alpha=0.5 → treats FP and FN equally
      alpha→1   → more sensitive to FP (precision-focused)
      alpha→0   → more sensitive to FN (recall-focused)
    """
    def __init__(self, alpha=0.5, smooth=1e-6):
        super().__init__()
        self.alpha  = alpha
        self.smooth = smooth
        # KEY FIX 1: reduction='none' → don't collapse to scalar yet
        # This gives us shape (B, 1, H, W) — one loss value per pixel
        self.ce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        """
        logits:  (B, 1, H, W) — raw network output before sigmoid
        targets: (B, 1, H, W) — binary masks {0, 1}
        """
        # ── Per-sample CE loss ──
        # self.ce gives (B, 1, H, W) pixel-wise losses
        # Average over spatial dims → shape (B,)   one number per image
        ce_loss = self.ce(logits, targets).mean(dim=(1, 2, 3))   # shape: (B,)

        # ── Probabilities ──
        probs = torch.sigmoid(logits)   # (B, 1, H, W)

        # ── Per-sample Dice loss ──
        # sum over spatial dims (1,2,3) → shape (B,) — one value per image
        intersection = (probs * targets).sum(dim=(1, 2, 3))       # (B,)
        union        = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))  # (B,)
        dice         = (2.0 * intersection + self.smooth) / (union + self.smooth)
        # KEY FIX 2: remove .mean() here — keep per-sample
        dice_loss    = 1 - dice    # shape: (B,)

        # ── Per-sample base loss ──
        base_loss = ce_loss + dice_loss  # shape: (B,)

        # ── Per-sample soft FP / FN ──
        fp = (probs * (1 - targets)).sum(dim=(1, 2, 3))       # (B,)
        fn = ((1 - probs) * targets).sum(dim=(1, 2, 3))       # (B,)

        # Normalize by number of pixels in ONE sample (not whole batch)
        # targets[0].numel() = 1 * H * W (pixels per image)
        n_pixels = targets[0].numel()
        fp_norm  = fp / (n_pixels + self.smooth)   # (B,)
        fn_norm  = fn / (n_pixels + self.smooth)   # (B,)

        # ── Per-sample coefficient ──
        # KEY FIX 3: remove .mean() — keep per-sample shape (B,)
        coeff = 1.0 + self.alpha * fp_norm + (1 - self.alpha) * fn_norm  # (B,)

        # ── Element-wise multiply, then average over batch ──
        # coeff * base_loss → (B,) element-wise product
        # .mean() → final scalar loss
        loss = (coeff * base_loss).mean()

        return loss


# ══════════════════════════════════════════════════════════════════
# TRAINING LOOP — with custom loss
# ══════════════════════════════════════════════════════════════════

def train_with_custom_loss(model, train_loader, val_loader, num_epochs=50,
                           lr=1e-3, device='cuda'):
    """
    Same as train_baseline, but uses FPFNWeightedLoss instead of Dice.
    Note: FPFNWeightedLoss takes LOGITS (before sigmoid), so we must
    change the UNet forward() to return logits for this training only.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = FPFNWeightedLoss(alpha=0.5)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            # IMPORTANT: FPFNWeightedLoss expects logits, not probabilities.
            # Use model WITHOUT sigmoid for this loss.
            # Simplest approach: call final_conv directly before sigmoid.
            logits = model.final_conv(
                model.dec1(
                    model.dec2(
                        model.dec3(
                            model.dec4(
                                model.bottleneck(model.enc4(model.enc3(model.enc2(model.enc1(imgs)[1])[1])[1])[1]),
                                model.enc4(model.enc3(model.enc2(model.enc1(imgs)[1])[1])[1])[0]
                            ),
                            model.enc3(model.enc2(model.enc1(imgs)[1])[1])[0]
                        ),
                        model.enc2(model.enc1(imgs)[1])[0]
                    ),
                    model.enc1(imgs)[0]
                )
            )
            # Cleaner approach: modify UNet to optionally return logits
            # See UNetWithLogits below.

            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                pred = model(imgs)
                val_loss += dice_loss(pred, masks).item()   # use dice for val monitoring

        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Train: {train_loss/len(train_loader):.4f}"
                  f" | Val Dice: {val_loss:.4f}")

    model.load_state_dict(best_state)
    return model


class UNetLogits(UNet):
    """
    UNet variant that returns logits (before sigmoid) — needed for FPFNWeightedLoss.
    FPFNWeightedLoss applies sigmoid internally via BCEWithLogitsLoss.
    """
    def forward(self, x):
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        return self.final_conv(x)  # logits, no sigmoid


def train_custom_loss_clean(model_logits, train_loader, val_loader, num_epochs=50,
                            lr=1e-3, device='cuda'):
    """
    Clean version using UNetLogits (returns logits for FPFNWeightedLoss).
    """
    model_logits = model_logits.to(device)
    optimizer  = optim.Adam(model_logits.parameters(), lr=lr)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion  = FPFNWeightedLoss(alpha=0.5)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(num_epochs):
        model_logits.train()
        train_loss = 0
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model_logits(imgs)              # logits: no sigmoid
            loss   = criterion(logits, masks)        # loss applies sigmoid internally
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model_logits.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model_logits(imgs)
                probs  = torch.sigmoid(logits)       # convert for Dice monitoring
                val_loss += dice_loss(probs, masks).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model_logits.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Train: {train_loss/len(train_loader):.4f}"
                  f" | Val Dice: {val_loss:.4f}")

    model_logits.load_state_dict(best_state)
    return model_logits


# ══════════════════════════════════════════════════════════════════
# MAIN: Run all experiments
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ── Paths (update to your structure) ──
    TRAIN_IMGS  = "data/imagesTr"
    TRAIN_LBLS  = "data/labelsTr"
    TEST_IMGS   = "data/imagesTs"
    TEST_LBLS   = "data/labelsTs"
    IMG_SIZE    = (256, 256)
    BATCH_SIZE  = 8
    EPOCHS      = 50

    # ── Get train/val filenames ──
    train_files, val_files = split_train_val(TRAIN_IMGS, val_volumes=range(81, 91))
    print(f"Train: {len(train_files)} images | Val: {len(val_files)} images")

    def make_loaders(reg_target, z_norm):
        """Helper to create train/val/test loaders for a given config."""
        train_ds = LiverDataset(TRAIN_IMGS, TRAIN_LBLS, reg_target=reg_target,
                                z_norm=z_norm, img_size=IMG_SIZE)
        # Filter to only train/val files
        train_ds.images = train_files
        val_ds = LiverDataset(TRAIN_IMGS, TRAIN_LBLS, reg_target=reg_target,
                              z_norm=z_norm, img_size=IMG_SIZE)
        val_ds.images = val_files
        test_ds = LiverDataset(TEST_IMGS, TEST_LBLS, reg_target=reg_target,
                               z_norm=z_norm, img_size=IMG_SIZE)
        return (DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
                DataLoader(val_ds,   batch_size=BATCH_SIZE),
                DataLoader(test_ds,  batch_size=BATCH_SIZE))

    # ════════════════════════════════════════════
    # Experiment loop: with and without z-norm
    # ════════════════════════════════════════════
    for z_norm in [True, False]:
        tag = "WITH z-norm" if z_norm else "WITHOUT z-norm"
        print(f"\n{'='*60}")
        print(f"  {tag}")
        print(f"{'='*60}")

        # ── Baseline UNet ──
        print("\n--- Baseline UNet ---")
        train_l, val_l, test_l = make_loaders('image', z_norm)
        baseline = UNet(in_channels=1, out_channels=1, init_features=16)
        baseline = train_baseline(baseline, train_l, val_l, num_epochs=EPOCHS, device=device)
        evaluate(baseline, test_l, device=device, multi_task=False)

        # ── MT-UNet with image regression, sweep alpha ──
        print("\n--- MT-UNet (image regression), alpha sweep ---")
        train_l, val_l, test_l = make_loaders('image', z_norm)
        best_alpha_img, best_f_img = None, -1
        for alpha in [100, 10, 1, 0.1, 0.01]:
            print(f"  alpha={alpha}")
            model_mt = MTUNet(in_channels=1, init_features=16)
            model_mt = train_mt(model_mt, train_l, val_l, alpha=alpha,
                                num_epochs=EPOCHS, device=device)
            p, r, f = evaluate(model_mt, test_l, device=device, multi_task=True)
            if f > best_f_img:
                best_f_img = f
                best_alpha_img = alpha
        print(f"  Best alpha for MT-UNet(image): {best_alpha_img}")

        # ── MT-UNet with distance map regression, sweep alpha ──
        print("\n--- MT-UNet (distance map), alpha sweep ---")
        train_l, val_l, test_l = make_loaders('distance', z_norm)
        best_alpha_dist, best_f_dist = None, -1
        for alpha in [100, 10, 1, 0.1, 0.01]:
            print(f"  alpha={alpha}")
            model_mt = MTUNet(in_channels=1, init_features=16)
            model_mt = train_mt(model_mt, train_l, val_l, alpha=alpha,
                                num_epochs=EPOCHS, device=device)
            p, r, f = evaluate(model_mt, test_l, device=device, multi_task=True)
            if f > best_f_dist:
                best_f_dist = f
                best_alpha_dist = alpha
        print(f"  Best alpha for MT-UNet(distance): {best_alpha_dist}")

        # ── Part 3: UNet with custom FPFNWeightedLoss ──
        print("\n--- UNet with Custom FPFNWeightedLoss ---")
        train_l, val_l, test_l = make_loaders('image', z_norm)
        model_custom = UNetLogits(in_channels=1, out_channels=1, init_features=16)
        model_custom = train_custom_loss_clean(model_custom, train_l, val_l,
                                               num_epochs=EPOCHS, device=device)
        # For evaluation, sigmoid the logits
        class EvalWrapper(nn.Module):
            def __init__(self, logit_model):
                super().__init__()
                self.m = logit_model
            def forward(self, x):
                return torch.sigmoid(self.m(x))
        eval_model = EvalWrapper(model_custom)
        evaluate(eval_model, test_l, device=device, multi_task=False)