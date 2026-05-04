"""
COMP 448/548 - Homework 3
UNet for Teeth Segmentation
Covers: Part 1 (base UNet), Part 2 (architecture mods), Part 3 (dropout)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# ─────────────────────────────────────────────
# BUILDING BLOCK: Double Convolution
# Used in every encoder and decoder block
# ─────────────────────────────────────────────

class DoubleConv(nn.Module):
    """
    Two consecutive: Conv2d → BatchNorm → ReLU
    This is the basic repeated unit in UNet.
    dropout_p: if > 0, adds dropout after second conv (used in Part 3)
    """
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


# ─────────────────────────────────────────────
# ENCODER BLOCK
# DoubleConv → save for skip → MaxPool (halve size)
# ─────────────────────────────────────────────

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, dropout_p)
        self.pool = nn.MaxPool2d(2)  # halves H and W

    def forward(self, x):
        features = self.conv(x)      # save this for skip connection
        pooled = self.pool(features) # this goes to the next encoder level
        return features, pooled


# ─────────────────────────────────────────────
# DECODER BLOCK
# Upsample → concatenate skip → DoubleConv
# ─────────────────────────────────────────────

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        # ConvTranspose2d doubles H and W (learnable upsampling)
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # After concatenation with skip, channels double → need to halve
        self.conv = DoubleConv(in_channels, out_channels, dropout_p)

    def forward(self, x, skip):
        x = self.upsample(x)           # double spatial size, halve channels
        x = torch.cat([skip, x], dim=1) # concatenate skip connection → channels double
        x = self.conv(x)               # learn to combine coarse + fine info
        return x


# ─────────────────────────────────────────────
# PART 1: BASE UNET
# 4 downsampling levels, starting channels = 16
# Channel sequence: 16 → 32 → 64 → 128 → 256 (bottleneck)
# ─────────────────────────────────────────────

class UNet(nn.Module):
    """
    Base UNet for Part 1.
    init_features: starting number of channels (default 16 as required)
    dropout_p: dropout probability for Part 3 (default 0 = no dropout)
    """
    def __init__(self, in_channels=1, out_channels=1, init_features=16, dropout_p=0.0):
        super().__init__()
        f = init_features  # shorthand

        # ENCODER (going down)
        self.enc1 = EncoderBlock(in_channels, f,      dropout_p)   # 16
        self.enc2 = EncoderBlock(f,           f*2,    dropout_p)   # 32
        self.enc3 = EncoderBlock(f*2,         f*4,    dropout_p)   # 64
        self.enc4 = EncoderBlock(f*4,         f*8,    dropout_p)   # 128

        # BOTTLENECK (deepest point)
        self.bottleneck = DoubleConv(f*8, f*16, dropout_p)         # 256

        # DECODER (going up)
        self.dec4 = DecoderBlock(f*16, f*8,  dropout_p)            # 128
        self.dec3 = DecoderBlock(f*8,  f*4,  dropout_p)            # 64
        self.dec2 = DecoderBlock(f*4,  f*2,  dropout_p)            # 32
        self.dec1 = DecoderBlock(f*2,  f,    dropout_p)            # 16

        # FINAL LAYER: collapse to 1 channel, then sigmoid for probability
        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder: each block returns (skip_features, pooled_for_next)
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder: each block takes (current, skip_from_encoder)
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        # Output: 1 probability per pixel
        return self.sigmoid(self.final_conv(x))


# ─────────────────────────────────────────────
# PART 2 - MODIFICATION 1: 3-level UNet
# Fewer downsampling steps: 4 → 3
# Channel sequence: 16 → 32 → 64 → 128 (bottleneck)
# ─────────────────────────────────────────────

class UNet3Level(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout_p=0.0):
        super().__init__()

        # ENCODER (3 levels instead of 4)
        self.enc1 = EncoderBlock(in_channels, 16,  dropout_p)
        self.enc2 = EncoderBlock(16,          32,  dropout_p)
        self.enc3 = EncoderBlock(32,          64,  dropout_p)

        # BOTTLENECK
        self.bottleneck = DoubleConv(64, 128, dropout_p)

        # DECODER (3 levels)
        self.dec3 = DecoderBlock(128, 64, dropout_p)
        self.dec2 = DecoderBlock(64,  32, dropout_p)
        self.dec1 = DecoderBlock(32,  16, dropout_p)

        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        x = self.bottleneck(x)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        return self.sigmoid(self.final_conv(x))


# ─────────────────────────────────────────────
# PART 2 - MODIFICATION 2: Different init channels
# Use UNet class with init_features=8 or init_features=32
# Example:
#   model_8ch  = UNet(init_features=8)   → 8-16-32-64-128
#   model_32ch = UNet(init_features=32)  → 32-64-128-256-512
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# DATASET
# Loads images and their binary masks
# ─────────────────────────────────────────────

class TeethDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.images = sorted(os.listdir(image_dir))

        self.img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(),              # X-rays are grayscale
            transforms.ToTensor(),               # → [0,1] float, shape (1,H,W)
            transforms.Normalize([0.5], [0.5])   # → [-1,1]
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(),
            transforms.ToTensor()                # mask stays [0,1], no normalize
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir,  self.images[idx])

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")  # grayscale mask

        image = self.img_transform(image)
        mask  = self.mask_transform(mask)
        mask  = (mask > 0.5).float()  # binarize: 0 or 1

        return image, mask


# ─────────────────────────────────────────────
# METRICS
# Pixel-level precision, recall, F-score
# ─────────────────────────────────────────────

def compute_metrics(pred, target, threshold=0.5):
    """
    pred:   model output after sigmoid, shape (B, 1, H, W)
    target: binary mask,               shape (B, 1, H, W)
    Returns: precision, recall, f_score (averaged over batch)
    """
    pred_binary = (pred > threshold).float()

    TP = (pred_binary * target).sum()
    FP = (pred_binary * (1 - target)).sum()
    FN = ((1 - pred_binary) * target).sum()

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f_score   = 2 * precision * recall / (precision + recall + 1e-8)

    return precision.item(), recall.item(), f_score.item()


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def train(model, train_loader, val_loader, num_epochs=50, lr=1e-3, device='cuda'):
    model = model.to(device)

    # Adam optimizer: adaptive learning rate, works well without much tuning
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ReduceLROnPlateau: halve LR if val loss doesn't improve for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Dice loss: better than BCE for imbalanced foreground/background
    def dice_loss(pred, target):
        numerator   = 2 * (pred * target).sum()
        denominator = pred.sum() + target.sum() + 1e-8
        return 1 - numerator / denominator

    best_val_loss = float('inf')
    best_model_state = None
    history = []

    for epoch in range(num_epochs):
        # ── TRAINING PHASE ──
        model.train()
        train_loss, train_prec, train_rec, train_f = 0, 0, 0, 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            p, r, f = compute_metrics(outputs, masks)
            train_loss += loss.item()
            train_prec += p; train_rec += r; train_f += f

        n = len(train_loader)
        train_loss /= n; train_prec /= n; train_rec /= n; train_f /= n

        # ── VALIDATION PHASE ──
        model.eval()
        val_loss, val_prec, val_rec, val_f = 0, 0, 0, 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = dice_loss(outputs, masks)

                p, r, f = compute_metrics(outputs, masks)
                val_loss += loss.item()
                val_prec += p; val_rec += r; val_f += f

        m = len(val_loader)
        val_loss /= m; val_prec /= m; val_rec /= m; val_f /= m

        scheduler.step(val_loss)

        # Save best model (stopping condition: best validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        history.append({
            'epoch': epoch+1,
            'train_loss': train_loss, 'train_prec': train_prec,
            'train_rec':  train_rec,  'train_f':    train_f,
            'val_loss':   val_loss,   'val_prec':   val_prec,
            'val_rec':    val_rec,    'val_f':       val_f,
        })

        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {train_loss:.4f} F: {train_f:.4f} | "
              f"Val Loss: {val_loss:.4f} F: {val_f:.4f}")

    # Load the best model before returning
    model.load_state_dict(best_model_state)
    return model, history


# ─────────────────────────────────────────────
# EVALUATION on test set
# ─────────────────────────────────────────────

def evaluate(model, test_loader, device='cuda'):
    model.eval()
    model.to(device)
    all_prec, all_rec, all_f = [], [], []

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            # Compute per-image metrics then average
            for i in range(images.shape[0]):
                p, r, f = compute_metrics(outputs[i:i+1], masks[i:i+1])
                all_prec.append(p); all_rec.append(r); all_f.append(f)

    print(f"\nTest Results:")
    print(f"  Precision: {np.mean(all_prec):.4f}")
    print(f"  Recall:    {np.mean(all_rec):.4f}")
    print(f"  F-score:   {np.mean(all_f):.4f}")
    return np.mean(all_prec), np.mean(all_rec), np.mean(all_f)


# ─────────────────────────────────────────────
# MAIN: Run everything
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ── Dataset paths (adjust to your folder structure) ──
    train_dataset = TeethDataset("data/images/train", "data/masks/train")
    val_dataset   = TeethDataset("data/images/val",   "data/masks/val")
    test_dataset  = TeethDataset("data/images/test",  "data/masks/test")

    # batch_size=8: fits most GPUs, stable gradient estimates
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=8)
    test_loader  = DataLoader(test_dataset,  batch_size=8)

    # ════════════════════════════════
    # PART 1: Base UNet (16 channels)
    # ════════════════════════════════
    print("\n=== PART 1: Base UNet ===")
    model_base = UNet(in_channels=1, out_channels=1, init_features=16)
    model_base, history = train(model_base, train_loader, val_loader, num_epochs=50, device=device)
    torch.save(model_base.state_dict(), "unet_base.pth")
    evaluate(model_base, test_loader, device)

    # ════════════════════════════════
    # PART 2a: 3-level UNet
    # ════════════════════════════════
    print("\n=== PART 2a: 3-Level UNet ===")
    model_3level = UNet3Level(in_channels=1, out_channels=1)
    model_3level, _ = train(model_3level, train_loader, val_loader, num_epochs=50, device=device)
    evaluate(model_3level, test_loader, device)

    # ════════════════════════════════
    # PART 2b: Different channel counts
    # ════════════════════════════════
    print("\n=== PART 2b: 8-channel UNet ===")
    model_8ch = UNet(in_channels=1, out_channels=1, init_features=8)
    model_8ch, _ = train(model_8ch, train_loader, val_loader, num_epochs=50, device=device)
    evaluate(model_8ch, test_loader, device)

    # OR: 32-channel UNet (uncomment to use instead)
    # print("\n=== PART 2b: 32-channel UNet ===")
    # model_32ch = UNet(in_channels=1, out_channels=1, init_features=32)
    # model_32ch, _ = train(model_32ch, train_loader, val_loader, num_epochs=50, device=device)
    # evaluate(model_32ch, test_loader, device)

    # ════════════════════════════════
    # PART 3: Dropout (3 p-values)
    # ════════════════════════════════
    for p_val in [0.1, 0.3, 0.5]:
        print(f"\n=== PART 3: Dropout p={p_val} ===")
        model_drop = UNet(in_channels=1, out_channels=1, init_features=16, dropout_p=p_val)
        model_drop, _ = train(model_drop, train_loader, val_loader, num_epochs=50, device=device)
        evaluate(model_drop, test_loader, device)