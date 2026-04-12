# HW2 - Skin Lesion Classification

Classification of dermatoscopic images from DermaMNIST dataset 
(7 classes, 10,015 images).

## Models
- **SimpleCNN** — trained from scratch, ~58% accuracy
- **ResNet50** — fine-tuned from ImageNet, ~72% accuracy  
- **ViT** — fine-tuned from ImageNet, ~?% accuracy

## Key Techniques
- Class imbalance handling via weighted CrossEntropyLoss
- Data augmentation (random flips)
- Early stopping
- ImageNet normalization

## Dataset

DermaMNIST — not included in repo, download and extract manually:

```bash
# Download
curl -L "https://zenodo.org/records/10519652/files/dermamnist_128.npz?download=1" -o dermamnist_128.npz

# Extract into folder
python3 -c "
import numpy as np
import os

data = np.load('dermamnist_128.npz')
os.makedirs('dermamnist_128', exist_ok=True)

for key in data.files:
    np.save(f'dermamnist_128/{key}.npy', data[key])
    print(f'Saved: {key} -> shape {data[key].shape}')
"
```

Place `dermamnist_128/` folder next to the notebook.