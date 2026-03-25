# LPRNet — License Plate Recognition

A PyTorch implementation of **LPRNet**, a lightweight and high-performance License Plate Recognition framework. This project includes a training pipeline and an evaluation script.

> Based on the paper: [LPRNet: License Plate Recognition via Deep Neural Networks](https://arxiv.org/abs/1806.10447v1)

---

## Table of Contents

- [Project Structure](#project-structure)
- [Package Organization](#package-organization)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [1. Prepare Dataset](#1-prepare-dataset)
  - [2. Train the Model](#2-train-the-model)
  - [3. Test / Evaluate](#3-test--evaluate)
- [Model Architecture](#model-architecture)
- [Training Arguments](#training-arguments)
- [Testing Arguments](#testing-arguments)
- [Character Set](#character-set)
- [Pre-trained Weights](#pre-trained-weights)
- [Performance](#performance)
- [References](#references)

---

## Project Structure

```
LPRNet-Training-Pipeline/
├── README.md
├── scripts/                             # Entry point scripts
│   ├── train.py                         # Training entry point
│   └── test.py                          # Evaluation entry point
│
├── lprnet/                              # Core package (modular)
│   ├── data/
│   │   └── loader.py                    # Dataset loader + character set
│   └── model/
│       └── lprnet.py                    # Architecture + model builder
│
├── data/                                # Dataset + compatibility exports
│   ├── Dataset/
│   │   ├── Images/                      # Training/validation images
│   │   └── Labels/                      # Optional label text files
│   └── test/                            # Evaluation images
│
├── model/                               # Compatibility exports
│
├── outputs/
│   ├── checkpoints/                     # Optional custom save directory
│   └── logs/                            # Reserved for logs/plots
│
└── weights/
    └── Final_LPRNet_model.pth           # Default pretrained/final weights
```

### Folder Descriptions

| Folder | Purpose |
|--------|---------|
| **scripts/** | Entry points for training and evaluation. Use `python -m scripts.train` or `python -m scripts.test` |
| **lprnet/** | **Core implementation package** — Contains data loading and model architecture code. Clean, modular structure |
| **lprnet/data/** | Data loading module with `LPRDataLoader` class and character definitions |
| **lprnet/model/** | Neural network model definitions and factory functions |
| **data/** | Raw dataset folder. Compatibility layer re-exports from `lprnet/data/` |
| **outputs/** | Optional location for checkpoints/logs if you override defaults |
| **outputs/checkpoints/** | Placeholder for custom checkpoint output |
| **outputs/logs/** | Placeholder for custom logs/metrics |
| **model/** | Compatibility layer re-exports from `lprnet/model/` |
| **weights/** | Default location where training saves checkpoints/final model |

---

## Package Organization

The project uses a **modular package structure** for clean code organization and reusability:

### Core Package: `lprnet/`
- **lprnet.data**: Character definitions and data loading
  - `CHARS`: 37-character vocabulary (0-9, A-Z excluding I/O, plus CTC blank)
  - `CHARS_DICT`: Character-to-index mapping
  - `LPRDataLoader`: PyTorch Dataset for loading and preprocessing images

- **lprnet.model**: Neural network implementation
  - `LPRNet`: Main model class with CTC-compatible output
  - `build_lprnet()`: Factory function for model instantiation
  - `small_basic_block`: Factorized convolution block for efficiency

### Import Examples

```python
# Recommended: Direct imports from lprnet modules
from lprnet.data import CHARS, LPRDataLoader
from lprnet.model import LPRNet, build_lprnet

# Backward compatibility imports (still work)
from data import CHARS, LPRDataLoader
from model import LPRNet, build_lprnet
```

### Compatibility Layers

The `data/` and `model/` folders at the project root provide backward compatibility, automatically re-exporting from the core `lprnet/` package. This allows older code to continue working while maintaining the clean modular structure.

---

## Requirements

- Python 3.x
- PyTorch >= 1.0.0
- opencv-python 3.x
- imutils
- Pillow
- numpy
- matplotlib

---

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd LPRNet-Training-Pipeline

# Install dependencies
pip install torch torchvision
pip install opencv-python imutils Pillow numpy matplotlib
```

---

## Dataset

### Image Format
- Images must be **94 × 24 pixels** (width × height). Larger images are automatically resized.
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

### Naming Convention
The data loader extracts the license plate label **directly from the image filename** using the first token before `-` (or `_`):

```
<PLATE_TEXT>-<anything>.jpg
```

**Examples:**
```
DL10CJ0330-001.jpg   →   label: DL10CJ0330
UP84T8969-002.jpg    →   label: UP84T8969
```

> **Note:** If images are named like `<timestamp>-<PLATE>.jpg`, the loader will treat the timestamp as label. Rename images to `<PLATE_TEXT>-<id>.jpg` before training.

### Directory Layout
```
data/Dataset/
├── Images/
│   ├── DL10CJ0330-001.jpg
│   └── UP84T8969-002.jpg
└── Labels/
  ├── DL10CJ0330-001.txt
  └── UP84T8969-002.txt
```

### Labeling Instructions (TXT Files)

For license plate training datasets, keep one `.txt` file per image in `data/Dataset/Labels/`.

Rules:
- Label filename must match image filename (same base name).
- Each `.txt` file must contain the **complete vehicle number plate text**.
- Use one plate string per file (single line), for example: `DL10CJ0330`.
- Keep plate text uppercase and without extra spaces.

Example:
- Image: `data/Dataset/Images/DL10CJ0330-001.jpg`
- Label: `data/Dataset/Labels/DL10CJ0330-001.txt`
- File content: `DL10CJ0330`

Current scripts parse labels from image filenames, but keeping correct `.txt` labels is recommended for dataset quality and compatibility with other pipelines.

---

## Usage

### 1. Prepare Dataset

1. Place training/validation images in `data/Dataset/Images/`.
2. Ensure each filename starts with the true plate text (for example: `DL10CJ0330-001.jpg`).
3. Create matching `.txt` labels in `data/Dataset/Labels/`, where each file contains the full number plate text.
4. Place test images in `data/test/`.

---

### 2. Train the Model

```bash
# Quick start (uses default settings)
python -m scripts.train

# Custom hyperparameters
python -m scripts.train \
    --max_epoch 100 \
    --train_img_dirs ./data/Dataset \
    --test_img_dirs ./data/Dataset \
    --train_batch_size 128 \
    --learning_rate 0.1 \
    --cuda True

# Save checkpoints into outputs/checkpoints instead of default weights/
python -m scripts.train \
    --save_folder ./outputs/checkpoints/
```

**Output:**
- By default, checkpoints are saved to: `weights/LPRNet__iteration_<N>.pth` every `--save_interval` steps
- Final model is saved to: `weights/Final_LPRNet_model.pth`
- If `--save_folder ./outputs/checkpoints/` is used, files are saved under `outputs/checkpoints/`

---

### 3. Test / Evaluate

```bash
# Test with pre-trained weights
python -m scripts.test

# With custom weights
python -m scripts.test \
    --test_img_dirs ./data/test \
    --pretrained_model ./weights/Final_LPRNet_model.pth \
    --cuda True

# Show visual predictions on each image
python -m scripts.test \
    --test_img_dirs ./data/test \
    --pretrained_model ./weights/Final_LPRNet_model.pth \
    --show True
```

**Output Format:**
```
[Info] Test Accuracy: 0.960 [tp:tn_len_mismatch:tn_wrong_chars:total]
```

**Note:** By default, the script loads from `./weights/Final_LPRNet_model.pth`. If you train with a custom `--save_folder`, pass that model path via `--pretrained_model`.

---

## Model Architecture

LPRNet is a fully convolutional network that uses **CTC (Connectionist Temporal Classification)** loss for sequence recognition without requiring character segmentation.

### Backbone Layers

| Index | Layer | Output Channels |
|-------|-------|-----------------|
| 0–2   | Conv2d 3×3 → BN → ReLU | 64 |
| 3     | MaxPool3d 3×3 | — |
| 4–6   | small_basic_block → BN → ReLU | 128 |
| 7     | MaxPool3d 3×3 stride 2×1×2 | — |
| 8–10  | small_basic_block → BN → ReLU | 256 |
| 11–13 | small_basic_block → BN → ReLU | 256 |
| 14    | MaxPool3d 3×3 stride 4×1×2 | — |
| 15–18 | Dropout → Conv2d 1×4 → BN → ReLU | 256 |
| 19–22 | Dropout → Conv2d 13×1 → BN → ReLU | class_num |

### `small_basic_block`

A factorized convolution block that reduces parameters:
```
Conv2d(ch_in → ch_out//4, 1×1) → ReLU
→ Conv2d(3×1, pad=(1,0)) → ReLU   # vertical conv
→ Conv2d(1×3, pad=(0,1)) → ReLU   # horizontal conv
→ Conv2d(ch_out//4 → ch_out, 1×1)
```

### Feature Fusion

Features are extracted at layers **2, 6, 13, 22**, spatially aligned via `AvgPool2d`, L2-normalised, and concatenated into a **448 + class_num** channel tensor. A final `1×1` Conv reduces this to `class_num` logits per time-step.

---

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_epoch` | `100` | Number of training epochs |
| `--img_size` | `[94, 24]` | Input image size [W, H] |
| `--train_img_dirs` | `./data/Dataset` | Training data directory |
| `--test_img_dirs` | `./data/Dataset` | Validation data directory |
| `--dropout_rate` | `0.5` | Dropout probability |
| `--learning_rate` | `0.1` | Initial learning rate |
| `--lpr_max_len` | `8` | Maximum plate character length |
| `--train_batch_size` | `128` | Training batch size |
| `--test_batch_size` | `120` | Validation batch size |
| `--cuda` | `True` | Use GPU (set `False` for CPU) |
| `--resume_epoch` | `0` | Resume training from this epoch |
| `--save_interval` | `2000` | Save checkpoint every N iterations |
| `--test_interval` | `2000` | Run evaluation every N iterations |
| `--momentum` | `0.9` | RMSprop momentum |
| `--weight_decay` | `2e-5` | L2 regularization |
| `--lr_schedule` | `[4,8,12,14,16]` | Epochs to decay LR by ×0.1 |
| `--pretrained_model` | `""` | Path to pre-trained weights (optional) |

**Output Locations:**
- Default checkpoints: `weights/` — model weights at each save interval
- Optional custom checkpoints: set `--save_folder`, e.g. `./outputs/checkpoints/`

---

## Testing Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--test_img_dirs` | `./data/test` | Test image directory |
| `--pretrained_model` | `./weights/Final_LPRNet_model.pth` | Model weights path |
| `--dropout_rate` | `0` | Dropout (disabled at test time) |
| `--cuda` | `True` | Use GPU |
| `--show` | `False` | Display images with predictions |
| `--lpr_max_len` | `8` | Maximum plate character length |
| `--test_batch_size` | `100` | Batch size for inference |
| `--img_size` | `[94, 24]` | Input image size [W, H] |

---

## Character Set

The model recognises **36 characters** (37 including the CTC blank token):

| Category | Characters | Indices |
|----------|-----------|---------|
| Digits | `0–9` | 0–9 |
| Letters | `A–H, J–N, P–Z` | 10–33 |
| Special | `I`, `O` | 34–35 |
| CTC Blank | `-` | 36 |

> `I` and `O` are placed at the end because they are ambiguous with `1` and `0` on physical plates, but they are still supported.

---

## Pre-trained Weights

A pre-trained model is included at:

```
weights/Final_LPRNet_model.pth
```

This model was trained on a mixed dataset of Chinese blue and green (new-energy) license plates.

---

## Performance

| Model Size | Accuracy (personal test set) | Inference Speed (GTX 1060) |
|------------|------------------------------|---------------------------|
| ~1.7 MB | 96.0%+ | < 0.5 ms / image |

- Test set: 27,320 images
- Plate types: blue plate + green (new-energy) plate

---

## References

1. [LPRNet: License Plate Recognition via Deep Neural Networks](https://arxiv.org/abs/1806.10447v1)
2. [PyTorch Documentation](https://pytorch.org/docs/)
