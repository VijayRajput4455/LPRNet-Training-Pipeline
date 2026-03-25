# LPRNet — License Plate Recognition

A PyTorch implementation of **LPRNet**, a lightweight and high-performance License Plate Recognition framework. This project includes a data labeling utility, a training pipeline, and an evaluation script.

> Based on the paper: [LPRNet: License Plate Recognition via Deep Neural Networks](https://arxiv.org/abs/1806.10447v1)

---

## Table of Contents

- [Project Structure](#project-structure)
- [Package Organization](#package-organization)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [1. Label Images (demo.py)](#1-label-images-demopy)
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
LPRNET/
├── README.md
└── LPRNet_Pytorch/
    ├── scripts/                         # Entry point scripts
    │   ├── __init__.py
    │   ├── train.py                     # Training entry point
    │   └── test.py                      # Evaluation entry point
    │
    ├── lprnet/                          # Core package (modular)
    │   ├── __init__.py
    │   ├── data/                        # Data module
    │   │   ├── __init__.py
    │   │   └── loader.py                # Dataset & character definitions
    │   └── model/                       # Model module
    │       ├── __init__.py
    │       └── lprnet.py                # Architecture & building functions
    │
    ├── data/                            # Dataset (compatibility layer)
    │   ├── __init__.py
    │   ├── Dataset/
    │   │   ├── Images/                  # Training images (94×24)
    │   │   └── Labels/                  # Label .txt files
    │   └── test/                        # Evaluation images
    │
    ├── outputs/                         # Training artifacts
    │   ├── checkpoints/                 # Model weights during training
    │   │   └── *.pth files
    │   └── logs/                        # Training metrics & loss curves
    │
    ├── model/                           # Model (compatibility layer)
    │   └── __init__.py
    │
    ├── weights/                         # Pre-trained model storage
    │   └── Final_LPRNet_model.pth       # Pre-trained weights
    │
    └── demo.py                          # Image labeling utility
```

### Folder Descriptions

| Folder | Purpose |
|--------|---------|
| **scripts/** | Entry points for training and evaluation. Use `python -m scripts.train` or `python -m scripts.test` |
| **lprnet/** | **Core implementation package** — Contains data loading and model architecture code. Clean, modular structure |
| **lprnet/data/** | Data loading module with `LPRDataLoader` class and character definitions |
| **lprnet/model/** | Neural network model definitions and factory functions |
| **data/** | Raw dataset folder. Compatibility layer re-exports from `lprnet/data/` |
| **outputs/** | Training outputs — checkpoints and logs go here, keeping source code clean |
| **outputs/checkpoints/** | Model weights saved during training at different epochs |
| **outputs/logs/** | Training metrics, loss curves, validation results |
| **model/** | Compatibility layer re-exports from `lprnet/model/` |
| **weights/** | Final pre-trained model storage |

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
cd LPRNET

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
The data loader extracts the license plate label **directly from the image filename**. The filename must follow this pattern:

```
<PLATE_TEXT>-<anything>.jpg
```

**Examples:**
```
DL10CJ0330-001.jpg   →   label: DL10CJ0330
UP84T8969-002.jpg    →   label: UP84T8969
```

> **Note:** The current dataset images are named `<timestamp>-<PLATE_TEXT>.jpg`. In this case the data loader will read the timestamp as the label. Rename your images to `<PLATE_TEXT>-<id>.jpg` before training.

### Directory Layout
```
data/Dataset/
├── Images/
│   ├── DL10CJ0330-001.jpg
│   └── UP84T8969-002.jpg
└── Labels/
    ├── DL10CJ0330-001.txt   ← contains: DL10CJ0330
    └── UP84T8969-002.txt    ← contains: UP84T8969
```

---

## Usage

### 1. Label Images (demo.py)

Use this utility to manually label images from the `Images/` folder. It displays each image and prompts you to type the license plate text, saving the result to a `.txt` file in `Labels/`.

```bash
# Run from the project root
python demo.py
```

- **Input directory:** `LPRNet_Pytorch/data/Dataset/Images/`
- **Output directory:** `LPRNet_Pytorch/data/Dataset/Labels/`
- For each image, type the plate text and press **Enter**. Type `q` to quit.

---

### 2. Train the Model

```bash
cd LPRNet_Pytorch

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
```

**Output:**
- Checkpoints saved to: `outputs/checkpoints/LPRNet__iteration_<N>.pth` every `--save_interval` steps
- Final model saved to: `outputs/checkpoints/Final_LPRNet_model.pth`
- Training logs saved to: `outputs/logs/` (loss curves, metrics)

---

### 3. Test / Evaluate

```bash
cd LPRNet_Pytorch

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
Test Accuracy: 0.960 [tp:tn_len_mismatch:tn_wrong_chars:total]
```

**Note:** By default, the script loads from `./weights/Final_LPRNet_model.pth`. After training, you can copy the best checkpoint from `outputs/checkpoints/` to `weights/` for evaluation.

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
| `--save_interval` | `2000` | Save checkpoint every N iterations (saved to `outputs/checkpoints/`) |
| `--test_interval` | `2000` | Run evaluation every N iterations |
| `--momentum` | `0.9` | RMSprop momentum |
| `--weight_decay` | `2e-5` | L2 regularization |
| `--lr_schedule` | `[4,8,12,14,16]` | Epochs to decay LR by ×0.1 |
| `--pretrained_model` | `""` | Path to pre-trained weights (optional) |

**Output Locations:**
- Checkpoints: `outputs/checkpoints/` — model weights at each save interval
- Training logs: `outputs/logs/` — loss curves and metrics

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
LPRNet_Pytorch/weights/Final_LPRNet_model.pth
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
