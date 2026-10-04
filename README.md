# PUMix

PyTorch implementation of:

**Asymmetric Uncertainty-guided Mixing and Structure-aware Prototype Contrastive Learning for Semi-supervised Medical Image Segmentation**

PUMix is a Mean Teacher-based semi-supervised segmentation framework that improves supervision reliability at the data level and feature discrimination at the representation level.

It contains three main components:

- **AUPM** — Asymmetric Uncertainty-guided PatchMix.
- **MSPR** — Multi-Scale Prototype Representation.
- **SPCL** — Structure-aware Prototype Contrastive Learning.

---

## Repository Structure

```text
PUMix/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── acdc_5.yaml
│   ├── acdc_10.yaml
│   ├── synapse_10.yaml
│   └── synapse_50.yaml
├── code/
│   ├── train.py
│   ├── test.py
│   ├── dataloaders/
│   ├── methods/
│   ├── networks/
│   └── utils/
├── data/

```

---

## Installation

The original experiments were conducted on an NVIDIA Tesla T4 GPU.

The released implementation is compatible with PyTorch 2.6–2.10. For GPU training, install a CUDA-enabled PyTorch build appropriate for your system before installing the remaining dependencies.

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install torch==2.8.0 torchvision==0.23.0 `
    --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('CUDA build:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected output for the recommended setup is similar to:

```text
Torch: 2.8.0+cu126
CUDA: True
CUDA build: 12.6
GPU: NVIDIA ...
```

> **Note:** If CUDA is not available, install another CUDA-enabled PyTorch build compatible with your local NVIDIA driver and GPU.

---

## Datasets

Experiments are conducted on the **ACDC** and **Synapse** datasets.

| Dataset | Download |
|---|---|
| ACDC | [Google Drive](https://drive.google.com/file/d/1GRIONz95IwJo5pkPUePzs-8wVUZky2Ew/view?usp=sharing) |
| Synapse | [Google Drive](https://drive.google.com/file/d/1NGTbugbrhia74PE63l-FjyvmVIgZc7Sm/view?usp=sharing) |

After downloading, extract the datasets into the `data/` directory.

### ACDC

Expected structure:

```text
data/ACDC/
├── train_slices.list
├── val.list
├── test.list
└── data/
    ├── slices/
    │   ├── caseXXX_sliceXXX.h5
    │   └── ...
    ├── patientXXX_frameXX.h5
    └── ...
```

Each HDF5 file contains `image` and `label`.

Experimental settings:

- **5% labeled:** 3 labeled patients
- **10% labeled:** 7 labeled patients

### Synapse

Expected structure:

```text
data/Synapse/
├── train_slices.txt
├── val.txt
├── train_npz/
│   ├── caseXXXX_sliceXXX.npz
│   └── ...
└── test_vol_h5/
    ├── caseXXXX.npy.h5
    └── ...
```

Each training `.npz` file and evaluation `.npy.h5` volume contains `image` and `label`.

Experimental settings:

- **10% labeled:** 2 labeled scans
- **50% labeled:** 9 labeled scans

---

## Training

Run the corresponding configuration:

| Dataset | Setting | Command |
|---|---:|---|
| ACDC | 5% | `python code/train.py --config configs/acdc_5.yaml` |
| ACDC | 10% | `python code/train.py --config configs/acdc_10.yaml` |
| Synapse | 10% | `python code/train.py --config configs/synapse_10.yaml` |
| Synapse | 50% | `python code/train.py --config configs/synapse_50.yaml` |

The code automatically uses CUDA when a compatible GPU and CUDA-enabled PyTorch installation are available.

---

## Evaluation

Example:

```bash
python code/test.py \
    --config configs/acdc_10.yaml \
    --checkpoint outputs/pumix_acdc_10pct/best.pth
```

The evaluation reports:

- Dice
- Jaccard
- ASD
- 95HD

for each foreground class and their mean values.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{pumix,
  title   = {Asymmetric Uncertainty-guided Mixing and Structure-aware Prototype Contrastive Learning for Semi-supervised Medical Image Segmentation},
  author  = {...},
  journal = {...},
  year    = {2026}
}
```

The complete citation information will be updated after publication.

---

## Acknowledgements

We thank the authors of the public datasets and open-source methods used in this work.
