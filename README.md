# Multimodal Diffusion Policies

This repository integrates tactile representations with diffusion policies for robotic manipulation tasks. It supports diverse vision backbones (ResNet18/34/50, DinoV2) and two diffusion policy variants: **Diffusion Behavior Cloning (DBC)** and **Diffusion Policy (DP)**.

---

## Setup

### 1. Environment Setup

Create and activate a Conda environment:

```bash
conda create -n multimodal-diffusion python=3.9
conda activate multimodal-diffusion
```

### 2. Install CleanDiffuser

This project depends on [CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser) for diffusion policy primitives and vision backbones. It is **not on PyPI** and must be installed from source before this package:

```bash
git clone https://github.com/CleanDiffuserTeam/CleanDiffuser.git
pip install -e CleanDiffuser/
```

### 3. Install This Package

```bash
pip install -e .
```

This installs the `source` package and all Python dependencies listed in `pyproject.toml`.
For pinned versions (recommended for exact reproducibility), install from `requirements.txt` instead:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Project Structure

```
multimodal-diffusion/
├── configs/                       # Hydra YAML configurations
│   ├── dbc/                       # DBC (DiT backbone) configs
│   │   ├── vision/
│   │   └── vision_tactile/
│   └── dp/                        # DP (ChiTransformer backbone) configs
│       ├── vision/
│       └── vision_tactile/
├── source/                        # Installable Python package
│   ├── __init__.py
│   ├── realworld_dataset.py       # HDF5 dataset loader with zarr replay buffer
│   ├── utils.py                   # Vision backbones, logging, training utilities
│   ├── vision_tactile_concat.py   # Multi-modal encoder (concatenation)
│   └── vision_tactile_film.py     # Multi-modal encoder (FiLM — in progress)
├── scripts/                       # Runnable entry points
│   ├── train_dbc.py               # Train DBC (DiT1d backbone)
│   ├── train_dp.py                # Train DP (ChiTransformer backbone)
│   ├── diffusion_server.py        # FastAPI inference server
│   ├── diffusion_client.py        # Real-robot client (Franka + RealSense + Digit)
│   ├── diffusion_fake_client.py   # Fake client for server testing (no hardware)
│   └── inspect_data.py            # Dataset inspection and visualisation
├── pyproject.toml
└── requirements.txt
```

---

## Data Format

Datasets are stored as HDF5 files with the following structure:

```
demo_0/
  obs/
    agentview/
      color       # (T, H, W, C) uint8 RGB frames
      depth       # (T, H, W) float32 depth frames
    tactile/
      finger_left # (T, 2, H, W, C) tactile images (first image used)
    ee_pos        # (T, 3) end-effector Cartesian position
    ee_euler      # (T, 3) end-effector Euler angles
  actions         # (T, action_dim) float32 actions
demo_1/
  ...
```

Place your dataset at the path specified by `dataset_path` in the config (default: `data/circle_m_peg_insert_limited.hdf5`).

---

## Training

Both training scripts are configured via Hydra. Edit the corresponding YAML in `configs/` to adjust dataset path, model architecture, and hyperparameters.

All scripts must be run from the **project root** so that relative paths (configs, data, logs) resolve correctly.

### DBC (Diffusion Behavior Cloning with DiT)

```bash
python scripts/train_dbc.py
```

To override config values from the command line:

```bash
python scripts/train_dbc.py dataset_path=data/my_dataset.hdf5 batch_size=32 device=cuda:1
```

### DP (Diffusion Policy with ChiTransformer)

```bash
python scripts/train_dp.py
```

Checkpoints and metrics are saved to `logs/` and logged to Weights & Biases (set `wandb_mode: offline` in the config to disable W&B sync).

---

## Deployment

### 1. Start the Inference Server

The server loads a trained checkpoint and exposes a `/act` REST endpoint.

```bash
python scripts/diffusion_server.py --checkpoint_dir ckpt/my_experiment/ --host 0.0.0.0 --port 8777
```

The checkpoint directory must contain:
- `config.yaml` — the training config (saved automatically during training)
- `model_<step>.pt` — the model checkpoint (specify the exact file via `--checkpoint_dir`)

### 2. Run the Real-Robot Client

Configure the hardware parameters (IPs, serial numbers) in the `Config` class inside `scripts/diffusion_client.py`, then run:

```bash
python scripts/diffusion_client.py
```

Hardware requirements:
- Franka robot arm with [Polymetis](https://facebookresearch.github.io/fairo/polymetis/) controller
- Intel RealSense camera (RGB + depth)
- [Digit](https://digit.ml/) tactile sensor

### 3. Test Without Hardware (Fake Client)

Use the fake client to verify server connectivity and action shapes without a real robot:

```bash
python scripts/diffusion_fake_client.py --checkpoint_dir ckpt/my_experiment/ [--enable_depth] [--enable_tactile]
```

---

## Configuration Reference

Key config parameters (see `configs/` for full examples):

| Parameter | Description |
|---|---|
| `nn` | Network type: `dit` (DBC) or `chi_transformer` (DP) |
| `diffusion` | Diffusion scheduler: `edm` |
| `rgb_model` | Vision backbone: `resnet18`, `resnet34`, `resnet50`, `vit_large_patch14_reg4_dinov2` |
| `conditioning` | Conditioning mode: `concat` (FiLM support planned) |
| `obs_steps` | Number of observation frames in the context window |
| `action_steps` | Number of actions to execute per inference call |
| `horizon` | Total action prediction horizon |
| `embedding_dim` | Image feature embedding dimension |
| `gradient_steps` | Total number of training gradient steps |
| `batch_size` | Training batch size |
| `lr` | Learning rate |
| `wandb_mode` | `online`, `offline`, or `disabled` |
