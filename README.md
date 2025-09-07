# Multimodal Diffusion Policies

The repository integrates tactile representations with diffusion policies for manipulation tasks, and integrates more diverse vision backbones such as DinoV2 and SigLip.

---

## Setup

### 1. Environment Setup

Create and activate a Conda or Mamba environment:

```bash
# Create Python 3.9 environment
conda create -n multimodal-diffusion python=3.9

# Activate the environment
conda activate multimodal-diffusion
```

### 2. Install Clean Diffuser

This project relies on the [Clean Diffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser) implementation for diffusion policies and some vision backbones.

```bash
# (Optional) Create a directory for libraries
mkdir libs && cd libs

# Clone the Clean Diffuser repository
git clone https://github.com/CleanDiffuserTeam/CleanDiffuser.git
cd CleanDiffuser

# Install in editable mode
pip install -e .
```

### 3. Additional Modules

Install additional libs through the following command.

```python
pip install -r requirements.txt
```