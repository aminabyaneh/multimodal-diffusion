# Multimodal Diffusion Policies

The repository integrates tactile representations with diffusion policies for manipulation tasks.

---

## Setup

### 1. Environment Setup

Create and activate a Conda or Mamba environment:

```bash
# Create Python 3.9 environment
conda create -n contractive-diffuser python=3.9

# Activate the environment
conda activate contractive-diffuser
```

### 2. Install Clean Diffuser

This project relies on the [Clean Diffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser) implementation for diffusion policies.

```bash
# (Optional) Create a directory for libraries
mkdir libs && cd libs

# Clone the Clean Diffuser repository
git clone https://github.com/CleanDiffuserTeam/CleanDiffuser.git
cd CleanDiffuser

# Install in editable mode
pip install -e .
```
