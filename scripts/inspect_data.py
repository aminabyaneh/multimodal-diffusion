"""
Inspect and visualise the contents of a real-world HDF5 dataset.

Checks RGB / depth images, observation keys, Z-channel action profiles,
action norms, and episode lengths for a given dataset file.

Usage:
    python inspect_data.py                          # uses default dataset path
    python inspect_data.py --dataset data/my.hdf5  # custom path
"""

import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


# ---------------------------------------------------------------------------
# Inspection routines
# ---------------------------------------------------------------------------

def inspect_obs_keys(dataset: h5py.File):
    """Print the observation keys present in the first demo."""
    first_demo = next(iter(dataset.keys()))
    keys = list(dataset[first_demo]["obs"].keys())
    print_section("Observation keys")
    print(f"Demo '{first_demo}' obs keys: {keys}")


def inspect_depth(dataset: h5py.File, demo: str = "demo_2", timestep: int = 20):
    """Display a single depth frame and print its value range."""
    print_section(f"Depth image — {demo} / t={timestep}")
    sample = dataset[demo]["obs"]["agentview"]["depth"][timestep]
    print(f"Shape: {sample.shape}")
    print(f"Min: {np.min(sample):.4f}  Max: {np.max(sample):.4f}")

    plt.figure()
    plt.imshow(sample.squeeze(), cmap='gray')
    plt.axis('off')
    plt.title(f"Depth — {demo} t={timestep}")
    plt.tight_layout()
    plt.show()


def inspect_rgb(dataset: h5py.File, demo: str = "demo_25", timestep: int = 30):
    """Display a single RGB frame and print its associated action."""
    print_section(f"RGB image — {demo} / t={timestep}")
    sample = dataset[demo]["obs"]["agentview"]["color"][timestep]
    action = dataset[demo]["actions"][timestep]
    print(f"Associated action: {action}")

    plt.figure()
    plt.imshow(sample)
    plt.axis('off')
    plt.title(f"RGB — {demo} t={timestep}")
    plt.tight_layout()
    plt.show()


def inspect_z_actions(dataset: h5py.File, demos: list = None):
    """Plot the Z-channel actions for a set of demos, highlighting positive steps."""
    print_section("Z-channel actions")

    # fall back to first 3 demos if not specified
    if demos is None:
        demos = ["demo_10", "demo_25", "demo_48"]
    demos = [d for d in demos if d in dataset.keys()]
    if not demos:
        demos = list(dataset.keys())[:3]

    n = len(demos)
    fig, axs = plt.subplots(n, 1, figsize=(10, 4 * n), squeeze=False)

    for i, demo in enumerate(demos):
        demo_actions = dataset[demo]["actions"][:]
        z = demo_actions[:, 2] if demo_actions.ndim > 1 else demo_actions
        t = np.arange(len(z))

        mask_pos = z > 0
        ax = axs[i, 0]
        ax.plot(t[~mask_pos], z[~mask_pos], color='C0', marker='o', label='Z ≤ 0')
        if mask_pos.any():
            ax.plot(t[mask_pos], z[mask_pos], color='C1', marker='o', linestyle='', label='Z > 0')
            ax.plot(t[mask_pos], z[mask_pos], color='C1', linewidth=1, alpha=0.6)

        ax.set_xlabel('Timestep')
        ax.set_ylabel('Z action')
        ax.set_title(f'Z actions — {demo}')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()


def inspect_action_norms(dataset: h5py.File):
    """Compute and print statistics on the L2 norm of XYZ actions across all demos."""
    print_section("Action norm statistics")

    all_actions = []
    for demo in dataset.keys():
        all_actions.append(dataset[demo]["actions"][:][:, :3])
    actions = np.concatenate(all_actions, axis=0)

    norms = np.linalg.norm(actions, axis=1)
    print(f"  Mean   : {norms.mean():.6f}")
    print(f"  Median : {np.median(norms):.6f}")
    print(f"  Max    : {norms.max():.6f}")
    print(f"  Min    : {norms.min():.6f}")


def inspect_episode_lengths(dataset: h5py.File):
    """Compute and print episode length statistics."""
    print_section("Episode length statistics")

    lengths = [dataset[d]["actions"].shape[0] for d in dataset.keys()]
    print(f"  Num episodes : {len(lengths)}")
    print(f"  Mean length  : {np.mean(lengths):.2f}")
    print(f"  Max length   : {np.max(lengths)}")
    print(f"  Min length   : {np.min(lengths)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inspect a real-world HDF5 dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/circle_m_peg_insert_limited.hdf5",
        help="Path to the HDF5 dataset file.",
    )
    args = parser.parse_args()

    print(f"Opening dataset: {args.dataset}")
    with h5py.File(args.dataset, 'r') as dataset:
        inspect_obs_keys(dataset)
        inspect_depth(dataset)
        inspect_rgb(dataset)
        inspect_z_actions(dataset)
        inspect_action_norms(dataset)
        inspect_episode_lengths(dataset)

    print("\nDone.")


if __name__ == "__main__":
    main()
