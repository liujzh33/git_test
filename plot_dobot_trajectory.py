#!/usr/bin/env python3
"""Plot dobot real-robot trajectory analysis for cook_vegetable and pour_water tasks.

Layout (4 rows, matching Robotwin_clip_code style):
  Row 0: Gripper states (left & right)
  Row 1: Joint velocity (left & right arm, computed from state diff)
  Row 2: Left-arm joint positions (6 joints) — no Cartesian EEF available
  Row 3: Right-arm joint positions (6 joints)

Data format (parquet, 14-dim):
  observation.state:  [left_arm_0..5, left_gripper, right_arm_0..5, right_gripper]
  observation.velocity: appears buggy (gripper slots = 1.0), so we compute from state diff
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


JOINT_NAMES_LEFT = ["L_J1", "L_J2", "L_J3", "L_J4", "L_J5", "L_J6"]
JOINT_NAMES_RIGHT = ["R_J1", "R_J2", "R_J3", "R_J4", "R_J5", "R_J6"]
COLORS_LEFT = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
COLORS_RIGHT = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def load_episode(parquet_path: Path):
    """Load a single episode from parquet, return dict of numpy arrays."""
    table = pq.read_table(str(parquet_path))
    n = table.num_rows

    states = np.array([row.as_py() for row in table["observation.state"]], dtype=np.float64)
    actions = np.array([row.as_py() for row in table["action"]], dtype=np.float64)
    timestamps = np.array([row.as_py() for row in table["timestamp"]], dtype=np.float64)

    # Gripper indices: 6 (left), 13 (right)
    left_gripper = states[:, 6]
    right_gripper = states[:, 13]

    # Joint positions (6 joints per arm)
    left_joints = states[:, 0:6]
    right_joints = states[:, 7:13]

    # Compute joint velocity from state diff (observation.velocity seems unreliable)
    left_vel = np.linalg.norm(np.diff(left_joints, axis=0), axis=1)
    left_vel = np.insert(left_vel, 0, 0.0)
    right_vel = np.linalg.norm(np.diff(right_joints, axis=0), axis=1)
    right_vel = np.insert(right_vel, 0, 0.0)

    return {
        "states": states,
        "actions": actions,
        "timestamps": timestamps,
        "left_gripper": left_gripper,
        "right_gripper": right_gripper,
        "left_joints": left_joints,
        "right_joints": right_joints,
        "left_vel": left_vel,
        "right_vel": right_vel,
        "total_steps": n,
    }


def plot_episode(data: dict, task_name: str, episode_idx: int, save_path: Path):
    """Plot trajectory analysis for one episode."""
    n = data["total_steps"]
    time_steps = np.arange(n)

    fig, axes = plt.subplots(4, 1, figsize=(16, 16))
    fig.suptitle(
        f"{task_name} — Episode {episode_idx} ({n} steps)",
        fontweight="bold", fontsize=14,
    )

    # --- Row 0: Gripper states ---
    ax = axes[0]
    ax.plot(time_steps, data["left_gripper"], "b-", label="Left Gripper", alpha=0.8, linewidth=1.2)
    ax.plot(time_steps, data["right_gripper"], "g-", label="Right Gripper", alpha=0.8, linewidth=1.2)
    ax.axhline(y=0.95, color="orange", linestyle="--", alpha=0.5, label="Open ~0.95")
    ax.axhline(y=0.05, color="k", linestyle="--", alpha=0.3, label="Closed ~0.05")
    ax.set_ylabel("Gripper Value")
    ax.set_title("Gripper States (low=closed, high=open)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n)

    # --- Row 1: Joint velocity ---
    ax = axes[1]
    ax.plot(time_steps, data["left_vel"], "b-", label="Left Arm Vel", alpha=0.7, linewidth=1.0)
    ax.plot(time_steps, data["right_vel"], "g-", label="Right Arm Vel", alpha=0.7, linewidth=1.0)
    ax.axhline(y=0.01, color="r", linestyle="--", alpha=0.5, label="Threshold 0.01")
    ax.set_ylabel("Joint Velocity (L2 norm)")
    ax.set_title("Arm Joint Velocity (computed from state diff)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n)

    # --- Row 2: Left-arm joint positions ---
    ax = axes[2]
    for j in range(6):
        ax.plot(time_steps, data["left_joints"][:, j], color=COLORS_LEFT[j],
                label=JOINT_NAMES_LEFT[j], alpha=0.8, linewidth=1.0)
    ax.set_ylabel("Joint Position (rad)")
    ax.set_title("Left Arm Joint Positions")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n)

    # --- Row 3: Right-arm joint positions ---
    ax = axes[3]
    for j in range(6):
        ax.plot(time_steps, data["right_joints"][:, j], color=COLORS_RIGHT[j],
                label=JOINT_NAMES_RIGHT[j], alpha=0.8, linewidth=1.0)
    ax.set_ylabel("Joint Position (rad)")
    ax.set_xlabel("Frame Index")
    ax.set_title("Right Arm Joint Positions")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n)

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot dobot real-robot trajectories.")
    parser.add_argument("--task", choices=["cook_vegetable", "pour_water", "both"], default="both")
    parser.add_argument("--episodes", type=str, default="0,1,2,3,4",
                        help="Comma-separated episode indices, e.g. '0,1,5,10'")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots")
    args = parser.parse_args()

    base_dir = Path("/cache/wx1513998/dobot")
    output_base = Path(args.output_dir) if args.output_dir else Path("/home/ma-user/work/wx1513998/dobot_plot")

    task_dirs = {
        "cook_vegetable": base_dir / "dobot_cook_vegetable_full",
        "pour_water": base_dir / "dobot_pour_water_full",
    }

    if args.task == "both":
        tasks = list(task_dirs.keys())
    else:
        tasks = [args.task]

    episode_ids = [int(x.strip()) for x in args.episodes.split(",")]

    for task in tasks:
        task_dir = task_dirs[task]
        data_dir = task_dir / "data" / "chunk-000"
        out_dir = output_base / task

        for ep_idx in episode_ids:
            parquet_file = data_dir / f"episode_{ep_idx:06d}.parquet"
            if not parquet_file.exists():
                print(f"Skip: {parquet_file} not found")
                continue

            data = load_episode(parquet_file)
            save_path = out_dir / f"{task}_episode_{ep_idx:04d}.png"
            plot_episode(data, task, ep_idx, save_path)

    print("Done.")


if __name__ == "__main__":
    main()
