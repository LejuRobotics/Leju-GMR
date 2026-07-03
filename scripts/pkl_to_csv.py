"""Convert GMR PKL motion files to CSV with strict robot layout validation.

CSV row format:
    [root_pos(x, y, z), root_rot(x, y, z, w), dof_pos(...)]

The robot DoF layout is defined per model and kept independent for future changes.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class RobotJointLayout:
    """Resolved robot DoF layout for output ordering and validation."""

    robot: str
    leg_indices: tuple[int, ...]
    waist_indices: tuple[int, ...]
    arm_indices: tuple[int, ...]
    expected_leg_dof: int
    expected_waist_dof: int
    expected_arm_dof: int

    @property
    def expected_total_dof(self) -> int:
        return self.expected_leg_dof + self.expected_waist_dof + self.expected_arm_dof

    @property
    def ordered_indices(self) -> tuple[int, ...]:
        return self.leg_indices + self.waist_indices + self.arm_indices


@dataclass(frozen=True)
class RobotLayoutSpec:
    """Declarative spec for building RobotJointLayout safely."""

    robot: str
    leg_dof: int
    waist_dof: int
    arm_dof: int
    leg_indices: Optional[tuple[int, ...]] = None
    waist_indices: Optional[tuple[int, ...]] = None
    arm_indices: Optional[tuple[int, ...]] = None


def _build_default_contiguous_indices(
    leg_dof: int,
    waist_dof: int,
    arm_dof: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    start = 0
    leg = tuple(range(start, start + leg_dof))
    start += leg_dof
    waist = tuple(range(start, start + waist_dof))
    start += waist_dof
    arm = tuple(range(start, start + arm_dof))
    return leg, waist, arm


def make_robot_joint_layout(spec: RobotLayoutSpec) -> RobotJointLayout:
    """Build one robot layout with optional custom index overrides."""
    default_leg, default_waist, default_arm = _build_default_contiguous_indices(
        leg_dof=spec.leg_dof,
        waist_dof=spec.waist_dof,
        arm_dof=spec.arm_dof,
    )
    leg_indices = spec.leg_indices or default_leg
    waist_indices = spec.waist_indices or default_waist
    arm_indices = spec.arm_indices or default_arm
    return RobotJointLayout(
        robot=spec.robot,
        leg_indices=leg_indices,
        waist_indices=waist_indices,
        arm_indices=arm_indices,
        expected_leg_dof=spec.leg_dof,
        expected_waist_dof=spec.waist_dof,
        expected_arm_dof=spec.arm_dof,
    )


ROBOT_LAYOUT_SPECS: dict[str, RobotLayoutSpec] = {
    "roban_s14": RobotLayoutSpec(robot="roban_s14", leg_dof=12, waist_dof=1, arm_dof=8),
    "roban_s17": RobotLayoutSpec(robot="roban_s17", leg_dof=12, waist_dof=1, arm_dof=8),
    "kuavo_s52": RobotLayoutSpec(robot="kuavo_s52", leg_dof=12, waist_dof=1, arm_dof=14),
    "kuavo_s54": RobotLayoutSpec(robot="kuavo_s54", leg_dof=12, waist_dof=1, arm_dof=14),
    "aelos": RobotLayoutSpec(
        robot="aelos",
        leg_dof=10,
        waist_dof=0,
        arm_dof=6,
        leg_indices=tuple(range(6, 16)),
        arm_indices=tuple(range(0, 6)),
    ),
}

ROBOT_LAYOUTS: dict[str, RobotJointLayout] = {
    robot: make_robot_joint_layout(spec)
    for robot, spec in ROBOT_LAYOUT_SPECS.items()
}


def validate_robot_joint_layout(layout: RobotJointLayout) -> None:
    if len(layout.leg_indices) != layout.expected_leg_dof:
        raise ValueError(
            f"[{layout.robot}] layout error: leg indices must be "
            f"{layout.expected_leg_dof}, got {len(layout.leg_indices)}."
        )
    if len(layout.waist_indices) != layout.expected_waist_dof:
        raise ValueError(
            f"[{layout.robot}] layout error: waist indices must be "
            f"{layout.expected_waist_dof}, got {len(layout.waist_indices)}."
        )
    if len(layout.arm_indices) != layout.expected_arm_dof:
        raise ValueError(
            f"[{layout.robot}] layout error: arm indices must be "
            f"{layout.expected_arm_dof}, got {len(layout.arm_indices)}."
        )
    all_indices = layout.ordered_indices
    if len(set(all_indices)) != len(all_indices):
        raise ValueError(f"[{layout.robot}] layout error: duplicate joint indices found.")
    if min(all_indices) < 0:
        raise ValueError(f"[{layout.robot}] layout error: negative joint index found.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert GMR .pkl motion file(s) to .csv.")
    parser.add_argument(
        "--robot",
        required=True,
        choices=sorted(ROBOT_LAYOUTS.keys()),
        help="Robot model name for strict Leg/Waist/Arm DoF validation.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--pkl_file", type=Path, help="Path to one .pkl file.")
    input_group.add_argument("--folder", type=Path, help="Path to a folder containing .pkl files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path. For --pkl_file: .csv file path or output folder. "
            "For --folder: output folder. Default: sibling 'csv' directory."
        ),
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional max frame count to export from frame 0.",
    )
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    if args.pkl_file is not None:
        if not args.pkl_file.is_file():
            raise FileNotFoundError(f"PKL file not found: {args.pkl_file}")
        if args.pkl_file.suffix.lower() != ".pkl":
            raise ValueError(f"Input must be a .pkl file: {args.pkl_file}")
        return [args.pkl_file]

    folder: Path = args.folder
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder not found: {folder}")
    pkl_files = sorted(path for path in folder.iterdir() if path.suffix.lower() == ".pkl")
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in folder: {folder}")
    return pkl_files


def resolve_output_path(args: argparse.Namespace, pkl_path: Path) -> Path:
    default_csv_dir = pkl_path.parent / "csv"
    if args.output is None:
        return default_csv_dir / f"{pkl_path.stem}.csv"
    output: Path = args.output
    if args.pkl_file is not None:
        if output.suffix.lower() == ".csv":
            return output
        return output / f"{pkl_path.stem}.csv"
    return output / f"{pkl_path.stem}.csv"


def _require_key(data: dict, key: str, pkl_path: Path) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing key '{key}' in {pkl_path}")
    return np.asarray(data[key])


def load_and_validate_motion(
    pkl_path: Path,
    layout: RobotJointLayout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with pkl_path.open("rb") as file:
        motion_data = pickle.load(file)
    if not isinstance(motion_data, dict):
        raise TypeError(f"PKL content must be a dict, got {type(motion_data)} in {pkl_path}")

    root_pos = _require_key(motion_data, "root_pos", pkl_path)
    root_rot = _require_key(motion_data, "root_rot", pkl_path)
    dof_pos = _require_key(motion_data, "dof_pos", pkl_path)

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must have shape (N, 3), got {root_pos.shape} in {pkl_path}")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"root_rot must have shape (N, 4), got {root_rot.shape} in {pkl_path}")
    if dof_pos.ndim != 2:
        raise ValueError(f"dof_pos must be 2D array, got {dof_pos.shape} in {pkl_path}")

    frame_count = root_pos.shape[0]
    if root_rot.shape[0] != frame_count or dof_pos.shape[0] != frame_count:
        raise ValueError(
            "Frame count mismatch: "
            f"root_pos={root_pos.shape[0]}, root_rot={root_rot.shape[0]}, dof_pos={dof_pos.shape[0]} "
            f"in {pkl_path}"
        )

    actual_dof = int(dof_pos.shape[1])
    if actual_dof != layout.expected_total_dof:
        raise ValueError(
            f"dof_pos has {actual_dof} joints, expected {layout.expected_total_dof} for "
            f"{layout.robot} (Leg {layout.expected_leg_dof} + "
            f"Waist {layout.expected_waist_dof} + Arm {layout.expected_arm_dof})."
        )

    max_idx = max(layout.ordered_indices)
    if max_idx >= actual_dof:
        raise ValueError(
            f"[{layout.robot}] layout index out of range: max index={max_idx}, dof_dim={actual_dof}."
        )

    # Enforce output order: Leg -> Waist -> Arm
    dof_pos = dof_pos[:, layout.ordered_indices]
    return root_pos, root_rot, dof_pos


def build_csv_rows(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    max_frames: int | None,
) -> Iterable[np.ndarray]:
    frame_limit = root_pos.shape[0] if max_frames is None else min(max_frames, root_pos.shape[0])
    for frame_idx in range(frame_limit):
        yield np.concatenate((root_pos[frame_idx], root_rot_xyzw[frame_idx], dof_pos[frame_idx]))


def write_csv(csv_path: Path, rows: Iterable[np.ndarray]) -> int:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        for row in rows:
            writer.writerow(row.tolist())
            row_count += 1
    return row_count


def main() -> None:
    args = parse_args()
    layout = ROBOT_LAYOUTS[args.robot]
    validate_robot_joint_layout(layout)
    pkl_files = resolve_inputs(args)

    success_count = 0
    for pkl_path in pkl_files:
        try:
            root_pos, root_rot, dof_pos = load_and_validate_motion(
                pkl_path=pkl_path,
                layout=layout,
            )
            csv_path = resolve_output_path(args, pkl_path)
            rows = build_csv_rows(
                root_pos=root_pos,
                root_rot_xyzw=root_rot,
                dof_pos=dof_pos,
                max_frames=args.max_frames,
            )
            exported_rows = write_csv(csv_path, rows)
            print(f"[OK] {pkl_path} -> {csv_path} ({exported_rows} rows)")
            success_count += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[FAILED] {pkl_path}: {exc}")

    print(f"[DONE] Converted {success_count}/{len(pkl_files)} file(s).")
    if success_count == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
