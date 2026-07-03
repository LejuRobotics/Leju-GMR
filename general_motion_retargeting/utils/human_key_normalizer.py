# Standard library
from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

# Type aliases
VecQuatPair = Tuple[Any, Any]
HumanFrame = Dict[str, VecQuatPair]


# -----------------------------------------------------------------------------
# Alias tables
# -----------------------------------------------------------------------------

# Cross-source common aliases (short name -> canonical name).
HUMAN_KEY_ALIAS_PAIRS_COMMON = (
    ("LFootMod", "LeftFootMod"),
    ("RFootMod", "RightFootMod"),
    ("LFoot", "LeftFoot"),
    ("RFoot", "RightFoot"),
    ("LToe", "LeftToe"),
    ("RToe", "RightToe"),
    ("LToeBase", "LeftToeBase"),
    ("RToeBase", "RightToeBase"),
    ("LHand", "LeftHand"),
    ("RHand", "RightHand"),
)

# leju-source naming variants.
HUMAN_KEY_ALIAS_PAIRS_LEJU_COMMON = (
    ("Skeleton", "Hips"),
    ("Chest", "Spine"),
    ("LShoulder", "LeftShoulder"),
    ("RShoulder", "RightShoulder"),
    ("LUArm", "LeftArm"),
    ("RUArm", "RightArm"),
    ("LFArm", "LeftForeArm"),
    ("RFArm", "RightForeArm"),
    ("LThigh", "LeftUpLeg"),
    ("RThigh", "RightUpLeg"),
    ("LShin", "LeftLeg"),
    ("RShin", "RightLeg"),
    ("LFoot", "LeftFoot"),
    ("RFoot", "RightFoot"),
    ("LToe", "LeftToe"),
    ("RToe", "RightToe"),
    ("LFootMod", "LeftFootMod"),
    ("RFootMod", "RightFootMod"),
)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def _apply_alias_pairs(
    frame_data: HumanFrame,
    alias_pairs: Iterable[tuple[str, str]],
) -> HumanFrame:
    """Apply alias mapping with fill-only semantics."""
    normalized_frame_data: HumanFrame = dict(frame_data)
    for source_key, canonical_key in alias_pairs:
        if canonical_key not in normalized_frame_data and source_key in normalized_frame_data:
            normalized_frame_data[canonical_key] = normalized_frame_data[source_key]
    return normalized_frame_data


def _ensure_toe_keys_from_toebase_common(frame_data: HumanFrame) -> HumanFrame:
    """Fill LeftToe/RightToe from LeftToeBase/RightToeBase when missing."""
    normalized_frame_data: HumanFrame = dict(frame_data)
    if "LeftToe" not in normalized_frame_data and "LeftToeBase" in normalized_frame_data:
        normalized_frame_data["LeftToe"] = normalized_frame_data["LeftToeBase"]
    if "RightToe" not in normalized_frame_data and "RightToeBase" in normalized_frame_data:
        normalized_frame_data["RightToe"] = normalized_frame_data["RightToeBase"]
    return normalized_frame_data


def _ensure_toebase_keys_from_toe_common(frame_data: HumanFrame) -> HumanFrame:
    """Fill LeftToeBase/RightToeBase from LeftToe/RightToe when missing."""
    normalized_frame_data: HumanFrame = dict(frame_data)
    if "LeftToeBase" not in normalized_frame_data and "LeftToe" in normalized_frame_data:
        normalized_frame_data["LeftToeBase"] = normalized_frame_data["LeftToe"]
    if "RightToeBase" not in normalized_frame_data and "RightToe" in normalized_frame_data:
        normalized_frame_data["RightToeBase"] = normalized_frame_data["RightToe"]
    return normalized_frame_data


def _ensure_footmod_keys_common(frame_data: HumanFrame) -> HumanFrame:
    """Fill LeftFootMod/RightFootMod from (Foot position, Toe orientation)."""
    normalized_frame_data: HumanFrame = dict(frame_data)

    if (
        "LeftFootMod" not in normalized_frame_data
        and "LeftFoot" in normalized_frame_data
        and "LeftToe" in normalized_frame_data
    ):
        left_foot_position, _ = normalized_frame_data["LeftFoot"]
        _, left_toe_orientation = normalized_frame_data["LeftToe"]
        normalized_frame_data["LeftFootMod"] = (left_foot_position, left_toe_orientation)

    if (
        "RightFootMod" not in normalized_frame_data
        and "RightFoot" in normalized_frame_data
        and "RightToe" in normalized_frame_data
    ):
        right_foot_position, _ = normalized_frame_data["RightFoot"]
        _, right_toe_orientation = normalized_frame_data["RightToe"]
        normalized_frame_data["RightFootMod"] = (right_foot_position, right_toe_orientation)

    return normalized_frame_data


# -----------------------------------------------------------------------------
# Public source-level normalizers
# -----------------------------------------------------------------------------

def normalize_human_frame_keys_common(frame_data: HumanFrame) -> HumanFrame:
    """Normalize common short-name aliases to canonical keys."""
    return _apply_alias_pairs(frame_data, HUMAN_KEY_ALIAS_PAIRS_COMMON)


def normalize_human_frame_keys_leju_common(frame_data: HumanFrame) -> HumanFrame:
    """Normalize leju naming variants to canonical IK keys."""
    normalized_frame_data = normalize_human_frame_keys_common(frame_data)
    normalized_frame_data = _apply_alias_pairs(
        normalized_frame_data,
        HUMAN_KEY_ALIAS_PAIRS_LEJU_COMMON,
    )
    return normalized_frame_data


# -----------------------------------------------------------------------------
# Robot-specific normalizers (full model names in function names)
# -----------------------------------------------------------------------------

def ensure_required_human_keys_roban_s14(frame_data: HumanFrame) -> HumanFrame:
    """
    Ensure roban_s14-required canonical keys exist.

    Required:
    - LeftToe/RightToe
    - LeftFootMod/RightFootMod
    """
    normalized_frame_data = normalize_human_frame_keys_common(frame_data)
    normalized_frame_data = _ensure_toe_keys_from_toebase_common(normalized_frame_data)
    normalized_frame_data = _ensure_footmod_keys_common(normalized_frame_data)
    return normalized_frame_data


def ensure_required_human_keys_kuavo_s52(frame_data: HumanFrame) -> HumanFrame:
    """
    Ensure kuavo_s52-required canonical keys exist.

    Required:
    - LeftToe/RightToe
    - LeftFootMod/RightFootMod
    """
    normalized_frame_data = normalize_human_frame_keys_common(frame_data)
    normalized_frame_data = _ensure_toe_keys_from_toebase_common(normalized_frame_data)
    normalized_frame_data = _ensure_footmod_keys_common(normalized_frame_data)
    return normalized_frame_data


def ensure_required_human_keys_roban_s17(frame_data: HumanFrame) -> HumanFrame:
    """
    Ensure roban_s17-required canonical keys exist.

    Required:
    - LeftToeBase/RightToeBase
    - LeftFootMod/RightFootMod
    """
    normalized_frame_data = normalize_human_frame_keys_common(frame_data)
    normalized_frame_data = _ensure_toe_keys_from_toebase_common(normalized_frame_data)
    normalized_frame_data = _ensure_toebase_keys_from_toe_common(normalized_frame_data)
    normalized_frame_data = _ensure_footmod_keys_common(normalized_frame_data)
    return normalized_frame_data


def ensure_required_human_keys_aelos(frame_data: HumanFrame) -> HumanFrame:
    """
    Ensure aelos-required canonical keys exist.

    Required:
    - LeftToeBase/RightToeBase
    - LeftFootMod/RightFootMod
    """
    normalized_frame_data = normalize_human_frame_keys_common(frame_data)
    normalized_frame_data = _ensure_toe_keys_from_toebase_common(normalized_frame_data)
    normalized_frame_data = _ensure_toebase_keys_from_toe_common(normalized_frame_data)
    normalized_frame_data = _ensure_footmod_keys_common(normalized_frame_data)
    return normalized_frame_data


def ensure_required_human_keys_kuavo_s54(frame_data: HumanFrame) -> HumanFrame:
    """
    Ensure kuavo_s54-required canonical keys exist.

    Required:
    - LeftToeBase/RightToeBase
    - LeftFootMod/RightFootMod
    """
    normalized_frame_data = normalize_human_frame_keys_common(frame_data)
    normalized_frame_data = _ensure_toe_keys_from_toebase_common(normalized_frame_data)
    normalized_frame_data = _ensure_footmod_keys_common(normalized_frame_data)
    normalized_frame_data = _ensure_toebase_keys_from_toe_common(normalized_frame_data)
    return normalized_frame_data


# -----------------------------------------------------------------------------
# Optional dispatcher for simple extension
# -----------------------------------------------------------------------------

ROBOT_KEY_NORMALIZER_DICT = {
    "roban_s14": ensure_required_human_keys_roban_s14,
    "kuavo_s52": ensure_required_human_keys_kuavo_s52,
    "roban_s17": ensure_required_human_keys_roban_s17,
    "aelos": ensure_required_human_keys_aelos,
    "kuavo_s54": ensure_required_human_keys_kuavo_s54,
}


def normalize_human_frame_keys_by_robot_common(
    robot_name: str,
    frame_data: HumanFrame,
) -> HumanFrame:
    """
    Apply robot-specific key normalization when registered.

    Fallback:
    - If robot is not registered, apply common normalization only.
    """
    if robot_name in ROBOT_KEY_NORMALIZER_DICT:
        return ROBOT_KEY_NORMALIZER_DICT[robot_name](frame_data)
    return normalize_human_frame_keys_common(frame_data)


# How to add a new robot model:
# 1) Implement ensure_required_human_keys_<robot_name>(frame_data)
# 2) Reuse generic helpers; keep only robot-specific requirements
# 3) Register it in ROBOT_KEY_NORMALIZER_DICT
# 4) Call normalize_human_frame_keys_by_robot_common(...) in robot-specific loaders
