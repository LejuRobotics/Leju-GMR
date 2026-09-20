"""将 AELOS PKL 中的关节弧度转换为舵机原始值和 .src 指令

计算分为两个步骤：

    1. PKL dof_pos 关节弧度 -> 1～19号舵机原始用户值
    2. 舵机原始值 -> .src 的 MOTORA/MOTORB/MOTORC/MOTORD/MOTORE 指令

对于由关节控制的舵机：

    servo_value = servo_zero + joint_sign * joint_rad * USER_VALUE_PER_RAD

关节控制舵机的转换结果限制在 10～190
17、18、19号舵机不由 PKL dof_pos 驱动，因此保持预设零位值
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SERVO_MIN = 10
SERVO_MAX = 190
REVERSE_SERVO_SUM = 200
SRC_SPEED_MIN = 5
SRC_SPEED_MAX = 150
ADAPTIVE_SPEED_MIN = 30
ADAPTIVE_SPEED_MEDIAN = 75
ADAPTIVE_SPEED_MAX = 120
MOTION_SERVO_IDS: tuple[int, ...] = tuple(range(1, 17))

# 左右髋关节舵机（4号和12号）的最终 SRC 限位
# 两侧髋关节分别跟随各自的 PKL 关节，该限位仅提供与动作无关的实机安全范围
DEFAULT_LEG_HIP_SRC_MIN = 100
DEFAULT_LEG_HIP_SRC_MAX = 115
LEG_HIP_SERVO_IDS: tuple[int, ...] = (4, 12)

# 每条腿的5号踝关节用于补偿同侧1号髋关节
# 此处使用舵机原始编号，最终 SRC 反向由 ``src_value_from_raw_servo`` 处理
LEG_HIP_ANKLE_SERVO_PAIRS: tuple[tuple[int, int], ...] = (
    (4, 8),    # 左腿1号 -> 左腿5号
    (12, 16),  # 右腿1号 -> 右腿5号
)
LEG_ANKLE_SERVO_IDS: frozenset[int] = frozenset(
    ankle_id for _, ankle_id in LEG_HIP_ANKLE_SERVO_PAIRS
)
LEG_ANKLE_RAD_MIN = -0.35
LEG_ANKLE_RAD_MAX = 0.35

REVERSE_MAP: tuple[int, ...] = (
    0,
    0,
    0,
    1,
    0,
    1,
    1,
    0,
    1,
    1,
    1,
    0,
    1,
    0,
    0,
    1,
    0,
    0,
    0,
)

# 用户提供的舵机原始零位值，下标为舵机编号减1
SERVO_ZERO_BY_ID: tuple[int, ...] = (
    100,  # 舵机1
    10,   # 舵机2
    100,  # 舵机3
    100,  # 舵机4
    115,  # 舵机5
    55,   # 舵机6
    140,  # 舵机7
    100,  # 舵机8
    100,  # 舵机9
    190,  # 舵机10
    100,  # 舵机11
    100,  # 舵机12
    85,   # 舵机13
    145,  # 舵机14
    60,   # 舵机15
    100,  # 舵机16
    0,    # 舵机17，左夹爪
    0,    # 舵机18，右夹爪
    100,  # 舵机19，头部
)

SRC_SERVO_ORDER_BY_COMMAND: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("MOTORA", (8, 7, 6, 5, 4)),
    ("MOTORB", (16, 15, 14, 13, 12)),
    ("MOTORC", (3, 2, 1)),
    ("MOTORD", (11, 10, 9)),
    ("MOTORE", (19, 18, 17)),
)

DEFAULT_RIGID_BY_GROUP: dict[str, tuple[int, ...]] = {
    "A": (70, 70, 70, 70, 70),
    "B": (70, 70, 70, 70, 70),
    "C": (65, 65, 65),
    "D": (65, 65, 65),
    "E": (0, 0, 0),
}

# AELOS PKL 的 dof_pos 顺序
JOINT_NAMES: tuple[str, ...] = (
    "jointRarm1",
    "jointRarm2",
    "jointRarm3",
    "jointLarm1",
    "jointLarm2",
    "jointLarm3",
    "jointRleg1",
    "jointRleg2",
    "jointRleg3",
    "jointRleg4",
    "jointRleg5",
    "jointLleg1",
    "jointLleg2",
    "jointLleg3",
    "jointLleg4",
    "jointLleg5",
)

# PKL 关节到实体舵机编号的映射，此处为舵机原始顺序而非 .src 顺序
JOINT_TO_SERVO_ID: dict[str, int] = {
    "jointLarm3": 1,
    "jointLarm2": 2,
    "jointLarm1": 3,
    "jointLleg1": 4,
    "jointLleg2": 5,
    "jointLleg3": 6,
    "jointLleg4": 7,
    "jointLleg5": 8,
    "jointRarm3": 9,
    "jointRarm2": 10,
    "jointRarm1": 11,
    "jointRleg1": 12,
    "jointRleg2": 13,
    "jointRleg3": 14,
    "jointRleg4": 15,
    "jointRleg5": 16,
}

# 无法通过零位姿态判断转动方向，以下符号根据实机舵机方向标定
JOINT_SIGN_BY_NAME: dict[str, float] = {
    joint_name: -1.0
    if JOINT_TO_SERVO_ID[joint_name] in {3, 5, 8, 9, 10, 14, 15, 16}
    else 1.0
    for joint_name in JOINT_NAMES
}

# 腿部2～4号关节使用站立姿态的舵机原始值
# 只有腿部1号关节跟随重定向后的 PKL 轨迹，5号关节与1号关节联动
# 以下数值由经过实机测试的站立姿态反向换算得到
#   MOTORA,*,75,145,95,*  （舵机8、7、6、5、4）
#   MOTORB,*,75,145,95,*  （舵机16、15、14、13、12）
FIXED_LEG_SERVO_RAW_VALUES: dict[int, int] = {
    5: 95,    # jointLleg2
    6: 55,    # jointLleg3
    7: 125,   # jointLleg4
    13: 105,  # jointRleg2
    14: 145,  # jointRleg3
    15: 75,   # jointRleg4
}

# 换算依据
# angle_deg = ((user_val - 10) * 18.9718 + 340) / 4095 * 230
# 1个用户值 = 18.9718 / 4095 * 230度
USER_VALUE_PER_RAD = 4095.0 / (18.9718 * 230.0) * 180.0 / math.pi


@dataclass(frozen=True)
class JointServoMapping:
    joint_name: str
    joint_index: int
    servo_id: int
    servo_zero: int
    joint_sign: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert AELOS PKL dof_pos joint radians to raw servo values."
    )
    parser.add_argument("--pkl_file", type=Path, required=True, help="Input AELOS GMR .pkl file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output .src file. Default: src/ next to the pkl/ directory "
            "(e.g. output/aelos/pkl/foo.pkl -> output/aelos/src/foo.src)."
        ),
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional CSV output. Columns are frame, servo_1, ..., servo_19.",
    )
    parser.add_argument(
        "--print_frames",
        type=int,
        default=1,
        help="Number of converted frames to print for inspection.",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Write every Nth PKL frame. Default: 1.",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=None,
        help=(
            "Use one fixed SRC SPEED value for every frame. If omitted, SPEED "
            "is mapped adaptively from frame displacement to 30..120, with the "
            "non-zero median displacement mapped to 75."
        ),
    )
    parser.add_argument(
        "--delay_ms",
        type=int,
        default=None,
        help="DELAY in milliseconds between frames. "
        "Defaults to auto-detect from PKL frame_time (recommended). "
        "Use 0 to disable DELAY lines entirely.",
    )
    parser.add_argument(
        "--keep_wait",
        action="store_true",
        help="Write WAIT after every motion frame (not recommended for smooth motion).",
    )
    parser.add_argument(
        "--leg_hip_min",
        type=int,
        default=DEFAULT_LEG_HIP_SRC_MIN,
        help=(
            "Minimum final SRC value applied independently to both hip servos. "
            f"Default: {DEFAULT_LEG_HIP_SRC_MIN}."
        ),
    )
    parser.add_argument(
        "--leg_hip_max",
        type=int,
        default=DEFAULT_LEG_HIP_SRC_MAX,
        help=(
            "Maximum final SRC value applied independently to both hip servos. "
            f"Default: {DEFAULT_LEG_HIP_SRC_MAX}."
        ),
    )
    return parser.parse_args()


def clamp_joint_servo_value(value: float) -> int:
    rounded = int(round(value))
    return max(SERVO_MIN, min(SERVO_MAX, rounded))


def validate_leg_hip_src_limits(min_value: int, max_value: int) -> None:
    if not SERVO_MIN <= min_value <= max_value <= SERVO_MAX:
        raise ValueError(
            "Leg hip SRC limits must satisfy "
            f"{SERVO_MIN} <= min <= max <= {SERVO_MAX}, got "
            f"min={min_value}, max={max_value}."
        )


def src_value_from_raw_servo(servo_id: int, raw_value: int) -> int:
    if REVERSE_MAP[servo_id - 1]:
        return REVERSE_SERVO_SUM - raw_value
    return raw_value


def raw_value_from_src_servo(servo_id: int, src_value: int) -> int:
    """对单个舵机执行 ``src_value_from_raw_servo`` 的逆转换"""
    if REVERSE_MAP[servo_id - 1]:
        return REVERSE_SERVO_SUM - src_value
    return src_value


def install_numpy_pickle_compat() -> None:
    """允许旧版 NumPy 1.x 环境加载由 NumPy 2.x 生成的 pickle"""
    try:
        import numpy.core as numpy_core  # type: ignore[import-not-found]
    except Exception:
        return

    sys.modules.setdefault("numpy._core", numpy_core)
    for module_name in ("multiarray", "numeric", "umath", "_multiarray_umath"):
        try:
            module = importlib.import_module(f"numpy.core.{module_name}")
        except Exception:
            continue
        sys.modules.setdefault(f"numpy._core.{module_name}", module)


def load_motion_data(pkl_file: Path) -> dict[str, Any]:
    install_numpy_pickle_compat()
    with pkl_file.open("rb") as file:
        data = pickle.load(file)

    if not isinstance(data, dict):
        raise TypeError(f"PKL content must be a dict, got {type(data)}.")
    if "dof_pos" not in data:
        raise KeyError("PKL is missing required key 'dof_pos'.")
    return data


def as_frame_list(array_like: Any) -> list[list[float]]:
    frames: list[list[float]] = []
    for row in array_like:
        if hasattr(row, "tolist"):
            row = row.tolist()
        frames.append([float(value) for value in row])
    return frames


def build_joint_servo_mappings() -> list[JointServoMapping]:
    mappings: list[JointServoMapping] = []
    for joint_index, joint_name in enumerate(JOINT_NAMES):
        servo_id = JOINT_TO_SERVO_ID[joint_name]
        mappings.append(
            JointServoMapping(
                joint_name=joint_name,
                joint_index=joint_index,
                servo_id=servo_id,
                servo_zero=SERVO_ZERO_BY_ID[servo_id - 1],
                joint_sign=JOINT_SIGN_BY_NAME[joint_name],
            )
        )
    return mappings


def validate_input(dof_frames: list[list[float]]) -> None:
    if len(SERVO_ZERO_BY_ID) != 19:
        raise ValueError(f"SERVO_ZERO_BY_ID must contain 19 values, got {len(SERVO_ZERO_BY_ID)}.")
    if len(REVERSE_MAP) != 19:
        raise ValueError(f"REVERSE_MAP must contain 19 values, got {len(REVERSE_MAP)}.")
    if not dof_frames:
        raise ValueError("dof_pos has no frames.")

    expected_joint_count = len(JOINT_NAMES)
    for frame_index, frame in enumerate(dof_frames):
        if len(frame) != expected_joint_count:
            raise ValueError(
                f"dof_pos frame {frame_index} has {len(frame)} joints, "
                f"expected {expected_joint_count}."
            )


def joint_rad_to_servo_value(mapping: JointServoMapping, joint_rad: float) -> int:
    raw_value = mapping.servo_zero + mapping.joint_sign * joint_rad * USER_VALUE_PER_RAD
    return clamp_joint_servo_value(raw_value)


def servo_value_to_joint_rad(mapping: JointServoMapping, raw_value: int) -> float:
    """还原舵机原始值所表示的有效关节角"""
    return (
        (raw_value - mapping.servo_zero)
        / mapping.joint_sign
        / USER_VALUE_PER_RAD
    )


def frame_to_servo_values(
    dof_frame: list[float],
    mappings: list[JointServoMapping],
    leg_hip_min: int = DEFAULT_LEG_HIP_SRC_MIN,
    leg_hip_max: int = DEFAULT_LEG_HIP_SRC_MAX,
) -> list[int]:
    servo_values = list(SERVO_ZERO_BY_ID)
    for mapping in mappings:
        if (
            mapping.servo_id in FIXED_LEG_SERVO_RAW_VALUES
            or mapping.servo_id in LEG_ANKLE_SERVO_IDS
        ):
            continue
        joint_rad = dof_frame[mapping.joint_index]
        servo_values[mapping.servo_id - 1] = joint_rad_to_servo_value(mapping, joint_rad)

    # 腿部2～4号关节保持经过实机测试的站立姿态
    for servo_id, raw_value in FIXED_LEG_SERVO_RAW_VALUES.items():
        servo_values[servo_id - 1] = raw_value

    # 左右髋关节分别跟随各自的 PKL 关节，并在最终 SRC 数值域中独立执行安全限位
    for servo_id in LEG_HIP_SERVO_IDS:
        src_value = src_value_from_raw_servo(servo_id, servo_values[servo_id - 1])
        limited_src_value = max(leg_hip_min, min(leg_hip_max, src_value))
        servo_values[servo_id - 1] = raw_value_from_src_servo(
            servo_id, limited_src_value
        )

    # 腿部5号关节不使用 PKL 角度，而是根据同侧已限幅的1号关节计算
    # 使用 q5=-q1 使两个滚转角相互抵消
    # 必须在髋关节限幅后再计算，否则限幅后的髋关节与未限幅的踝关节无法保持配对
    mappings_by_servo = {mapping.servo_id: mapping for mapping in mappings}
    for hip_id, ankle_id in LEG_HIP_ANKLE_SERVO_PAIRS:
        hip_mapping = mappings_by_servo[hip_id]
        ankle_mapping = mappings_by_servo[ankle_id]
        hip_rad = servo_value_to_joint_rad(
            hip_mapping, servo_values[hip_id - 1]
        )
        ankle_rad = max(
            LEG_ANKLE_RAD_MIN,
            min(LEG_ANKLE_RAD_MAX, -hip_rad),
        )
        servo_values[ankle_id - 1] = joint_rad_to_servo_value(
            ankle_mapping, ankle_rad
        )
    return servo_values


def sampled_frame_indices(frame_count: int, frame_stride: int) -> list[int]:
    if frame_stride <= 0:
        raise ValueError("--frame_stride must be > 0.")
    return list(range(0, frame_count, frame_stride))


def convert_frames_to_servo_values(
    dof_frames: list[list[float]],
    frame_indices: list[int],
    leg_hip_min: int = DEFAULT_LEG_HIP_SRC_MIN,
    leg_hip_max: int = DEFAULT_LEG_HIP_SRC_MAX,
) -> list[tuple[int, list[int]]]:
    validate_leg_hip_src_limits(leg_hip_min, leg_hip_max)
    mappings = build_joint_servo_mappings()
    return [
        (
            frame_index,
            frame_to_servo_values(
                dof_frames[frame_index],
                mappings,
                leg_hip_min=leg_hip_min,
                leg_hip_max=leg_hip_max,
            ),
        )
        for frame_index in frame_indices
    ]


def write_servo_csv(output_csv: Path, servo_frames: list[tuple[int, list[int]]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", *(f"servo_{servo_id}" for servo_id in range(1, 20))])
        for frame_index, servo_values in servo_frames:
            writer.writerow([frame_index, *servo_values])


def print_servo_frames(servo_frames: list[tuple[int, list[int]]], print_frames: int) -> None:
    count = min(max(0, print_frames), len(servo_frames))
    for sampled_index in range(count):
        frame_index, servo_values = servo_frames[sampled_index]
        values_text = " ".join(str(value) for value in servo_values)
        print(f"frame {frame_index}: {values_text}")


def metadata_float(motion_data: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in motion_data:
            continue
        raw_value = motion_data[key]
        if hasattr(raw_value, "item"):
            raw_value = raw_value.item()
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def resolve_delay_ms(args: argparse.Namespace, motion_data: dict[str, Any]) -> int | None:
    if args.delay_ms is not None:
        if args.delay_ms < 0:
            raise ValueError("--delay_ms must be >= 0.")
        if args.delay_ms == 0:
            return None
        return int(args.delay_ms)

    frame_time = metadata_float(motion_data, ("frame_time",))
    if frame_time is None:
        return None

    return int(math.floor(frame_time * 1000.0 * args.frame_stride))


def format_src_command(command_name: str, values: tuple[int, ...] | list[int]) -> str:
    return f"{command_name}," + ",".join(str(value) for value in values)


def src_commands_for_servo_values(servo_values: list[int]) -> list[str]:
    lines: list[str] = []
    for command_name, servo_ids in SRC_SERVO_ORDER_BY_COMMAND:
        src_values = [
            src_value_from_raw_servo(servo_id, servo_values[servo_id - 1])
            for servo_id in servo_ids
        ]
        lines.append(format_src_command(command_name, src_values))
    return lines


def frame_displacements(
    servo_frames: list[tuple[int, list[int]]],
) -> list[int]:
    """返回每帧相对上一帧的最大最终 SRC 位移量"""
    if not servo_frames:
        return []

    displacements = [0]
    for (_, previous_values), (_, current_values) in zip(
        servo_frames,
        servo_frames[1:],
    ):
        displacement = max(
            abs(
                src_value_from_raw_servo(
                    servo_id, current_values[servo_id - 1]
                )
                - src_value_from_raw_servo(
                    servo_id, previous_values[servo_id - 1]
                )
            )
            for servo_id in MOTION_SERVO_IDS
        )
        displacements.append(displacement)
    return displacements


def map_displacement_to_speed(
    displacement: int,
    minimum: float,
    median: float,
    maximum: float,
) -> int:
    """将最小、中位和最大位移量分段线性映射到30、75和120"""
    if displacement <= 0:
        return ADAPTIVE_SPEED_MIN

    if minimum == maximum:
        return ADAPTIVE_SPEED_MEDIAN

    if displacement <= median:
        if median == minimum:
            return ADAPTIVE_SPEED_MEDIAN
        ratio = (displacement - minimum) / (median - minimum)
        speed = ADAPTIVE_SPEED_MIN + ratio * (
            ADAPTIVE_SPEED_MEDIAN - ADAPTIVE_SPEED_MIN
        )
    else:
        if maximum == median:
            return ADAPTIVE_SPEED_MEDIAN
        ratio = (displacement - median) / (maximum - median)
        speed = ADAPTIVE_SPEED_MEDIAN + ratio * (
            ADAPTIVE_SPEED_MAX - ADAPTIVE_SPEED_MEDIAN
        )

    return max(
        ADAPTIVE_SPEED_MIN,
        min(ADAPTIVE_SPEED_MAX, int(math.floor(speed + 0.5))),
    )


def adaptive_frame_speeds(
    servo_frames: list[tuple[int, list[int]]],
) -> tuple[list[int], tuple[float, float, float] | None]:
    """生成自适应速度并返回非零位移量的最小值、中位数和最大值"""
    displacements = frame_displacements(servo_frames)
    nonzero = sorted(value for value in displacements[1:] if value > 0)
    if not nonzero:
        return [ADAPTIVE_SPEED_MIN] * len(displacements), None

    minimum = float(nonzero[0])
    maximum = float(nonzero[-1])
    middle_index = len(nonzero) // 2
    if len(nonzero) % 2:
        median = float(nonzero[middle_index])
    else:
        median = (nonzero[middle_index - 1] + nonzero[middle_index]) / 2.0

    speeds = [
        map_displacement_to_speed(
            displacement,
            minimum=minimum,
            median=median,
            maximum=maximum,
        )
        for displacement in displacements
    ]
    return speeds, (minimum, median, maximum)


def resolve_frame_speeds(
    fixed_speed: int | None,
    servo_frames: list[tuple[int, list[int]]],
) -> tuple[list[int], tuple[float, float, float] | None]:
    if fixed_speed is None:
        return adaptive_frame_speeds(servo_frames)
    if not SRC_SPEED_MIN <= fixed_speed <= SRC_SPEED_MAX:
        raise ValueError(
            f"--speed must be in {SRC_SPEED_MIN}..{SRC_SPEED_MAX}, got {fixed_speed}."
        )
    return [fixed_speed] * len(servo_frames), None


def build_src_lines(
    servo_frames: list[tuple[int, list[int]]],
    frame_speeds: list[int],
    delay_ms: int | None,
    keep_wait: bool,
) -> list[str]:
    if len(frame_speeds) != len(servo_frames):
        raise ValueError(
            "frame_speeds and servo_frames must contain the same number of frames."
        )
    lines = [
        format_src_command("RIGIDA", DEFAULT_RIGID_BY_GROUP["A"]),
        format_src_command("RIGIDB", DEFAULT_RIGID_BY_GROUP["B"]),
        format_src_command("RIGIDC", DEFAULT_RIGID_BY_GROUP["C"]),
        format_src_command("RIGIDD", DEFAULT_RIGID_BY_GROUP["D"]),
        format_src_command("RIGIDE", DEFAULT_RIGID_BY_GROUP["E"]),
        "RIGEND",
        "",
        "",
    ]

    for (frame_index, servo_values), speed in zip(servo_frames, frame_speeds):
        lines.append(f"// frame {frame_index}")
        lines.append(f"SPEED {speed}")
        lines.extend(src_commands_for_servo_values(servo_values))
        if keep_wait:
            lines.append("WAIT")
        if delay_ms is not None:
            lines.append(f"DELAY {delay_ms}")
        lines.append("")

    return lines


def write_src_file(output: Path, lines: list[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def default_src_path(pkl_path: Path) -> Path:
    """Put src beside the pkl directory, not inside it."""
    parent = pkl_path.parent
    if parent.name == "pkl":
        return parent.parent / "src" / f"{pkl_path.stem}.src"
    return parent / "src" / f"{pkl_path.stem}.src"


def main() -> None:
    args = parse_args()
    motion_data = load_motion_data(args.pkl_file)
    dof_frames = as_frame_list(motion_data["dof_pos"])
    validate_input(dof_frames)

    frame_indices = sampled_frame_indices(len(dof_frames), args.frame_stride)
    servo_frames = convert_frames_to_servo_values(
        dof_frames,
        frame_indices,
        leg_hip_min=args.leg_hip_min,
        leg_hip_max=args.leg_hip_max,
    )
    frame_speeds, adaptive_speed_stats = resolve_frame_speeds(
        args.speed,
        servo_frames,
    )
    delay_ms = resolve_delay_ms(args, motion_data)

    print(f"[OK] loaded {len(dof_frames)} frames from {args.pkl_file}")
    print(f"[OK] sampled {len(servo_frames)} frames with frame_stride={args.frame_stride}")
    print(f"[OK] USER_VALUE_PER_RAD = {USER_VALUE_PER_RAD:.6f}")
    print(
        f"[OK] independent leg hip SRC limit = "
        f"{args.leg_hip_min}..{args.leg_hip_max}"
    )
    if args.speed is None:
        if adaptive_speed_stats is None:
            print(f"[OK] adaptive SPEED = {ADAPTIVE_SPEED_MIN} (no moving transitions)")
        else:
            minimum, median, maximum = adaptive_speed_stats
            print(
                f"[OK] adaptive SPEED = {ADAPTIVE_SPEED_MIN}/{ADAPTIVE_SPEED_MEDIAN}/"
                f"{ADAPTIVE_SPEED_MAX} for displacement min/median/max = "
                f"{minimum:g}/{median:g}/{maximum:g}"
            )
    else:
        print(f"[OK] fixed SPEED = {args.speed}")
    if delay_ms is not None:
        print(f"[OK] DELAY = {delay_ms} ms (auto-detected from PKL frame_time)")
    else:
        print(f"[OK] DELAY = disabled")

    wait_info = "ON" if args.keep_wait else "OFF"
    print(f"[OK] WAIT = {wait_info}")

    print_servo_frames(servo_frames, args.print_frames)

    output_src = args.output if args.output is not None else default_src_path(args.pkl_file)

    if args.output_csv is not None:
        write_servo_csv(args.output_csv, servo_frames)
        print(f"[OK] wrote raw servo CSV: {args.output_csv}")

    src_lines = build_src_lines(
        servo_frames,
        frame_speeds,
        delay_ms=delay_ms,
        keep_wait=args.keep_wait,
    )
    write_src_file(output_src, src_lines)
    print(f"[OK] wrote SRC: {output_src}")


if __name__ == "__main__":
    main()
