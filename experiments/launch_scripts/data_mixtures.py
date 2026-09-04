from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from launch_scripts.data_constants import (
    SO100_SO101_MOLMOACT2,
    YAM_BIMANUAL_MOLMOACT2,
)

TAG_METADATA_BY_TAG: Dict[str, Dict[str, object]] = {}
LEROBOT_TAG_PREFIX = "lerobot:"
DEFAULT_TAG_ACTION_HORIZON = 50
DEFAULT_TAG_N_ACTION_STEPS = 25
RawMixtureEntry = Tuple[str, List[object], float]
MixtureBuilder = Callable[[], Tuple[List[RawMixtureEntry], Dict[str, Dict[str, object]]]]


def reset_tag_metadata() -> None:
    TAG_METADATA_BY_TAG.clear()


def is_lerobot_tag(tag: object) -> bool:
    return isinstance(tag, str) and tag.startswith(LEROBOT_TAG_PREFIX)


def strip_lerobot_tag_prefix(tag: str) -> str:
    normalized = str(tag).strip()
    if normalized.startswith(LEROBOT_TAG_PREFIX):
        normalized = normalized[len(LEROBOT_TAG_PREFIX):].strip()
    if not normalized:
        raise ValueError("LeRobot tag names must be non-empty after removing prefix.")
    return normalized


def _make_tag_metadata(
    *,
    action_key: str,
    state_keys: List[str],
    camera_keys: Optional[List[str]] = None,
    camera_keys_alternative: Optional[List[str]] = None,
    normalize_gripper: bool,
    action_dim: int = 32,
    action_horizon: int = DEFAULT_TAG_ACTION_HORIZON,
    n_action_steps: int = DEFAULT_TAG_N_ACTION_STEPS,
    setup_type: str = "",
    control_mode: str = "",
) -> Dict[str, object]:
    if not isinstance(normalize_gripper, bool):
        raise TypeError("normalize_gripper must be a bool.")
    action_dim = int(action_dim)
    action_horizon = int(action_horizon)
    n_action_steps = int(n_action_steps)
    if action_dim < 1:
        raise ValueError("action_dim must be >= 1.")
    if action_horizon < 1:
        raise ValueError("action_horizon must be >= 1.")
    if n_action_steps < 1:
        raise ValueError("n_action_steps must be >= 1.")
    if n_action_steps > action_horizon:
        raise ValueError(
            f"n_action_steps ({n_action_steps}) cannot exceed action_horizon ({action_horizon})."
        )
    if not isinstance(state_keys, list) or not state_keys:
        raise TypeError("state_keys must be a non-empty list of strings.")
    normalized_state_keys = [str(v) for v in state_keys]
    if any(not key for key in normalized_state_keys):
        raise ValueError("state_keys must contain only non-empty strings.")
    if camera_keys is None:
        normalized_camera_keys: List[str] = []
    elif isinstance(camera_keys, list):
        normalized_camera_keys = [str(v) for v in camera_keys]
    else:
        raise TypeError("camera_keys must be a list of strings or None.")
    if camera_keys_alternative is None:
        normalized_camera_keys_alternative: Optional[List[str]] = None
    elif isinstance(camera_keys_alternative, list):
        normalized_camera_keys_alternative = [str(v) for v in camera_keys_alternative]
    else:
        raise TypeError("camera_keys_alternative must be a list of strings or None.")

    metadata = {
        "action_key": str(action_key),
        "state_keys": normalized_state_keys,
        "camera_keys": normalized_camera_keys,
        "normalize_gripper": normalize_gripper,
        "action_dim": action_dim,
        "action_horizon": action_horizon,
        "n_action_steps": n_action_steps,
        "setup_type": str(setup_type),
        "control_mode": str(control_mode),
    }
    if normalized_camera_keys_alternative is not None:
        metadata["camera_keys_alternative"] = normalized_camera_keys_alternative
    return metadata


def _with_lerobot_prefix(value: str) -> str:
    normalized = str(value).strip()
    if normalized.startswith(LEROBOT_TAG_PREFIX):
        return normalized
    return f"{LEROBOT_TAG_PREFIX}{normalized}"


def build_single_lerobot_mixture(
    *,
    name: str,
    tag: str,
    repo_ids: Sequence[str],
    action_key: str,
    state_keys: List[str],
    camera_keys: Optional[List[str]] = None,
    camera_keys_alternative: Optional[List[str]] = None,
    normalize_gripper: bool,
    action_horizon: int,
    n_action_steps: int,
    setup_type: str,
    control_mode: str,
    action_dim: int = 32,
    rate: float = 1.0,
) -> Tuple[List[RawMixtureEntry], Dict[str, Dict[str, object]]]:
    """Build a one-tag LeRobot mixture.

    Add most new LeRobot datasets by adding a small builder that calls this helper
    and then registering it in MOLMOACT2_LEROBOT_MIXTURES below.
    """
    mixture_tag = _with_lerobot_prefix(tag)
    repos = [_with_lerobot_prefix(repo_id) for repo_id in repo_ids]
    if not repos:
        raise ValueError(f"Mixture '{name}' must include at least one repo id.")
    data_mixture = [
        (
            mixture_tag,
            repos,
            float(rate),
        ),
    ]
    metadata_per_tag = {
        mixture_tag: _make_tag_metadata(
            action_key=action_key,
            state_keys=state_keys,
            camera_keys=camera_keys,
            camera_keys_alternative=camera_keys_alternative,
            normalize_gripper=normalize_gripper,
            action_dim=action_dim,
            action_horizon=action_horizon,
            n_action_steps=n_action_steps,
            setup_type=setup_type,
            control_mode=control_mode,
        ),
    }
    return data_mixture, metadata_per_tag


def build_molmoact2_pre_post_train() -> Tuple[List[RawMixtureEntry], Dict[str, Dict[str, object]]]:
    data_mixture = [
        (
            "lerobot:yam_dual_molmoact2",
            list(YAM_BIMANUAL_MOLMOACT2),
            0.3
        ),
        (
            "lerobot:so100_so101_molmoact2",
            list(SO100_SO101_MOLMOACT2),
            0.3
        ),
        (
            "lerobot:franka_molmoact",
            [
                "lerobot:allenai/molmoact_tabletop_lerobot",
                "lerobot:allenai/molmoact_household_lerobot",
            ],
            0.025
        ),
        (
            "lerobot:franka_droid",
            [
                "lerobot:allenai/droid_lerobot"
            ],
            0.3
        ),
        (
            "lerobot:google_robot_bc_z",
            [
                "lerobot:allenai/bc_z_lerobot",
            ],
            0.025
        ),
        (
            "lerobot:google_robot_fractal",
            [
                "lerobot:allenai/fractal_lerobot",
            ],
            0.025
        ),
        (
            "lerobot:widowx_bridge",
            [
                "lerobot:allenai/bridge_lerobot",
            ],
            0.025
        ),
    ]

    metadata_per_tag = {
        "lerobot:yam_dual_molmoact2": _make_tag_metadata(
            action_key="action",
            state_keys=["observation.state"],
            camera_keys=[
                "observation.images.top",
                "observation.images.left",
                "observation.images.right",
            ],
            normalize_gripper=False,
            setup_type="bimanual yam robotic arms in molmoact2",
            control_mode="absolute joint pose",
            action_horizon=30,
            n_action_steps=30,
        ),
        "lerobot:so100_so101_molmoact2": _make_tag_metadata(
            action_key="action",
            state_keys=["observation.state"],
            normalize_gripper=True,
            setup_type="single so100/so101 robotic arm in molmoact2",
            control_mode="absolute joint pose",
            action_horizon=30,
            n_action_steps=30,
        ),
        "lerobot:franka_molmoact": _make_tag_metadata(
            action_key="action.del_ee_action",
            state_keys=["observation.state"],
            camera_keys=[
                "observation.images.primary",
                "observation.images.secondary",
            ],
            normalize_gripper=False,
            setup_type="single franka robotic arm in molmoact2",
            control_mode="delta end-effector pose",
            action_horizon=10,
            n_action_steps=10,
        ),
        "lerobot:franka_droid": _make_tag_metadata(
            action_key="action",
            state_keys=["observation.state"],
            camera_keys=[
                "observation.images.exterior_1_left",
                "observation.images.exterior_2_left",
                "observation.images.wrist_left",
            ],
            normalize_gripper=False,
            setup_type="single franka robotic arm in droid",
            control_mode="absolute joint pose",
            action_horizon=15,
            n_action_steps=15,
        ),
        "lerobot:google_robot_bc_z": _make_tag_metadata(
            action_key="action",
            state_keys=["observation.state"],
            camera_keys=[
                "observation.images.image",
            ],
            normalize_gripper=False,
            setup_type="google robot in bc_z",
            control_mode="delta end-effector pose",
            action_horizon=10,
            n_action_steps=10,
        ),
        "lerobot:google_robot_fractal": _make_tag_metadata(
            action_key="action",
            state_keys=["observation.state"],
            camera_keys=[
                "observation.images.image",
            ],
            normalize_gripper=False,
            setup_type="google robot in rt_1",
            control_mode="delta end-effector pose",
            action_horizon=3,
            n_action_steps=3,
        ),
        "lerobot:widowx_bridge": _make_tag_metadata(
            action_key="action",
            state_keys=["observation.state"],
            camera_keys=[
                "observation.images.image_0",
                "observation.images.image_1",
                "observation.images.image_2",
                "observation.images.image_3",
            ],
            normalize_gripper=False,
            setup_type="single widowx robotic arm in bridge",
            control_mode="delta end-effector pose",
            action_horizon=5,
            n_action_steps=5,
        ),
    }

    return data_mixture, metadata_per_tag


def build_molmoact2_droid() -> Tuple[List[RawMixtureEntry], Dict[str, Dict[str, object]]]:
    return build_single_lerobot_mixture(
        name="droid",
        tag="franka_droid",
        repo_ids=["allenai/droid_lerobot"],
        action_key="action",
        state_keys=["observation.state"],
        camera_keys=[
            "observation.images.exterior_1_left",
            "observation.images.exterior_2_left",
            "observation.images.wrist_left",
        ],
        normalize_gripper=False,
        setup_type="single franka robotic arm in droid",
        control_mode="absolute joint pose",
        action_horizon=15,
        n_action_steps=15,
    )


def build_molmoact2_libero() -> Tuple[List[RawMixtureEntry], Dict[str, Dict[str, object]]]:
    return build_single_lerobot_mixture(
        name="libero",
        tag="libero",
        repo_ids=["allenai/MolmoAct2-LIBERO-Dataset"],
        action_key="action",
        state_keys=["observation.state"],
        camera_keys=[
            "observation.images.image",
            "observation.images.wrist_image",
        ],
        normalize_gripper=False,
        setup_type="single franka robotic arm in libero",
        control_mode="delta end-effector pose",
        action_horizon=10,
        n_action_steps=10,
    )


def build_molmoact2_libero_goal() -> Tuple[List[RawMixtureEntry], Dict[str, Dict[str, object]]]:
    return build_single_lerobot_mixture(
        name="libero_goal",
        tag="libero",
        repo_ids=["allenai/MolmoAct2-LIBERO-Dataset"],
        action_key="action",
        state_keys=["observation.state"],
        camera_keys=[
            "observation.images.image",
            "observation.images.wrist_image",
        ],
        normalize_gripper=False,
        setup_type="single franka robotic arm in libero",
        control_mode="delta end-effector pose",
        action_horizon=10,
        n_action_steps=10,
    )


def build_molmoact2_yam() -> Tuple[List[RawMixtureEntry], Dict[str, Dict[str, object]]]:
    return build_single_lerobot_mixture(
        name="yam",
        tag="yam_dual_molmoact2",
        repo_ids=YAM_BIMANUAL_MOLMOACT2,
        action_key="action",
        state_keys=["observation.state"],
        camera_keys=[
            "observation.images.top",
            "observation.images.left",
            "observation.images.right",
        ],
        normalize_gripper=False,
        setup_type="bimanual yam robotic arms in molmoact2",
        control_mode="absolute joint pose",
        action_horizon=30,
        n_action_steps=30,
    )


def build_molmoact2_so100_so101() -> Tuple[List[RawMixtureEntry], Dict[str, Dict[str, object]]]:
    return build_single_lerobot_mixture(
        name="so100_so101",
        tag="so100_so101_molmoact2",
        repo_ids=SO100_SO101_MOLMOACT2,
        action_key="action",
        state_keys=["observation.state"],
        normalize_gripper=True,
        setup_type="single so100/so101 robotic arm in molmoact2",
        control_mode="absolute joint pose",
        action_horizon=30,
        n_action_steps=30,
    )


def build_molmoact2_vlareplica() -> Tuple[List[RawMixtureEntry], Dict[str, Dict[str, object]]]:
    """VLAReplica SO-101 data in native LeRobot v3 joint coordinates."""
    return build_single_lerobot_mixture(
        name="vlareplica",
        tag="vlareplica_so101_v3",
        repo_ids=["HenryZhang/VLAReplica_SFT_data"],
        action_key="action",
        state_keys=["observation.state"],
        camera_keys=[
            "observation.images.top",
            "observation.images.wrist",
        ],
        normalize_gripper=True,
        setup_type="single so101 robotic arm in VLAReplica tabletop workspace",
        control_mode="absolute joint pose",
        action_horizon=32,
        n_action_steps=32,
    )


# Deterministic, task-stratified 81/10/10 split (seed 50189). Every one of the
# 27 tasks is represented in all three subsets. LeRobot's ``repo@episodes``
# syntax selects episodes without copying the dataset.
VLAREPLICA_EPISODE_SPLITS = {
    "train": "0-5,7,9-10,13-31,33-41,43-44,46,48-54,56,58-62,64-68,70,72-78,80-81,83-87,89-90,92-94,96-99,101-104,106-110,112-115,117-119,121-125,127-135,137-138,140-141,143-146,148-157,159,161-162,164-171,173-179,182-187,189,191-203,205-207,210,212-219,222-223,225-234,237-242,244-250,253-269,272,274-281,283-284,286-289,292-293,295-306,309-311,313-315,317-320,322-326,328,330-331,333-339,341-343,346-349,351-355,357,359-361,363-369,372-373,375-381,383-384,386-393,395-396,398-400,402-403,405,407-412,414-426,429-431,433-448,450,453-456,458-463,466-469,472-478,480-491,493-496,498-500",
    "val": "8,12,32,42,55,63,71,82,91,105,116,120,139,147,160,163,180,190,209,220-221,224,235,252,270-271,285,294,307,316,321,332,340,350,358,362,374,385,397,401,406,428,449,451,457,465,492,497",
    "test": "6,11,45,47,57,69,79,88,95,100,111,126,136,142,158,172,181,188,204,208,211,236,243,251,273,282,290-291,308,312,327,329,344-345,356,370-371,382,394,404,413,427,432,452,464,470-471,479",
}


def _build_molmoact2_vlareplica_split(split: str):
    return build_single_lerobot_mixture(
        name=f"vlareplica_{split}",
        tag="vlareplica_so101_v3",
        repo_ids=[f"HenryZhang/VLAReplica_SFT_data@{VLAREPLICA_EPISODE_SPLITS[split]}"],
        action_key="action",
        state_keys=["observation.state"],
        camera_keys=[
            "observation.images.top",
            "observation.images.wrist",
        ],
        normalize_gripper=True,
        setup_type="single so101 robotic arm in VLAReplica tabletop workspace",
        control_mode="absolute joint pose",
        action_horizon=32,
        n_action_steps=32,
    )


def build_molmoact2_vlareplica_train():
    return _build_molmoact2_vlareplica_split("train")


def build_molmoact2_vlareplica_val():
    return _build_molmoact2_vlareplica_split("val")


def build_molmoact2_vlareplica_test():
    return _build_molmoact2_vlareplica_split("test")


MOLMOACT2_LEROBOT_MIXTURES: Dict[str, MixtureBuilder] = {
    "pre_post_train": build_molmoact2_pre_post_train,
    "droid": build_molmoact2_droid,
    "libero": build_molmoact2_libero,
    "libero_goal": build_molmoact2_libero_goal,
    "yam": build_molmoact2_yam,
    "so100_so101": build_molmoact2_so100_so101,
    "vlareplica": build_molmoact2_vlareplica,
    "vlareplica_train": build_molmoact2_vlareplica_train,
    "vlareplica_val": build_molmoact2_vlareplica_val,
    "vlareplica_test": build_molmoact2_vlareplica_test,
}
