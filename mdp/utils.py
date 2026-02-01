import torch
from isaaclab.assets import Articulation
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import matrix_from_quat, quat_apply


def is_inserted(peg_pos, hole_pos, threshold=0.001):
    """Check if the peg is inserted into the hole based on their positions."""
    distance = torch.norm(peg_pos - hole_pos, dim=1)
    return distance < threshold


def distance(
    n_envs: int,
    n_points: int,
    length: float,
    asset1: Articulation | FrameTransformer,
    asset2: Articulation | FrameTransformer,
    device: torch.device,
) -> torch.Tensor:
    """Compute the Euclidean distance between two assets."""
    Z = torch.tensor([0.0, 0.0, 1.0], device=device).broadcast_to(n_envs, 3)

    if isinstance(asset1, Articulation):
        pos1 = asset1.data.root_pos_w
        quat1 = asset1.data.root_quat_w
    else:
        pos1 = asset1.data.target_pos_w[:, 0, :]
        quat1 = asset1.data.target_quat_w[:, 0, :]

    if isinstance(asset2, Articulation):
        pos2 = asset2.data.root_pos_w
        quat2 = asset2.data.root_quat_w
    else:
        pos2 = asset2.data.target_pos_w[:, 0, :]
        quat2 = asset2.data.target_quat_w[:, 0, :]

    ax1 = quat_apply(quat1, Z)
    ax2 = quat_apply(quat2, Z)

    keypoints1 = []
    keypoints2 = []
    for i in range(n_points):
        keypoints1.append(pos1 + i * (length / (n_points - 1)) * ax1)
        keypoints2.append(pos2 + i * (length / (n_points - 1)) * ax2)

    keypoints1_tensor = torch.stack(keypoints1, dim=0)  # (n_points, num_envs, 3)
    keypoints2_tensor = torch.stack(keypoints2, dim=0)  # (n_points, num_envs, 3)

    return torch.norm(keypoints1_tensor - keypoints2_tensor, dim=2)
