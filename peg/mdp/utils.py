import torch


def is_inserted(peg_pos, hole_pos, threshold=0.001):
    """Check if the peg is inserted into the hole based on their positions."""
    distance = torch.norm(peg_pos - hole_pos, dim=1)
    return distance < threshold
