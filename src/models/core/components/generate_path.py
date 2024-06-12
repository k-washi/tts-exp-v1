import torch
import torch.nn.functional as F

from src.models.core.components.utils.commons import sequence_mask
def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)

    padding_shape = [[0, 0], [1, 0], [0, 0]]
    padding = [item for sublist in padding_shape[::-1] for item in sublist]

    path = path - F.pad(path, padding)[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path