import torch
import torch.nn.functional as F


def _full_lengths(batch_size, length, device):
    return torch.full((batch_size,), length, dtype=torch.long, device=device)


def _rescale_lengths(lengths, source_length, target_length):
    if lengths is None:
        return None
    lengths = lengths.to(dtype=torch.float32)
    if source_length <= 0:
        return torch.zeros_like(lengths, dtype=torch.long)
    scaled = torch.ceil(lengths * float(target_length) / float(source_length))
    return scaled.to(dtype=torch.long).clamp(min=0, max=target_length)


def _sequence_mask(lengths, max_length):
    steps = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return steps < lengths.unsqueeze(1)


def masked_feature_mse_loss(
    student_feats,
    teacher_feats,
    student_lengths=None,
    teacher_lengths=None,
):
    """
    Compute feature MSE on valid temporal positions only.

    Args:
        student_feats: Tensor shaped [B, T_s, D].
        teacher_feats: Tensor shaped [B, T_t, D].
        student_lengths: Optional valid lengths in the student time scale.
        teacher_lengths: Optional valid lengths in the original teacher time scale.
    """
    if student_feats.dim() != 3 or teacher_feats.dim() != 3:
        raise ValueError("student_feats and teacher_feats must both be [B, T, D] tensors")
    if student_feats.size(0) != teacher_feats.size(0):
        raise ValueError("student_feats and teacher_feats must have the same batch size")
    if student_feats.size(2) != teacher_feats.size(2):
        raise ValueError("student_feats and teacher_feats must have the same feature dim")

    batch_size, student_time, _ = student_feats.shape
    teacher_time = teacher_feats.size(1)

    if teacher_time != student_time:
        teacher_feats = F.interpolate(
            teacher_feats.transpose(1, 2),
            size=student_time,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        teacher_lengths = _rescale_lengths(teacher_lengths, teacher_time, student_time)

    if student_lengths is None:
        student_lengths = _full_lengths(batch_size, student_time, student_feats.device)
    else:
        student_lengths = student_lengths.to(device=student_feats.device, dtype=torch.long)
        student_lengths = student_lengths.clamp(min=0, max=student_time)

    if teacher_lengths is None:
        teacher_lengths = _full_lengths(batch_size, student_time, student_feats.device)
    else:
        teacher_lengths = teacher_lengths.to(device=student_feats.device, dtype=torch.long)
        teacher_lengths = teacher_lengths.clamp(min=0, max=student_time)

    valid_mask = _sequence_mask(student_lengths, student_time) & _sequence_mask(teacher_lengths, student_time)
    valid_mask = valid_mask.unsqueeze(-1).to(dtype=student_feats.dtype)
    valid_count = valid_mask.sum() * student_feats.size(2)

    if valid_count.item() == 0:
        return (student_feats.sum() + teacher_feats.sum()) * 0.0

    squared_error = (student_feats - teacher_feats).pow(2) * valid_mask
    return squared_error.sum() / valid_count
