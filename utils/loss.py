import torch
import torch.nn.functional as F


def reconstruction_loss(prediction, target):
    return F.mse_loss(prediction, target)


def cluster_aggregation_loss(point_features, candidate_periodic_points, candidate_weights):
    if candidate_periodic_points is None or candidate_weights is None:
        return point_features.new_tensor(0.0)
    if point_features.dim() not in {2, 3}:
        raise ValueError(f"point_features expects [B,d] or [B,S,d], got {tuple(point_features.shape)}")
    if candidate_periodic_points.shape[:-2] != point_features.shape[:-1]:
        raise ValueError(
            "candidate_periodic_points must match point_features leading shape: "
            f"got {tuple(candidate_periodic_points.shape)} and {tuple(point_features.shape)}"
        )
    if candidate_weights.shape != candidate_periodic_points.shape[:-1]:
        raise ValueError(
            "candidate_weights must match candidate_periodic_points without feature dim: "
            f"got {tuple(candidate_weights.shape)} and {tuple(candidate_periodic_points.shape)}"
        )

    squared_distances = (candidate_periodic_points - point_features.unsqueeze(-2)).pow(2).sum(dim=-1)
    return (candidate_weights * squared_distances).sum(dim=-1).mean()


def training_loss(
    local_loss,
    periodic_point_loss,
    consistency_loss,
    cluster_loss,
    cluster_loss_weight=1.0,
):
    return (
        local_loss
        + periodic_point_loss
        + consistency_loss
        + float(cluster_loss_weight) * cluster_loss
    )
