import torch


def target_from_window(x):
    return x[:, -1, :]


def score_from_reconstruction(xhat_local, xhat_periodic, top_beta=1):
    diff = (xhat_local - xhat_periodic) ** 2
    if diff.dim() != 2:
        raise ValueError(f"Expected sensor-wise score tensor [B,S], got {tuple(diff.shape)}")
    top_k = min(max(1, int(top_beta)), int(diff.size(-1)))
    return torch.topk(diff, k=top_k, dim=-1, largest=True, sorted=False).values.mean(dim=-1)


def embedding_std(z):
    z = z.detach().reshape(-1, z.size(-1))
    if z.size(0) <= 1:
        return 0.0
    return float(z.std(dim=0, unbiased=False).mean().item())
