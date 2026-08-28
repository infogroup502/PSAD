import numpy as np
import torch


def print_periodic_point_initialization(model):
    memory = model.periodic_point_memory
    periodic_points = memory.periodic_points.detach()
    print("\n========== Periodic-point initialization ==========")
    print(
        "periodic_point_init=normal(0,0.02) | "
        f"shape={tuple(periodic_points.shape)}"
    )


def print_periodic_point_usage_diagnostics(
    model,
    train_loader,
    device,
    center_count,
    make_local_view,
    progress_bar=None,
):
    memory = model.periodic_point_memory
    center_count = int(getattr(memory, "center_count", center_count))
    candidate_count = int(getattr(memory, "candidate_count", 1))
    model.eval()

    print("\n========== Periodic-point Usage Diagnostics ==========")
    counts = torch.zeros(center_count, dtype=torch.long)
    weight_sum = torch.zeros(candidate_count, dtype=torch.float64)
    max_weight_sum = 0.0
    entropy_sum = 0.0
    hard_count = 0
    uniform_count = 0
    weight_count = 0

    iterable = train_loader
    if progress_bar is not None:
        iterable = progress_bar(
            train_loader,
            desc="Periodic-point usage train",
            leave=True,
            dynamic_ncols=True,
        )

    with torch.no_grad():
        for input_data, _ in iterable:
            x = input_data.float().to(device)
            x_local = make_local_view(x)
            z_local = model.encode_local(x_local)
            _, nearest_idx, _ = memory(z_local)

            nearest_idx = nearest_idx.detach().cpu().reshape(-1)
            counts += torch.bincount(nearest_idx, minlength=center_count)

            top_weights = getattr(memory, "last_candidate_weights", None)
            if top_weights is not None:
                top_weights = top_weights.detach().cpu().reshape(-1, candidate_count)
                max_weights = top_weights.max(dim=-1).values
                entropies = -(
                    top_weights * torch.log(top_weights.clamp_min(1e-12))
                ).sum(dim=-1)

                weight_sum += top_weights.sum(dim=0).double()
                max_weight_sum += float(max_weights.sum().item())
                entropy_sum += float(entropies.sum().item())
                hard_count += int((max_weights > 0.90).sum().item())
                uniform_count += int((max_weights < 0.45).sum().item())
                weight_count += int(top_weights.size(0))

    total = int(counts.sum().item())
    used = int((counts > 0).sum().item())
    dead = int(center_count - used)

    if total > 0:
        top_values = torch.topk(counts, k=min(5, center_count)).values
        top1_share = float(counts.max().item() / total)
        top5_share = float(top_values.sum().item() / total)
    else:
        top1_share = 0.0
        top5_share = 0.0

    print(
        f"periodic-point memory: used {used}/{center_count}, dead {dead}, "
        f"top1 {top1_share:.4f}, top5 {top5_share:.4f}"
    )

    if weight_count > 0:
        mean_weights = weight_sum / float(weight_count)
        weights_csv = ",".join(f"{float(value):.3f}" for value in mean_weights.tolist())
        print(
            f"periodic-point top-k weights: k={candidate_count}, w=[{weights_csv}], "
            f"w_max={max_weight_sum / weight_count:.4f}, "
            f"w_entropy={entropy_sum / weight_count:.4f}, "
            f"hard_weight>0.9 {hard_count / weight_count:.4f}, "
            f"uniform_weight<0.45 {uniform_count / weight_count:.4f}"
        )
