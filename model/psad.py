import math

import torch
import torch.nn as nn


from model.spatiotemporal_encoder import SpatioTemporalViewEncoder


def normalize_candidate_count(candidate_count, center_count):
    return min(max(1, int(candidate_count)), int(center_count))


class SharedPeriodicPointMemory(nn.Module):
    def __init__(
        self,
        center_count=50,
        embed_dim=128,
        candidate_count=3,
    ):
        super().__init__()
        self.center_count = int(center_count)
        self.embed_dim = int(embed_dim)
        self.candidate_count = normalize_candidate_count(candidate_count, self.center_count)
        if self.center_count <= 0:
            raise ValueError("center_count must be > 0")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")

        self.periodic_points = nn.Parameter(
            torch.empty(self.center_count, self.embed_dim)
        )
        self.periodic_point_fusion = nn.Linear(
            self.candidate_count * self.embed_dim,
            self.embed_dim,
        )
        nn.init.normal_(self.periodic_points, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.periodic_point_fusion.weight)
        nn.init.zeros_(self.periodic_point_fusion.bias)
        self.last_topk_indices = None
        self.last_topk_distances = None
        self.last_topk_weights = None
        self.last_topk_periodic_points = None
        self.last_fused_periodic_point = None

    def forward(self, z):
        if z.dim() not in {2, 3}:
            raise ValueError(f"Periodic-point memory expects [B,d] or [B,S,d], got {tuple(z.shape)}")
        embed_dim = z.shape[-1]
        if int(embed_dim) != self.embed_dim:
            raise ValueError(f"Expected d={self.embed_dim}, got d={int(embed_dim)}")

        if z.dim() == 2:
            z_norm = (z ** 2).sum(dim=-1, keepdim=True)
            point_norm = (self.periodic_points ** 2).sum(dim=-1).view(1, -1)
            dot = torch.matmul(z, self.periodic_points.t())
            distances = z_norm + point_norm - 2.0 * dot
        else:
            z_norm = (z ** 2).sum(dim=-1, keepdim=True)
            point_norm = (self.periodic_points ** 2).sum(dim=-1).view(1, 1, -1)
            dot = torch.einsum("bsd,kd->bsk", z, self.periodic_points)
            distances = z_norm + point_norm - 2.0 * dot

        top_distances, top_indices = torch.topk(
            distances,
            k=int(self.candidate_count),
            dim=-1,
            largest=False,
            sorted=True,
        )
        top_periodic_points = self.periodic_points[top_indices]
        leading_shape = top_periodic_points.shape[:-2]
        fused_input = top_periodic_points.reshape(
            *leading_shape,
            int(self.candidate_count) * int(self.embed_dim),
        )
        z_periodic = self.periodic_point_fusion(fused_input)

        similarity_logits = (top_periodic_points * z_periodic.unsqueeze(-2)).sum(dim=-1)
        similarity_logits = similarity_logits / math.sqrt(float(self.embed_dim))
        top_weights = torch.softmax(similarity_logits, dim=-1)

        nearest_idx = top_indices[..., 0]
        min_distances = top_distances[..., 0]
        self.last_topk_indices = top_indices.detach()
        self.last_topk_distances = top_distances.detach()
        self.last_candidate_weights = top_weights
        self.last_candidate_points = top_periodic_points
        self.last_fused_periodic_point = z_periodic
        return z_periodic, nearest_idx, min_distances


class PSADModel(nn.Module):
    def __init__(
        self,
        sensor_count,
        embed_dim=128,
        temporal_length=20,
        dropout=0.1,
        recon_hidden_dim=0,
        center_count=50,
        candidate_count=3,
        conv_channels=0,
    ):
        super().__init__()
        sensor_count = int(sensor_count)
        embed_dim = int(embed_dim)
        self.temporal_length = int(temporal_length)
        self.sensor_count = sensor_count
        candidate_count = normalize_candidate_count(candidate_count, center_count)

        self.encoder = SpatioTemporalViewEncoder(
            sensor_count=sensor_count,
            embed_dim=embed_dim,
            temporal_length=int(temporal_length),
            dropout=float(dropout),
            conv_channels=conv_channels,
        )
        latent_dim = int(self.encoder.output_dim)
        self.periodic_point_memory = SharedPeriodicPointMemory(
            center_count=int(center_count),
            embed_dim=latent_dim,
            candidate_count=candidate_count,
        )
        self.embed_dim = int(latent_dim)
        hidden_dim = int(recon_hidden_dim) if int(recon_hidden_dim) > 0 else embed_dim

        self.reconstructor = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def encode_local(self, x_local):
        return self.encoder(x_local)

    def reconstruct(self, z):
        return self.reconstruct_local(z)

    def reconstruct_local(self, z):
        return self.reconstructor(z).squeeze(-1)

    def reconstruct_periodic(self, z):
        return self.reconstructor(z).squeeze(-1)

    def forward(self, x_local):
        z_local = self.encode_local(x_local)
        z_periodic, _, _ = self.periodic_point_memory(z_local)
        xhat_local = self.reconstruct_local(z_local)
        xhat_periodic = self.reconstruct_periodic(z_periodic)
        return z_local, z_periodic, xhat_local, xhat_periodic
