import torch
import torch.nn as nn


class SpatioTemporalViewEncoder(nn.Module):

    def __init__(
        self,
        sensor_count: int,
        embed_dim: int = 128,
        temporal_length: int = 20,
        dropout: float = 0.0,
        conv_channels: int = 0,
    ):
        super().__init__()
        self.sensor_count = int(sensor_count)
        self.temporal_length = int(temporal_length)
        if self.sensor_count <= 0:
            raise ValueError("sensor_count must be > 0")
        if self.temporal_length <= 0:
            raise ValueError("temporal_length must be > 0")

        configured_channels = int(conv_channels)
        self.conv_channels = (
            configured_channels if configured_channels > 0 else max(1, int(embed_dim) // 2)
        )
        self.spatial_kernel_size = self.sensor_count * self.temporal_length
        self.spatial_conv = nn.Conv1d(
            in_channels=1,
            out_channels=self.conv_channels,
            kernel_size=self.spatial_kernel_size,
            bias=False,
        )
        self.temporal_conv = nn.Conv1d(
            in_channels=1,
            out_channels=self.conv_channels,
            kernel_size=self.temporal_length,
            dilation=self.sensor_count,
            bias=False,
        )
        self.output_dim = 2 * self.conv_channels

        self.dropout = nn.Dropout(float(dropout))
        self._init_weights()

    def _init_weights(self):
        for module in [self.spatial_conv, self.temporal_conv]:
            nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def _validate_input(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError(f"SpatioTemporalViewEncoder expects [B,time,sensor], got {tuple(x.shape)}")
        available_flat_len = int(x.size(1)) * self.sensor_count
        required_flat_len = self.spatial_kernel_size + self.sensor_count - 1
        if available_flat_len < required_flat_len:
            raise ValueError(
                "SpatioTemporalViewEncoder needs enough rows to take one "
                f"full spatio-temporal window ending at each target sensor: need flat length "
                f"{required_flat_len}, got {available_flat_len} "
                f"(sequence length {int(x.size(1))}, sensors={self.sensor_count})."
            )
        if int(x.size(2)) != self.sensor_count:
            raise ValueError(f"Expected {self.sensor_count} sensors, got {int(x.size(2))}")

    def _flatten_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = int(x.size(0))
        x_flat = x.contiguous().reshape(batch_size, 1, -1)
        x_flat = torch.flip(x_flat, dims=(-1,))

        spatial_features = self.spatial_conv(x_flat)[..., : self.sensor_count]
        spatial_features = spatial_features.transpose(1, 2).contiguous()
        spatial_features = torch.flip(spatial_features, dims=(1,))

        temporal_features = self.temporal_conv(x_flat)[..., : self.sensor_count]
        temporal_features = temporal_features.transpose(1, 2).contiguous()
        temporal_features = torch.flip(temporal_features, dims=(1,))

        features = torch.cat([spatial_features, temporal_features], dim=-1)
        return features.reshape(batch_size, self.sensor_count, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        features = self._flatten_features(x)
        return self.dropout(features)
