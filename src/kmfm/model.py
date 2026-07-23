from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SpectralMLP(nn.Module):
    def __init__(self, bands: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(bands),
            nn.Linear(bands, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        return self.network(spectrum)


class SpectralConv1D(nn.Module):
    """Multi-scale Conv1d whose kernel really slides along the band axis."""

    def __init__(
        self,
        hidden_dim: int,
        kernels: tuple[int, ...] = (3, 7, 11),
        branch_dim: int = 24,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("All spectral kernels must be positive odd integers")
        self.kernels = tuple(int(kernel) for kernel in kernels)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, branch_dim, kernel_size=kernel, padding=kernel // 2, bias=False),
                    nn.BatchNorm1d(branch_dim),
                    nn.GELU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                for kernel in self.kernels
            ]
        )
        self.project = nn.Sequential(
            nn.Linear(branch_dim * len(self.kernels), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        # (batch, bands) -> (batch, one signal channel, bands).
        sequence = spectrum.unsqueeze(1)
        features = [branch(sequence).squeeze(-1) for branch in self.branches]
        return self.project(torch.cat(features, dim=-1))


class SelectiveScan1D(nn.Module):
    """A compact, pure-PyTorch diagonal selective state-space scan.

    This is deliberately described as a selective SSM rather than the official
    Mamba implementation. It avoids platform-specific CUDA compilation while
    retaining input-dependent state decay, write and read gates.
    """

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = int(dim)
        self.norm = nn.LayerNorm(dim)
        self.value_gate = nn.Linear(dim, dim * 2)
        self.dt_proj = nn.Linear(dim, dim)
        self.write_proj = nn.Linear(dim, dim)
        self.read_proj = nn.Linear(dim, dim)
        self.a_log = nn.Parameter(torch.zeros(dim))
        self.skip = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[-1] != self.dim:
            raise ValueError(f"Expected (batch, length, {self.dim}), got {tuple(sequence.shape)}")
        normalized = self.norm(sequence)
        value, gate = self.value_gate(normalized).chunk(2, dim=-1)
        value = F.gelu(value)
        delta = F.softplus(self.dt_proj(normalized)) + 1e-4
        decay_rate = F.softplus(self.a_log).view(1, 1, -1) + 1e-4
        decay = torch.exp(-delta * decay_rate)
        write = torch.sigmoid(self.write_proj(normalized))
        read = torch.sigmoid(self.read_proj(normalized))

        state = torch.zeros_like(value[:, 0])
        outputs: list[torch.Tensor] = []
        for step in range(sequence.shape[1]):
            state = decay[:, step] * state + (1.0 - decay[:, step]) * write[:, step] * value[:, step]
            current = read[:, step] * state + self.skip * value[:, step]
            outputs.append(current * torch.sigmoid(gate[:, step]))
        scanned = torch.stack(outputs, dim=1)
        return sequence + self.dropout(self.out_proj(scanned))


class MultiDirectionSpatialSSM(nn.Module):
    def __init__(self, bands: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(bands, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.GELU(),
        )
        # The same scan is shared across directions to keep the contribution small.
        self.scan = SelectiveScan1D(hidden_dim, dropout=dropout)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def _scan_row_major(self, feature: torch.Tensor, reverse: bool) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        sequence = feature.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        if reverse:
            sequence = torch.flip(sequence, dims=(1,))
        scanned = self.scan(sequence)
        if reverse:
            scanned = torch.flip(scanned, dims=(1,))
        return scanned.reshape(batch, height, width, channels)

    def _scan_col_major(self, feature: torch.Tensor, reverse: bool) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        sequence = feature.permute(0, 3, 2, 1).reshape(batch, width * height, channels)
        if reverse:
            sequence = torch.flip(sequence, dims=(1,))
        scanned = self.scan(sequence)
        if reverse:
            scanned = torch.flip(scanned, dims=(1,))
        return scanned.reshape(batch, width, height, channels).permute(0, 2, 1, 3)

    def forward(self, patch: torch.Tensor, context_mask: torch.Tensor | None = None) -> torch.Tensor:
        feature = self.stem(patch)
        if context_mask is not None:
            feature = feature * context_mask.unsqueeze(1).to(feature.dtype)
        maps = [
            self._scan_row_major(feature, reverse=False),
            self._scan_row_major(feature, reverse=True),
            self._scan_col_major(feature, reverse=False),
            self._scan_col_major(feature, reverse=True),
        ]
        merged = torch.stack(maps, dim=0).mean(dim=0)
        if context_mask is not None:
            merged = merged * context_mask.unsqueeze(-1).to(merged.dtype)
            denominator = context_mask.sum(dim=(1, 2), keepdim=False).clamp_min(1.0).unsqueeze(-1)
            pooled = merged.sum(dim=(1, 2)) / denominator
        else:
            pooled = merged.mean(dim=(1, 2))
        center = merged[:, merged.shape[1] // 2, merged.shape[2] // 2]
        return self.output(torch.cat([center, pooled], dim=-1))


def normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1).clamp_min(1e-8)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1, keepdim=True)
    return entropy / math.log(logits.shape[-1])


class FusionModule(nn.Module):
    def __init__(self, hidden_dim: int, mode: str = "reliability", dropout: float = 0.1) -> None:
        super().__init__()
        supported = {
            "spatial_only",
            "spectral_only",
            "sum",
            "concat",
            "global",
            "gate",
            "reliability",
        }
        if mode not in supported:
            raise ValueError(f"Unsupported fusion {mode!r}; choose from {sorted(supported)}")
        self.mode = mode
        if mode == "concat":
            self.project = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout)
            )
        elif mode == "global":
            self.global_logits = nn.Parameter(torch.zeros(2))
        elif mode in {"gate", "reliability"}:
            input_dim = hidden_dim * 3 + (2 if mode == "reliability" else 0)
            self.gate = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(
        self,
        spatial: torch.Tensor,
        spectral: torch.Tensor,
        spatial_entropy: torch.Tensor,
        spectral_entropy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "spatial_only":
            gate = torch.ones((spatial.shape[0], 1), device=spatial.device, dtype=spatial.dtype)
            return spatial, gate
        if self.mode == "spectral_only":
            gate = torch.zeros((spatial.shape[0], 1), device=spatial.device, dtype=spatial.dtype)
            return spectral, gate
        if self.mode == "sum":
            gate = torch.full((spatial.shape[0], 1), 0.5, device=spatial.device, dtype=spatial.dtype)
            return 0.5 * (spatial + spectral), gate
        if self.mode == "concat":
            gate = torch.full((spatial.shape[0], 1), float("nan"), device=spatial.device, dtype=spatial.dtype)
            return self.project(torch.cat([spatial, spectral], dim=-1)), gate
        if self.mode == "global":
            weights = torch.softmax(self.global_logits, dim=0)
            gate = weights[0].expand(spatial.shape[0], 1)
            return weights[0] * spatial + weights[1] * spectral, gate

        parts = [spatial, spectral, torch.abs(spatial - spectral)]
        if self.mode == "reliability":
            parts.extend([spatial_entropy, spectral_entropy])
        gate = torch.sigmoid(self.gate(torch.cat(parts, dim=-1)))
        return gate * spatial + (1.0 - gate) * spectral, gate


class LASSFNet(nn.Module):
    """Leakage-Aware Selective Spectral-Spatial Fusion network."""

    def __init__(
        self,
        bands: int,
        num_classes: int,
        hidden_dim: int = 64,
        spectral: Literal["conv1d", "mlp"] = "conv1d",
        fusion: Literal[
            "spatial_only",
            "spectral_only",
            "sum",
            "concat",
            "global",
            "gate",
            "reliability",
        ] = "reliability",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if spectral == "conv1d":
            self.spectral_encoder: nn.Module = SpectralConv1D(hidden_dim=hidden_dim, dropout=dropout)
        elif spectral == "mlp":
            self.spectral_encoder = SpectralMLP(bands=bands, hidden_dim=hidden_dim, dropout=dropout)
        else:
            raise ValueError("spectral must be 'conv1d' or 'mlp'")
        self.spatial_encoder = MultiDirectionSpatialSSM(bands=bands, hidden_dim=hidden_dim, dropout=dropout)
        self.spatial_head = nn.Linear(hidden_dim, num_classes)
        self.spectral_head = nn.Linear(hidden_dim, num_classes)
        self.fusion = FusionModule(hidden_dim=hidden_dim, mode=fusion, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.num_classes = int(num_classes)

    def forward(
        self, patch: torch.Tensor, context_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        center = patch[:, :, patch.shape[-2] // 2, patch.shape[-1] // 2]
        spectral_feature = self.spectral_encoder(center)
        spatial_feature = self.spatial_encoder(patch, context_mask=context_mask)
        spectral_logits = self.spectral_head(spectral_feature)
        spatial_logits = self.spatial_head(spatial_feature)
        spectral_entropy = normalized_entropy(spectral_logits)
        spatial_entropy = normalized_entropy(spatial_logits)
        fused_feature, gate = self.fusion(
            spatial_feature,
            spectral_feature,
            spatial_entropy,
            spectral_entropy,
        )
        logits = self.classifier(fused_feature)
        return {
            "logits": logits,
            "spatial_logits": spatial_logits,
            "spectral_logits": spectral_logits,
            "gate": gate,
            "spatial_entropy": spatial_entropy,
            "spectral_entropy": spectral_entropy,
            "spatial_feature": spatial_feature,
            "spectral_feature": spectral_feature,
        }
