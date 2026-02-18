"""Spectral temporal molecular property prediction model."""

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, Linear, ModuleList
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    SAGEConv,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)
from torch_geometric.nn.inits import glorot, zeros

logger = logging.getLogger(__name__)


class SpectralConvLayer(nn.Module):
    """Spectral convolution layer using graph wavelets."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_scales: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initialize spectral convolution layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            num_scales: Number of spectral scales.
            dropout: Dropout probability.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_scales = num_scales
        self.dropout = dropout

        # Scale-specific transformations
        self.scale_convs = ModuleList([
            Linear(in_channels, out_channels // num_scales)
            for _ in range(num_scales)
        ])

        # Spectral attention mechanism
        self.spectral_attention = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        # Normalization and activation
        self.batch_norm = BatchNorm1d(out_channels)
        self.dropout_layer = Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset layer parameters."""
        for conv in self.scale_convs:
            glorot(conv.weight)
            zeros(conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        spectral_features: List[torch.Tensor],
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through spectral convolution layer.

        Args:
            x: Input node features [N, in_channels].
            spectral_features: List of spectral features for each scale.
            batch: Batch assignment vector.

        Returns:
            Output node features [N, out_channels].
        """
        # Process each spectral scale
        scale_outputs = []
        for i, (conv, spectral_feat) in enumerate(zip(self.scale_convs, spectral_features)):
            # Combine input features with spectral features
            combined = x + spectral_feat
            scale_out = conv(combined)
            scale_outputs.append(scale_out)

        # Concatenate scale outputs
        h = torch.cat(scale_outputs, dim=-1)  # [N, out_channels]

        # Apply spectral attention
        if batch is not None:
            # Group by batch for attention
            batch_size = batch.max().item() + 1
            max_nodes = torch.bincount(batch).max().item()

            # Pad sequences for attention
            h_padded = torch.zeros(
                batch_size, max_nodes, h.size(-1),
                device=h.device, dtype=h.dtype
            )

            for b in range(batch_size):
                mask = (batch == b)
                h_batch = h[mask]
                h_padded[b, :h_batch.size(0)] = h_batch

            # Apply attention
            h_att, _ = self.spectral_attention(h_padded, h_padded, h_padded)

            # Unpad sequences
            h_out = torch.zeros_like(h)
            for b in range(batch_size):
                mask = (batch == b)
                num_nodes = mask.sum().item()
                h_out[mask] = h_att[b, :num_nodes]

            h = h_out
        else:
            # Single graph case
            h = h.unsqueeze(0)  # [1, N, out_channels]
            h, _ = self.spectral_attention(h, h, h)
            h = h.squeeze(0)  # [N, out_channels]

        # Normalization and dropout
        h = self.batch_norm(h)
        h = F.relu(h)
        h = self.dropout_layer(h)

        return h


class MultiScaleSpectralEncoder(nn.Module):
    """Multi-scale spectral graph encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_scales: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initialize multi-scale spectral encoder.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of encoder layers.
            num_scales: Number of spectral scales.
            dropout: Dropout probability.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_scales = num_scales

        # Input projection
        self.input_proj = Linear(input_dim, hidden_dim)

        # Spectral feature projection (raw features to hidden dim)
        self.spectral_proj = Linear(input_dim, hidden_dim)

        # Spectral convolution layers
        self.spectral_layers = ModuleList([
            SpectralConvLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                num_scales=num_scales,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Traditional graph convolution layers for comparison
        self.graph_layers = ModuleList([
            GINConv(
                nn=nn.Sequential(
                    Linear(hidden_dim, hidden_dim),
                    BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    Linear(hidden_dim, hidden_dim),
                )
            )
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.layer_norms = ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.dropout = Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        spectral_features: List[torch.Tensor],
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through multi-scale encoder.

        Args:
            x: Node features [N, input_dim].
            edge_index: Edge connectivity [2, E].
            spectral_features: List of spectral features for each scale.
            batch: Batch assignment vector.

        Returns:
            Encoded node features [N, hidden_dim].
        """
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)

        # Project spectral features from input_dim to hidden_dim
        projected_spectral = [self.spectral_proj(sf) for sf in spectral_features]

        # Multi-scale spectral processing
        for i, (spectral_layer, graph_layer, layer_norm) in enumerate(
            zip(self.spectral_layers, self.graph_layers, self.layer_norms)
        ):
            # Spectral convolution branch
            h_spectral = spectral_layer(h, projected_spectral, batch)

            # Traditional graph convolution branch
            h_graph = graph_layer(h, edge_index)

            # Combine branches with residual connection
            h_combined = h_spectral + h_graph + h  # Triple residual
            h = layer_norm(h_combined)
            h = self.dropout(h)

        return h


class TemporalCurriculumHead(nn.Module):
    """Temporal curriculum learning head for property prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_curriculum_stages: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initialize temporal curriculum head.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension (1 for regression).
            num_curriculum_stages: Number of curriculum learning stages.
            dropout: Dropout probability.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_curriculum_stages = num_curriculum_stages

        # Stage-specific heads
        self.stage_heads = ModuleList([
            nn.Sequential(
                Linear(input_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                nn.ReLU(),
                Dropout(dropout),
                Linear(hidden_dim, hidden_dim // 2),
                BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                Dropout(dropout),
                Linear(hidden_dim // 2, output_dim),
            )
            for _ in range(num_curriculum_stages)
        ])

        # Curriculum stage classifier
        self.stage_classifier = nn.Sequential(
            Linear(input_dim, hidden_dim),
            nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, num_curriculum_stages),
        )

        # Final prediction head
        self.final_head = nn.Sequential(
            Linear(input_dim + num_curriculum_stages, hidden_dim),
            BatchNorm1d(hidden_dim),
            nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        curriculum_stage: Optional[int] = None,
        training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through curriculum head.

        Args:
            x: Input graph-level features [B, input_dim].
            curriculum_stage: Current curriculum stage (if specified).
            training: Whether in training mode.

        Returns:
            Dictionary containing predictions and auxiliary outputs.
        """
        batch_size = x.size(0)
        outputs = {}

        # Stage classification
        stage_logits = self.stage_classifier(x)
        stage_probs = F.softmax(stage_logits, dim=-1)
        outputs["stage_logits"] = stage_logits
        outputs["stage_probs"] = stage_probs

        if curriculum_stage is not None:
            # Use specified curriculum stage head (both training and validation)
            stage_pred = self.stage_heads[curriculum_stage](x)
            outputs["prediction"] = stage_pred
        else:
            # No curriculum stage specified: use weighted ensemble of all stage heads
            stage_predictions = torch.stack([
                head(x) for head in self.stage_heads
            ], dim=1)  # [B, num_stages, output_dim]
            stage_pred = torch.sum(
                stage_predictions * stage_probs.unsqueeze(-1),
                dim=1
            )
            outputs["prediction"] = stage_pred

        return outputs


class SpectralTemporalMolecularNet(nn.Module):
    """Main model combining spectral graph wavelets with curriculum learning."""

    def __init__(
        self,
        input_dim: int = 9,  # Atomic features
        hidden_dim: int = 256,
        num_spectral_layers: int = 4,
        num_scales: int = 4,
        num_curriculum_stages: int = 4,
        dropout: float = 0.1,
        pool_type: str = "attention",
    ) -> None:
        """Initialize spectral temporal molecular network.

        Args:
            input_dim: Input atomic feature dimension.
            hidden_dim: Hidden layer dimension.
            num_spectral_layers: Number of spectral convolution layers.
            num_scales: Number of spectral scales.
            num_curriculum_stages: Number of curriculum learning stages.
            dropout: Dropout probability.
            pool_type: Graph pooling type ('mean', 'max', 'add', 'attention').
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.pool_type = pool_type

        # Multi-scale spectral encoder
        self.encoder = MultiScaleSpectralEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_spectral_layers,
            num_scales=num_scales,
            dropout=dropout,
        )

        # Graph pooling
        if pool_type == "attention":
            self.pool_attention = nn.Sequential(
                Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                Linear(hidden_dim // 2, 1),
            )

        # Temporal curriculum head
        self.curriculum_head = TemporalCurriculumHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=1,  # HOMO-LUMO gap prediction
            num_curriculum_stages=num_curriculum_stages,
            dropout=dropout,
        )

    def forward(
        self,
        data: Data,
        spectral_features: List[torch.Tensor],
        curriculum_stage: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the complete model.

        Args:
            data: Input molecular graph data.
            spectral_features: Precomputed spectral features.
            curriculum_stage: Current curriculum stage.

        Returns:
            Dictionary containing predictions and auxiliary outputs.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Encode with spectral features
        h = self.encoder(x, edge_index, spectral_features, batch)

        # Graph-level pooling
        if self.pool_type == "mean":
            graph_repr = global_mean_pool(h, batch)
        elif self.pool_type == "max":
            graph_repr = global_max_pool(h, batch)
        elif self.pool_type == "add":
            graph_repr = global_add_pool(h, batch)
        elif self.pool_type == "attention":
            # Attention pooling
            attn_weights = self.pool_attention(h)  # [N, 1]

            # Apply attention within each graph
            graph_reprs = []
            num_graphs = batch.max().item() + 1

            for i in range(num_graphs):
                mask = (batch == i)
                if mask.sum() > 0:  # Check if graph has nodes
                    h_graph = h[mask]  # [num_nodes_i, hidden_dim]
                    weights_graph = attn_weights[mask]  # [num_nodes_i, 1]

                    # Apply softmax within each graph separately
                    weights_graph = F.softmax(weights_graph, dim=0)
                    graph_repr_i = torch.sum(h_graph * weights_graph, dim=0)
                else:
                    # Handle empty graph case
                    graph_repr_i = torch.zeros(h.size(-1), device=h.device, dtype=h.dtype)

                graph_reprs.append(graph_repr_i)

            graph_repr = torch.stack(graph_reprs)  # [batch_size, hidden_dim]
        else:
            raise ValueError(f"Unknown pooling type: {self.pool_type}")

        # Predict with curriculum head
        outputs = self.curriculum_head(
            graph_repr,
            curriculum_stage=curriculum_stage,
            training=self.training
        )

        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        curriculum_stage: Optional[int] = None,
        stage_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-objective loss.

        Args:
            outputs: Model outputs dictionary.
            targets: Target HOMO-LUMO gap values.
            curriculum_stage: Current curriculum stage.
            stage_weight: Weight for stage classification loss.

        Returns:
            Dictionary containing loss components.
        """
        losses = {}

        # Primary regression loss (MAE)
        predictions = outputs["prediction"]
        regression_loss = F.l1_loss(predictions.squeeze(), targets)
        losses["regression"] = regression_loss

        # Stage classification loss (if curriculum stage is provided)
        if curriculum_stage is not None and "stage_logits" in outputs:
            stage_targets = torch.full(
                (predictions.size(0),),
                curriculum_stage,
                device=predictions.device,
                dtype=torch.long
            )
            stage_loss = F.cross_entropy(outputs["stage_logits"], stage_targets)
            losses["stage_classification"] = stage_loss
        else:
            losses["stage_classification"] = torch.tensor(0.0, device=predictions.device)

        # Total loss
        total_loss = regression_loss + stage_weight * losses["stage_classification"]
        losses["total"] = total_loss

        return losses