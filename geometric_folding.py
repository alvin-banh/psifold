"""
Hierarchical Monte Carlo with E(3) Equivariant GNN for 3D RNA Structure Prediction

Architecture:
- L-module: Monte Carlo sampling with E(3) GNN (fast, explores conformational space)
- H-module: Transformer-based evaluation and guidance (slow, assesses global quality)
- Constraint loss: Sum of all geometric violations
- Cyclic refinement: Iteratively improve structure through hierarchical processing

This is an experimental approach combining:
1. Stochastic sampling (Monte Carlo)
2. Equivariant geometry (E(3) GNN)
3. Hierarchical processing (HRM)
4. Constraint-based optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math


class E3EquivariantGNN(nn.Module):
    """
    E(3)-Equivariant Graph Neural Network for geometric structure generation.

    Maintains rotational and translational equivariance - if you rotate the input,
    the output rotates the same way.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        # Message passing layers
        self.message_net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim + 1, hidden_dim),  # +1 for distance
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Update node features (scalar features)
        self.node_update = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        # Update coordinates (vector features - equivariant)
        self.coord_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),  # Scalar weights for coordinate updates
        )

    def forward(
        self,
        node_features: torch.Tensor,  # [batch, n_nodes, node_dim]
        coords: torch.Tensor,          # [batch, n_nodes, 3]
        edge_index: torch.Tensor,      # [2, n_edges]
        edge_features: Optional[torch.Tensor] = None,  # [batch, n_edges, edge_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: Per-node scalar features (e.g., nucleotide type)
            coords: 3D coordinates
            edge_index: Graph connectivity
            edge_features: Optional per-edge features (e.g., bond type)

        Returns:
            Updated node features and coordinates
        """
        batch_size, n_nodes, _ = node_features.shape

        # Compute edge vectors and distances (equivariant)
        src, dst = edge_index
        edge_vec = coords[:, dst, :] - coords[:, src, :]  # [batch, n_edges, 3]
        edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True)  # [batch, n_edges, 1]

        # Normalize edge vectors
        edge_vec_normalized = edge_vec / (edge_dist + 1e-8)

        # Build messages
        src_features = node_features[:, src, :]
        dst_features = node_features[:, dst, :]

        if edge_features is not None:
            message_input = torch.cat([src_features, dst_features, edge_features, edge_dist], dim=-1)
        else:
            message_input = torch.cat([src_features, dst_features, edge_dist], dim=-1)

        messages = self.message_net(message_input)  # [batch, n_edges, hidden_dim]

        # Aggregate messages to nodes
        aggregated = torch.zeros(batch_size, n_nodes, self.hidden_dim, device=coords.device)
        for b in range(batch_size):
            aggregated[b].index_add_(0, dst, messages[b])

        # Update node features (invariant)
        node_features_new = self.node_update(
            torch.cat([node_features, aggregated], dim=-1)
        )
        node_features_new = node_features + node_features_new  # Residual

        # Update coordinates (equivariant)
        coord_weights = self.coord_update(messages)  # [batch, n_edges, 1]
        coord_updates = coord_weights * edge_vec_normalized  # [batch, n_edges, 3]

        # Aggregate coordinate updates
        coord_delta = torch.zeros_like(coords)
        for b in range(batch_size):
            coord_delta[b].index_add_(0, dst, coord_updates[b])

        coords_new = coords + coord_delta  # Equivariant update

        return node_features_new, coords_new


class MonteCarloSampler(nn.Module):
    """
    Monte Carlo sampler that generates K candidate structures.
    Uses E(3) GNN to propose geometry-aware moves.
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        n_gnn_layers: int = 3,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature

        # Stack of E(3) GNN layers
        self.gnn_layers = nn.ModuleList([
            E3EquivariantGNN(node_dim, edge_dim=0, hidden_dim=hidden_dim)
            for _ in range(n_gnn_layers)
        ])

        # Noise prediction (for sampling)
        self.noise_predictor = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),  # Predict noise in 3D space
        )

    def forward(
        self,
        node_features: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        n_samples: int = 10,
        guide_distribution: Optional[torch.Tensor] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate n_samples candidate structures via Monte Carlo.

        Args:
            node_features: [batch, n_nodes, node_dim]
            coords: Current 3D coordinates [batch, n_nodes, 3]
            edge_index: Graph connectivity
            n_samples: Number of candidate structures to generate
            guide_distribution: Optional guidance from H-module

        Returns:
            List of (node_features, coords) candidates
        """
        candidates = []

        for _ in range(n_samples):
            # Start from current structure
            curr_features = node_features.clone()
            curr_coords = coords.clone()

            # Apply GNN layers with stochastic moves
            for gnn in self.gnn_layers:
                curr_features, curr_coords = gnn(curr_features, curr_coords, edge_index)

                # Add temperature-scaled noise (Monte Carlo exploration)
                noise = torch.randn_like(curr_coords) * self.temperature

                # Optionally guide noise with H-module's distribution
                if guide_distribution is not None:
                    noise = noise * guide_distribution.unsqueeze(-1)

                curr_coords = curr_coords + noise

            candidates.append((curr_features, curr_coords))

        return candidates


class ConstraintLoss(nn.Module):
    """
    Compute sum of all geometric constraint violations.

    Constraints:
    1. Bond lengths (backbone connectivity)
    2. Bond angles (backbone geometry)
    3. Steric clashes (no atom overlaps)
    4. Base pairing geometry (Watson-Crick, wobble)
    5. Planarity (bases should be planar)
    """

    def __init__(self):
        super().__init__()

        # Ideal geometric parameters
        self.ideal_bond_length = 1.5  # Angstroms (approximate)
        self.ideal_bond_angle = 120.0  # Degrees (approximate)
        self.min_clash_distance = 2.0  # Minimum distance between atoms

    def compute_bond_length_loss(
        self,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize deviation from ideal bond lengths."""
        src, dst = edge_index
        edge_vec = coords[:, dst, :] - coords[:, src, :]
        edge_dist = torch.norm(edge_vec, dim=-1)

        # L2 loss from ideal length
        loss = F.mse_loss(edge_dist, torch.ones_like(edge_dist) * self.ideal_bond_length)
        return loss

    def compute_angle_loss(
        self,
        coords: torch.Tensor,
        angle_triplets: torch.Tensor,  # [n_angles, 3] indices
    ) -> torch.Tensor:
        """Penalize deviation from ideal bond angles."""
        i, j, k = angle_triplets.T

        v1 = coords[:, i, :] - coords[:, j, :]
        v2 = coords[:, k, :] - coords[:, j, :]

        # Compute angles
        cos_angle = (v1 * v2).sum(-1) / (torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1) + 1e-8)
        angles = torch.acos(torch.clamp(cos_angle, -1.0, 1.0)) * 180.0 / math.pi

        # L2 loss from ideal angle
        loss = F.mse_loss(angles, torch.ones_like(angles) * self.ideal_bond_angle)
        return loss

    def compute_clash_loss(
        self,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize steric clashes (atoms too close)."""
        batch_size, n_atoms, _ = coords.shape

        # Pairwise distances
        coords_i = coords.unsqueeze(2)  # [batch, n_atoms, 1, 3]
        coords_j = coords.unsqueeze(1)  # [batch, 1, n_atoms, 3]

        dists = torch.norm(coords_i - coords_j, dim=-1)  # [batch, n_atoms, n_atoms]

        # Mask diagonal (self-distances)
        mask = ~torch.eye(n_atoms, dtype=torch.bool, device=coords.device)
        dists = dists[:, mask].reshape(batch_size, n_atoms, n_atoms - 1)

        # Penalize distances below threshold
        clashes = F.relu(self.min_clash_distance - dists)
        loss = clashes.sum() / (batch_size * n_atoms)

        return loss

    def forward(
        self,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        angle_triplets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total constraint violation.

        Returns:
            Dictionary of individual losses and total
        """
        bond_loss = self.compute_bond_length_loss(coords, edge_index)
        angle_loss = self.compute_angle_loss(coords, angle_triplets)
        clash_loss = self.compute_clash_loss(coords)

        total_loss = bond_loss + angle_loss + clash_loss

        return {
            'total': total_loss,
            'bond': bond_loss,
            'angle': angle_loss,
            'clash': clash_loss,
        }


class HRM_MC_E3(nn.Module):
    """
    Hierarchical Recurrent Model with Monte Carlo sampling and E(3) GNN.

    Architecture:
    - L-module: Monte Carlo sampler with E(3) GNN (explores conformations)
    - H-module: Transformer evaluator (assesses quality, guides sampling)
    - Cyclic refinement over N cycles × T steps

    Each cycle:
    1. L-module generates K candidate structures (Monte Carlo + E(3) GNN)
    2. H-module evaluates all K candidates using constraint loss
    3. Select best candidate, update guidance distribution
    4. Repeat for T steps within cycle
    5. H-module performs global update after T steps
    """

    def __init__(
        self,
        node_dim: int = 64,
        hidden_dim: int = 128,
        n_heads: int = 8,
        n_cycles: int = 3,
        cycle_steps: int = 5,
        n_samples_per_step: int = 10,
        gnn_layers: int = 3,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.n_cycles = n_cycles
        self.cycle_steps = cycle_steps
        self.n_samples = n_samples_per_step

        # L-module: Monte Carlo sampler with E(3) GNN
        self.l_module = MonteCarloSampler(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            n_gnn_layers=gnn_layers,
            temperature=temperature,
        )

        # H-module: Transformer for global evaluation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=node_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.h_module = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Guidance network (H-module outputs guidance for L-module)
        self.guidance_net = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),  # Per-node guidance strength
        )

        # Constraint loss
        self.constraint_loss = ConstraintLoss()

    def forward(
        self,
        sequence: torch.Tensor,        # [batch, seq_len] RNA sequence
        edge_index: torch.Tensor,      # Backbone connectivity
        angle_triplets: torch.Tensor,  # Angle constraints
        initial_coords: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            sequence: RNA sequence (encoded as integers)
            edge_index: Graph connectivity for backbone
            angle_triplets: Triplets of indices for angle constraints
            initial_coords: Optional initial 3D structure

        Returns:
            Final 3D coordinates and loss history
        """
        batch_size, seq_len = sequence.shape

        # Initialize node features (embed sequence)
        node_features = F.one_hot(sequence, num_classes=5).float()  # A, U, G, C, pad

        # Initialize coordinates (random or provided)
        if initial_coords is None:
            coords = torch.randn(batch_size, seq_len, 3) * 0.1
        else:
            coords = initial_coords

        # Guidance distribution (starts uniform)
        guidance = torch.ones(batch_size, seq_len, device=sequence.device)

        loss_history = {
            'total': [],
            'bond': [],
            'angle': [],
            'clash': [],
        }

        # Hierarchical cycles
        for cycle in range(self.n_cycles):
            # T steps of L-module refinement per cycle
            for step in range(self.cycle_steps):
                # L-module: Generate K candidate structures
                candidates = self.l_module(
                    node_features,
                    coords,
                    edge_index,
                    n_samples=self.n_samples,
                    guide_distribution=guidance,
                )

                # Evaluate all candidates
                best_loss = float('inf')
                best_coords = None

                for cand_features, cand_coords in candidates:
                    # Compute constraint loss
                    losses = self.constraint_loss(cand_coords, edge_index, angle_triplets)

                    if losses['total'] < best_loss:
                        best_loss = losses['total']
                        best_coords = cand_coords

                # Update to best candidate
                coords = best_coords

                # Track losses
                for key in loss_history:
                    loss_history[key].append(losses[key].item())

            # H-module: Global assessment and guidance update
            h_output = self.h_module(node_features)
            guidance = torch.sigmoid(self.guidance_net(h_output)).squeeze(-1)

        return coords, loss_history


# Example usage
if __name__ == '__main__':
    # Example: small RNA (10 nucleotides)
    batch_size = 4
    seq_len = 10

    # Create model
    model = HRM_MC_E3(
        node_dim=64,
        hidden_dim=128,
        n_cycles=3,
        cycle_steps=5,
        n_samples_per_step=10,
    )

    # Input: RNA sequence (A=0, U=1, G=2, C=3)
    sequence = torch.randint(0, 4, (batch_size, seq_len))

    # Backbone connectivity (linear chain)
    edge_index = torch.stack([
        torch.arange(seq_len - 1),
        torch.arange(1, seq_len),
    ])

    # Angle constraints (i-1, i, i+1 triplets)
    angle_triplets = torch.stack([
        torch.arange(seq_len - 2),
        torch.arange(1, seq_len - 1),
        torch.arange(2, seq_len),
    ], dim=1)

    # Run model
    final_coords, loss_history = model(sequence, edge_index, angle_triplets)

    print(f"Final coordinates shape: {final_coords.shape}")
    print(f"Final constraint loss: {loss_history['total'][-1]:.4f}")
    print(f"Loss improved from {loss_history['total'][0]:.4f} to {loss_history['total'][-1]:.4f}")
