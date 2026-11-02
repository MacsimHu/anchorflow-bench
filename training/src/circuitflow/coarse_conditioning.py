"""Input-only, channel-masked polar conditioning of harmonic corrections."""
from dataclasses import dataclass

import torch

from circuitflow.model import ElectricalCoarseSpace


@dataclass(frozen=True)
class ConditionedBlock:
    buses: torch.Tensor
    anchors: torch.Tensor
    basis: torch.Tensor
    ranks: tuple[int, int]


@dataclass(frozen=True)
class ConditionedCorrection:
    blocks: tuple[ConditionedBlock, ...]
    buses: int
    anchors: int

    def prolong(self, coefficients):
        if coefficients.shape != (self.anchors, 2):
            raise ValueError("Conditioned coefficients must use [logVm, theta] at every anchor")
        result = coefficients.new_zeros((self.buses, 2))
        for block in self.blocks:
            values = coefficients.index_select(0, block.anchors)
            delta = torch.einsum("cna,ac->nc", block.basis, values)
            result.index_copy_(0, block.buses, delta)
        return result


@torch.no_grad()
def prepare_conditioned_correction(space: ElectricalCoarseSpace, free):
    """Equalize resolved singular gains while preserving their squared sum."""
    basis = space.prolongation
    if (free.dtype != torch.bool or free.shape != (space.bus_graph.numel(), 2)
            or free.device != basis.device or basis.dtype not in (torch.float32, torch.float64)):
        raise ValueError("Expected the model's [logVm, theta] free-state masks")
    blocks = []
    for component in range(basis.shape[0]):
        buses = (space.bus_graph == component).nonzero(as_tuple=True)[0]
        anchors = (space.anchor_graph == component).nonzero(as_tuple=True)[0]
        original = basis[component].index_select(0, space.bus_slot[buses])
        original = original.index_select(1, space.anchor_slot[anchors]).double()
        masks = free.index_select(0, buses)
        matrices, ranks = [], []
        for channel in range(2):
            if not masks[:, channel].any():
                matrices.append(torch.zeros_like(original))
                ranks.append(0)
                continue
            masked = original[masks[:, channel]]
            u, singular, vh = torch.linalg.svd(masked, full_matrices=False)
            cutoff = torch.finfo(basis.dtype).eps * max(masked.shape)
            keep = singular > singular[0] * cutoff
            rank = int(keep.sum())
            if rank == 0:
                raise FloatingPointError("A nonempty correction channel has no resolved harmonic direction")
            scale = singular[keep].square().mean().sqrt()
            conditioned = torch.zeros_like(original)
            conditioned[masks[:, channel]] = (u[:, keep] @ vh[keep]) * scale
            if not torch.isfinite(conditioned).all():
                raise FloatingPointError("Nonfinite conditioned harmonic correction")
            matrices.append(conditioned)
            ranks.append(rank)
        blocks.append(ConditionedBlock(
            buses, anchors, torch.stack(matrices).to(basis.dtype), tuple(ranks)))
    return ConditionedCorrection(tuple(blocks), free.shape[0], space.anchor_slot.numel())
