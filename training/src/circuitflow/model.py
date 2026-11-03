# SPDX-License-Identifier: Apache-2.0
# Reuses GraphKit 0912c734 PF primitives and local HFA routing.
# See third_party/genco_0912c734/LICENSE and research/flowcan/genco/PROVENANCE.json.
"""Opt-in two-scale physical-state iteration; not a default or a SOTA claim."""

from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.nn import TransformerConv

from circuitflow.genco.adapter import CONNECTS, GEN_BUS
from circuitflow.genco.flat_start import construct_flat_start_preset
from circuitflow.genco.flow_attention import SourceRoutedTransformerConv
from circuitflow.genco.runtime import Runtime


@dataclass(frozen=True)
class ElectricalCoarseSpace:
    prolongation: torch.Tensor
    restriction: torch.Tensor
    bus_graph: torch.Tensor
    bus_slot: torch.Tensor
    anchor_graph: torch.Tensor
    anchor_slot: torch.Tensor
    reference: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor

    def restrict(self, fine):
        shape = (*self.prolongation.shape[:2], fine.shape[-1])
        padded = fine.new_zeros(shape)
        padded[self.bus_graph, self.bus_slot] = fine
        coarse = self.restriction @ padded
        return coarse[self.anchor_graph, self.anchor_slot]

    def prolong(self, coarse):
        shape = (*self.prolongation.shape[:2], coarse.shape[-1])
        padded = coarse.new_zeros(shape)
        padded[self.anchor_graph, self.anchor_slot] = coarse
        fine = self.prolongation @ padded
        return fine[self.bus_graph, self.bus_slot]


@torch.no_grad()
def electrical_coarse_space(edge_index, edge_attr, pv, ref, *, max_component_buses=128,
                            weight_mode="electrical"):
    """Input-only harmonic basis; unit weights provide a topology-only control."""
    if weight_mode not in ("electrical", "unit", "mean_token"):
        raise ValueError("Unknown coarse-space weight mode")
    if (edge_index.dtype != torch.long or edge_index.ndim != 2
            or edge_index.shape[0] != 2 or edge_attr.ndim != 2
            or edge_attr.shape[0] != edge_index.shape[1] or edge_attr.shape[1] < 4
            or pv.dtype != torch.bool or ref.dtype != torch.bool
            or pv.ndim != 1 or ref.shape != pv.shape
            or edge_attr.dtype not in (torch.float32, torch.float64)
            or type(max_component_buses) is not int or max_component_buses < 1):
        raise ValueError("Invalid electrical coarse-space input schema")
    n = pv.numel()
    if (n == 0 or torch.any(pv & ref) or not torch.isfinite(edge_attr).all()
            or any(t.device != edge_attr.device for t in (edge_index, pv, ref))
            or (edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= n))):
        raise ValueError("Invalid electrical coarse-space topology or values")
    weight = torch.linalg.vector_norm(edge_attr[:, 2:4].double(), dim=-1)
    active = weight > 0
    if weight_mode == "unit":
        weight = active.to(dtype=weight.dtype)
    endpoints = edge_index[:, active].detach().cpu().tolist()
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for source, target in zip(*endpoints):
        a, b = root(source), root(target)
        if a != b:
            parent[b] = a
    components = {}
    for i in range(n):
        components.setdefault(root(i), []).append(i)
    groups = list(components.values())
    width = max(map(len, groups))
    if width > max_component_buses:
        raise ValueError("Dense prototype coarse setup exceeds its component-size limit")
    device, dtype = edge_attr.device, edge_attr.dtype
    graph = torch.empty(n, dtype=torch.long, device=device)
    slot = torch.empty_like(graph)
    valid = torch.zeros((len(groups), width), dtype=torch.bool, device=device)
    for g, buses in enumerate(groups):
        index = torch.tensor(buses, dtype=torch.long, device=device)
        graph[index] = g
        slot[index] = torch.arange(len(buses), device=device)
        valid[g, :len(buses)] = True
    reference = torch.empty(len(groups), dtype=torch.long, device=device)
    for g, buses in enumerate(groups):
        index = torch.tensor(buses, dtype=torch.long, device=device)
        selected = index[ref[index]]
        if selected.numel() != 1:
            raise ValueError("Each connected component requires exactly one REF bus")
        reference[g] = selected[0]
    if weight_mode == "mean_token":
        basis = torch.zeros((len(groups), width, width), device=device, dtype=dtype)
        restriction = torch.zeros_like(basis)
        anchor_graph = torch.arange(len(groups), device=device)
        anchor_slot = slot[reference]
        for g, buses in enumerate(groups):
            basis[g, :len(buses), anchor_slot[g]] = 1
            restriction[g, anchor_slot[g], :len(buses)] = 1 / len(buses)
        edges = torch.stack((anchor_graph, anchor_graph))
        return ElectricalCoarseSpace(
            basis, restriction, graph, slot, anchor_graph, anchor_slot,
            reference, edges, edge_attr.new_zeros((len(groups), 1)))
    anchors = torch.zeros_like(valid)
    anchors[graph, slot] = pv | ref
    senders, receivers = edge_index[:, active]
    flat_index = (graph[senders] * width + slot[senders]) * width + slot[receivers]
    adjacency = weight.new_zeros(len(groups) * width * width)
    adjacency.index_add_(0, flat_index, weight[active])
    adjacency = adjacency.reshape(len(groups), width, width)
    adjacency = (adjacency + adjacency.transpose(1, 2)) * .5
    degree = adjacency.sum(-1)
    laplacian = torch.diag_embed(degree) - adjacency
    identity = torch.eye(width, device=device, dtype=torch.float64).expand(len(groups), -1, -1)
    fixed = anchors | ~valid
    if torch.any((degree <= 0) & ~fixed):
        raise ValueError("An unanchored bus has no active electrical connection")
    row_scale = torch.where(fixed, torch.ones_like(degree), degree)
    system = torch.where(fixed.unsqueeze(-1), identity, laplacian / row_scale.unsqueeze(-1))
    # The pinned CPU MKL batched LU can stall on 500-bus components.
    if system.device.type == "cpu" and len(groups) > 1:
        boundary = torch.diag_embed(anchors.double())
        basis = torch.stack([torch.linalg.solve(matrix, rhs) for matrix, rhs in zip(system, boundary)])
    else:
        basis = torch.linalg.solve(system, torch.diag_embed(anchors.double()))
    if (not torch.isfinite(basis).all() or torch.any(basis < -1e-8)
            or not torch.allclose(basis.sum(-1), valid.double(), rtol=1e-8, atol=1e-8)):
        raise FloatingPointError("Electrical harmonic interpolation is not numerically admissible")
    mass = basis.sum(1)
    restriction = basis.transpose(1, 2) / torch.where(
        anchors, mass, torch.ones_like(mass)).unsqueeze(-1)
    coarse_adjacency = basis.transpose(1, 2) @ adjacency @ basis
    coarse_degree = coarse_adjacency.sum(-1)
    denominator = torch.sqrt(coarse_degree.unsqueeze(2) * coarse_degree.unsqueeze(1))
    normalized = coarse_adjacency / denominator.clamp_min(torch.finfo(torch.float64).tiny)
    anchor_graph, anchor_slot = anchors.nonzero(as_tuple=True)
    anchor_number = torch.full(valid.shape, -1, dtype=torch.long, device=device)
    anchor_number[anchor_graph, anchor_slot] = torch.arange(anchor_graph.numel(), device=device)
    batch, source, target = (anchors.unsqueeze(2) & anchors.unsqueeze(1)).nonzero(as_tuple=True)
    coarse_edges = torch.stack((anchor_number[batch, source], anchor_number[batch, target]))
    coarse_attr = torch.log1p(normalized[batch, source, target]).unsqueeze(-1).to(dtype)
    return ElectricalCoarseSpace(
        basis.to(dtype), restriction.to(dtype), graph, slot, anchor_graph, anchor_slot,
        reference, coarse_edges, coarse_attr)


def signed_log(value):
    return value.sign() * torch.log1p(value.abs())


class CircuitFlow(nn.Module):
    """Carry free voltage states through local prediction and coarse correction.

    ``deferred`` supervises the same cumulative global output at every layer but
    only feeds the local state into the next iteration. Both arms perform the
    same local and combined physical readouts; only the feedback trajectory differs.
    """

    def __init__(self, core, *, coupling="coupled", source_routing=True,
                 max_component_buses=128, coarse_conditioning="none",
                 coarse_weighting="electrical"):
        super().__init__()
        if coupling not in ("coupled", "deferred") or type(source_routing) is not bool:
            raise ValueError("Unknown CircuitFlow coupling or routing")
        if coarse_weighting not in ("electrical", "unit", "mean_token", "knockout"):
            raise ValueError("Unknown coarse-space weight mode")
        if (core.task != "PowerFlow" or core.num_layers != 12 or core.dropout != 0
                or type(max_component_buses) is not int or max_component_buses < 1):
            raise ValueError("CircuitFlow v1 requires the pinned 12-layer fixed-control PF contract")
        self.core = core
        self.config = dict(schema="circuitflow-v1", coupling=coupling,
                           source_routing=source_routing,
                           max_component_buses=max_component_buses,
                           state="logVm_theta", increment_scale=1 / core.num_layers)
        # The default keeps the original serialized checkpoint contract unchanged.
        if coarse_weighting != "electrical":
            self.config["coarse_weighting"] = coarse_weighting
        self.configure_coarse_conditioning(coarse_conditioning)
        width, latent = core.hidden_dim, core.hidden_dim * core.heads
        self.state_inputs = nn.ModuleList()
        self.edge_updates = nn.ModuleList()
        for i, layer in enumerate(core.layers):
            size = width if i == 0 else latent
            layer.convs[CONNECTS] = SourceRoutedTransformerConv(
                layer.convs[CONNECTS], source_routing=source_routing)
            self.state_inputs.append(nn.Linear(4, size))
            self.edge_updates.append(nn.Sequential(
                nn.Linear(2 * size + width + 2, width), nn.LeakyReLU(),
                nn.Linear(width, width), nn.LayerNorm(width)))
        self.coarse_context = nn.Sequential(
            nn.Linear(latent + core.input_bus_dim + 4, latent),
            nn.LayerNorm(latent), nn.LeakyReLU())
        self.coarse_edge = nn.Linear(1, width)
        self.coarse_conv = SourceRoutedTransformerConv(
            TransformerConv(latent, width, heads=core.heads, edge_dim=width, beta=True,
                            dropout=0.), source_routing=source_routing)
        self.coarse_norm = nn.LayerNorm(latent)
        self.coarse_head = nn.Sequential(
            nn.Linear(latent, width), nn.LayerNorm(width), nn.LeakyReLU(),
            nn.Linear(width, 2))
        # These are increments from a control-preserving flat state, not absolute Vm.
        for head in (core.mlp_bus[-1], self.coarse_head[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.correction = None
        self.last_work = {}

    @property
    def layer_residuals(self):
        return self.core.layer_residuals

    def get_extra_state(self):
        return dict(self.config)

    def set_extra_state(self, state):
        if state != self.config:
            raise ValueError("CircuitFlow checkpoint configuration differs")

    def configure_coarse_conditioning(self, mode):
        if mode not in ("none", "masked_polar_rms_v1"):
            raise ValueError("Unknown coarse-correction conditioning")
        if mode == "none":
            self.config.pop("coarse_conditioning", None)
        else:
            self.config["coarse_conditioning"] = mode

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, mask_dict, *, return_aux=False):
        from gridfm_graphkit._scatter import scatter_add
        from gridfm_graphkit.datasets.globals import VM_H, VA_H, PG_H

        core = self.core
        core.layer_residuals = {}
        bus_fixed = x_dict["bus"][:, VM_H:VA_H + 1]
        bus_mask = mask_dict["bus"][:, VM_H:VA_H + 1]
        gen_fixed = x_dict["gen"][:, :PG_H + 1]
        gen_mask = mask_dict["gen"][:, :PG_H + 1]
        pv, ref = mask_dict["PV"], mask_dict["REF"]
        if (not torch.equal(~bus_mask[:, 0], pv | ref)
                or not torch.equal(~bus_mask[:, 1], ref)
                or torch.any(bus_fixed[~bus_mask[:, 0], 0] <= 0)):
            raise ValueError("Known PV/REF voltage controls must match the fixed-PF contract")
        bus_edges = edge_index_dict[CONNECTS]
        raw_edge = edge_attr_dict[CONNECTS]
        coarse = electrical_coarse_space(
            bus_edges, raw_edge, pv, ref,
            max_component_buses=self.config["max_component_buses"],
            weight_mode=("electrical" if self.config.get("coarse_weighting") == "knockout"
                         else self.config.get("coarse_weighting", "electrical")))
        conditioned = None
        if self.config.get("coarse_conditioning") == "masked_polar_rms_v1":
            from circuitflow.coarse_conditioning import prepare_conditioned_correction
            conditioned = prepare_conditioned_correction(coarse, bus_mask)
        correction_inputs = None if self.correction is None else self.correction.prepare(
            coarse, bus_edges, raw_edge, x_dict["bus"], bus_mask)
        self.last_work = dict(physical_readouts=0, harmonic_setups=1,
                              harmonic_systems=coarse.prolongation.shape[0],
                              largest_component=coarse.prolongation.shape[-1],
                              fine_layers=core.num_layers, coarse_layers=core.num_layers)
        if self.config.get("coarse_weighting") == "mean_token":
            self.last_work.update(harmonic_setups=0, harmonic_systems=0, mean_token_setups=1)
        if conditioned is not None:
            self.last_work["conditioned_channel_ranks"] = [block.ranks for block in conditioned.blocks]
        senders, receivers = bus_edges
        _, gen_to_bus = edge_index_dict[GEN_BUS]
        h_bus, h_gen = core.input_proj_bus(x_dict["bus"]), core.input_proj_gen(x_dict["gen"])
        projected = {key: core.input_proj_edge(value) if value is not None else None
                     for key, value in edge_attr_dict.items()}
        edge_state = projected[CONNECTS]
        coarse_edge = self.coarse_edge(coarse.edge_attr)
        input_context = coarse.restrict(x_dict["bus"])
        initial_vm = torch.where(bus_mask[:, 0], torch.ones_like(bus_fixed[:, 0]), bus_fixed[:, 0])
        initial_angle = bus_fixed[coarse.reference, 1][coarse.bus_graph]
        state = torch.stack((initial_vm.log(), initial_angle), dim=-1)
        global_sum = torch.zeros_like(state)
        coarse_hidden = None
        traces = []

        def physics(z, gen):
            voltage = torch.stack((z[:, 0].exp(), z[:, 1]), dim=-1)
            voltage = torch.where(bus_mask, voltage, bus_fixed)
            if not torch.isfinite(voltage).all() or torch.any(voltage[:, 0] <= 0):
                raise FloatingPointError("CircuitFlow produced nonfinite/nonpositive voltage")
            pft, qft = core.branch_flow_layer(voltage, bus_edges, raw_edge)
            p_in, q_in = core.node_injection_layer(pft, qft, bus_edges, voltage.shape[0])
            generation = scatter_add(gen.squeeze(-1), gen_to_bus, dim=0, dim_size=voltage.shape[0])
            output = core.physics_decoder(p_in, q_in, voltage, x_dict["bus"], generation, mask_dict)
            residual = torch.stack(core.node_residuals_layer(
                p_in, q_in, output, x_dict["bus"]), dim=-1)
            self.last_work["physical_readouts"] += 1
            return output, residual, torch.stack((pft, qft), dim=-1)

        def controls(z):
            fixed = torch.stack((initial_vm.log(), bus_fixed[:, 1]), dim=-1)
            return torch.where(bus_mask, z, fixed)

        _, residual, branch_power = physics(state, gen_fixed)
        for i, conv in enumerate(core.layers):
            context = torch.cat((state, signed_log(residual)), dim=-1)
            h_bus = h_bus + self.state_inputs[i](context)
            edge_input = torch.cat((h_bus[senders], h_bus[receivers], edge_state,
                                    signed_log(branch_power)), dim=-1)
            edge_state = edge_state + self.edge_updates[i](edge_input)
            projected[CONNECTS] = edge_state
            out = conv({"bus": h_bus, "gen": h_gen}, edge_index_dict, projected)
            bus_update = core.activation(core.norms_bus[i](out["bus"]))
            gen_update = core.activation(core.norms_gen[i](out["gen"]))
            h_bus = h_bus + bus_update if h_bus.shape == bus_update.shape else bus_update
            h_gen = h_gen + gen_update if h_gen.shape == gen_update.shape else gen_update
            local = core.mlp_bus(h_bus) / core.num_layers
            local = local.masked_fill(~bus_mask, 0)
            local_state = controls(state + local)
            gen_temp = torch.where(gen_mask, core.mlp_gen(h_gen), gen_fixed)
            _, local_residual, local_power = physics(local_state, gen_temp)
            coarse_input = torch.cat((coarse.restrict(h_bus), input_context,
                                      coarse.restrict(local_state),
                                      coarse.restrict(signed_log(local_residual))), dim=-1)
            context = self.coarse_context(coarse_input)
            coarse_hidden = context if coarse_hidden is None else coarse_hidden + context
            coarse_hidden = coarse_hidden + core.activation(self.coarse_norm(
                self.coarse_conv(coarse_hidden, coarse.edge_index, coarse_edge)))
            prolongation = coarse if conditioned is None else conditioned
            global_delta = prolongation.prolong(self.coarse_head(coarse_hidden)) / core.num_layers
            global_delta = global_delta.masked_fill(~bus_mask, 0)
            if self.config.get("coarse_weighting") == "knockout":
                global_delta = global_delta * 0
            if self.config["coupling"] == "deferred":
                global_sum = global_sum + global_delta
                combined = controls(local_state + global_sum)
            else:
                combined = controls(local_state + global_delta)
            output, combined_residual, combined_power = physics(combined, gen_temp)
            metric_delta = None
            if self.correction is not None:
                before_metric = combined
                combined = controls(self.correction(
                    combined, combined_residual, h_bus, coarse_hidden, correction_inputs))
                output, combined_residual, combined_power = physics(combined, gen_temp)
                if return_aux:
                    metric_delta = combined - before_metric
                self.last_work["metric_corrections"] = i + 1
            core.layer_residuals[i] = torch.linalg.vector_norm(combined_residual, dim=-1).mean()
            if self.config["coupling"] == "coupled":
                state, residual, branch_power = combined, combined_residual, combined_power
            else:
                state, residual, branch_power = local_state, local_residual, local_power
            h_bus = h_bus + core.physics_mlp(residual)
            if return_aux:
                trace = dict(local_state=local_state, combined_state=combined,
                             feedback_state=state, local_delta=local,
                             global_delta=global_delta, residual=combined_residual)
                if metric_delta is not None:
                    trace["metric_delta"] = metric_delta
                traces.append(trace)
        predicted = {"bus": output, "gen": gen_temp}
        return (predicted, dict(iterations=traces, work=dict(self.last_work))) if return_aux else predicted


def construct_circuit_preset(preset, *, coupling="coupled", source_routing=True,
                             max_component_buses=128, coarse_conditioning="none",
                             coarse_weighting="electrical") -> Runtime:
    """Expose native capacity choices without declaring a tuned or scalable winner."""
    result = construct_flat_start_preset(preset)
    result.model = CircuitFlow(result.model, coupling=coupling, source_routing=source_routing,
                               max_component_buses=max_component_buses,
                               coarse_conditioning=coarse_conditioning,
                               coarse_weighting=coarse_weighting)
    result.metadata = {
        **{k: v for k, v in result.metadata.items() if k != "initialization_ablation"},
        "name": ("CircuitFlow-v1-" + ("" if preset == "tiny" else preset + "-") + coupling
                 + ("-unit-coarse" if coarse_weighting == "unit" else
                    "" if coarse_weighting == "electrical" else "-" + coarse_weighting)),
        "parent_model_module": result.metadata["model_module"], "model_module": __file__,
        "initialization": "Control-preserving flat state; zero local/global increment output layers",
        "parameter_count": sum(p.numel() for p in result.model.parameters()),
        "circuitflow": result.model.get_extra_state(),
        "prior_art": "GraphKit/GENCO PF primitives; local HFA; CANOS/GNS edge-state updates; harmonic coarse spaces",
        "not_claimed": "SOTA, firstness, Kirchhoff-conserving attention, exact coarse AC equivalence, or large-grid scaling",
    }
    return result


def construct_circuit_tiny(*, coupling="coupled", source_routing=True,
                          coarse_conditioning="none", coarse_weighting="electrical") -> Runtime:
    return construct_circuit_preset("tiny", coupling=coupling, source_routing=source_routing,
                                    coarse_conditioning=coarse_conditioning,
                                    coarse_weighting=coarse_weighting)
