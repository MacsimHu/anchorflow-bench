"""Label-free CANOS -> official GENCO PF tensors and original-unit readout.

This adapter is local integration code, not claimed to be an upstream dataset
implementation. CANOS powers/admittances are p.u.; Va and phase shifts are radians.
"""
import torch
from torch_geometric.data import HeteroData

from .normalization import PowerScale
from .runtime import PAPER_MASK, PRIMARY_MASK, load_official


BRANCH = ("bus", "branch", "bus")
CONNECTS = ("bus", "connects", "bus")
GEN_BUS = ("gen", "connected_to", "bus")
BUS_GEN = ("bus", "connected_to", "gen")
BUS_COLUMNS = ("Pd", "Qd", "Qg", "Vm", "Va", "PQ", "PV", "REF", "Vmin", "Vmax",
               "Qmin", "Qmax", "GS", "BS", "vn_kv")
GEN_COLUMNS = ("Pg", "Pmin", "Pmax", "c0", "c1", "c2")
EDGE_COLUMNS = ("P", "Q", "Yself.real", "Yself.imag", "Ycross.real", "Ycross.imag",
                "tap", "angmin", "angmax", "rate_a")


def _float(value, shape, name, dtype, device):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    value = value.to(dtype=dtype, device=device)
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def _links(value, count, n, name, device):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.shape != (2, count):
        raise ValueError(f"{name} must be int64 [2,{count}]")
    value = value.to(device=device)
    if torch.any(value[1] < 0) or torch.any(value[1] >= n):
        raise ValueError(f"{name} has an invalid bus index")
    if not torch.equal(value[0].sort().values, torch.arange(count, device=device)):
        raise ValueError(f"{name} must have exactly one link per node row")
    return value[:, value[0].argsort()]


def branch_coefficients(edge_attr):
    """Local physical branch admittances; forward Yff/Yft, reverse Ytt/Ytf."""
    r, x, gf, bf, gt, bt, tap, shift = edge_attr[:, :8].unbind(dim=1)
    if torch.any(r.abs() + x.abs() == 0) or torch.any(tap <= 0):
        raise ValueError("Nonzero impedance and positive tap are required")
    series = 1 / torch.complex(r, x)
    transformer = torch.polar(tap, shift)
    yff = (series + torch.complex(gf, bf)) / tap.square()
    yft = -series / transformer.conj()
    ytt = series + torch.complex(gt, bt)
    ytf = -series / transformer
    return yff, yft, ytt, ytf


def analytic_branch_flows(state, edge_index, edge_attr):
    """Stateless complex arithmetic matching FlowCAN.derive_branch_flows.

    Source verified: core/models/flowcan.py lines 615-655. The adapter does not
    import/instantiate a FlowCAN model or read its labels/generation metadata.
    """
    voltage = state[:, 1] * torch.exp(1j * state[:, 0])
    yff, yft, ytt, ytf = branch_coefficients(edge_attr)
    vf, vt = voltage[edge_index[0]], voltage[edge_index[1]]
    sf = vf * (yff * vf + yft * vt).conj()
    st = vt * (ytt * vt + ytf * vf).conj()
    result = torch.stack((sf.real, sf.imag, st.real, st.imag), dim=1)
    if edge_attr.shape[1] == 9:
        result = result * (edge_attr[:, 8:9] > 0).to(result.dtype)
    return result


class PFAdapter:
    def __init__(self, scale=None, *, power_scale_pu=None, mask=PRIMARY_MASK,
                 dtype=torch.float32, generator_layout="explicit_gen"):
        if scale is not None and power_scale_pu is not None:
            raise ValueError("Pass either scale or power_scale_pu, not both")
        if power_scale_pu is not None:
            scale = (power_scale_pu if isinstance(power_scale_pu, PowerScale)
                     else PowerScale.externally_frozen(power_scale_pu))
        if not isinstance(scale, PowerScale):
            raise TypeError("Pass a frozen PowerScale or explicit caller-frozen power_scale_pu")
        if mask not in (PRIMARY_MASK, PAPER_MASK):
            raise ValueError("Unknown PF mask")
        if dtype not in (None, torch.float32, torch.float64):
            raise ValueError("Only float32/float64 tensors are supported")
        if generator_layout not in ("explicit_gen", "aggregate_pf"):
            raise ValueError("generator_layout must be explicit_gen or aggregate_pf")
        self.scale, self.mask, self.dtype = scale, mask, dtype
        self.generator_layout = generator_layout

    @property
    def power_scale_pu(self):
        return self.scale.Bpu

    def _controls(self, data, *, dtype=None):
        n = data["bus"].num_nodes
        if not isinstance(n, int) or n <= 0:
            raise ValueError("A nonempty bus node count is required")
        reference = data["slack"].x
        dtype = dtype or self.dtype or reference.dtype
        device = reference.device
        kinds, indices = {}, {}
        coverage = torch.zeros(n, dtype=torch.long, device=device)
        bus_type = torch.zeros(n, dtype=torch.long, device=device)
        for kind, number in (("PQ", 1), ("PV", 2), ("slack", 3)):
            features = data[kind].x
            if not isinstance(features, torch.Tensor) or features.ndim != 2:
                raise ValueError(f"{kind}.x must have shape [count,2]")
            kinds[kind] = _float(features, (len(features), 2), f"{kind}.x", dtype, device)
            link = _links(data[kind, f"{kind}_link", "bus"].edge_index,
                          len(features), n, f"{kind} links", device)
            indices[kind] = link[1]
            coverage.index_add_(0, link[1], torch.ones_like(link[1]))
            bus_type[link[1]] = number
            if kind != "PQ" and torch.any(features[:, 1] <= 0):
                raise ValueError("Known PV/REF Vm must be positive")
        if not torch.all(coverage == 1) or len(indices["slack"]) == 0:
            raise ValueError("PQ/PV/slack links must partition buses, with a reference")
        prescribed_pq_gen = data["PQ"].get("known_generation")
        if prescribed_pq_gen is not None:
            prescribed_pq_gen = _float(prescribed_pq_gen, kinds["PQ"].shape,
                                       "PQ.known_generation", dtype, device)
            if torch.any(prescribed_pq_gen != 0):
                raise ValueError("Official GENCO PF does not support nonzero PQ generation")

        demand_value = data["bus"].get("bus_demand")
        demand = None if demand_value is None else _float(
            demand_value, (n, 2), "bus.bus_demand", dtype, device
        ).clone()
        if demand is None:
            demand = torch.zeros((n, 2), dtype=dtype, device=device)
            for kind in kinds:
                value = kinds[kind] if kind == "PQ" else data[kind].get("demand")
                if value is None and len(indices[kind]) == 0:
                    continue
                demand[indices[kind]] = _float(
                    value, kinds[kind].shape, f"{kind}.demand", dtype, device
                )
        for kind in kinds:
            value = kinds[kind] if kind == "PQ" else data[kind].get("demand")
            if value is not None and not torch.allclose(
                demand[indices[kind]], _float(value, kinds[kind].shape, f"{kind}.demand", dtype, device),
                rtol=1e-5, atol=1e-7,
            ):
                raise ValueError(f"Known {kind} demand disagrees with bus.bus_demand")

        expected = torch.cat((indices["PV"], indices["slack"])).sort().values
        gen_links, metadata_rows = self._generator_links(data, expected, n, device)
        if len(expected) < 2:
            raise ValueError("Pinned GENCO squeeze() requires at least two aggregate generators")
        return n, kinds, indices, demand, bus_type, gen_links, metadata_rows

    def _generator_links(self, data, expected, n, device):
        forward_key, reverse_key = ("gen", "gen_link", "bus"), ("bus", "gen_link", "gen")
        if self.generator_layout == "explicit_gen" and forward_key not in data.edge_types:
            raise ValueError("Explicit CANOS gen links required; opt into generator_layout='aggregate_pf' for PF-only data")
        explicit = None
        for relation in (forward_key, reverse_key):
            if relation not in data.edge_types:
                continue
            raw = data[relation].get("edge_index")
            if not isinstance(raw, torch.Tensor) or raw.ndim != 2:
                raise ValueError("Explicit CANOS gen links must contain edge_index")
            if relation == reverse_key:
                raw = raw.flip(0)
            candidate = _links(raw, raw.shape[1], n, "gen links", device)
            if not torch.equal(candidate[1].sort().values, expected):
                raise ValueError("Require exactly one aggregate generator per PV/REF bus and none at PQ")
            if explicit is not None and not torch.equal(candidate, explicit):
                raise ValueError("Forward/reverse explicit gen links disagree")
            explicit = candidate
        if self.generator_layout == "explicit_gen":
            return explicit, torch.arange(len(expected), device=device)
        metadata_rows = None if explicit is None else explicit[1].argsort()
        links = torch.stack((torch.arange(len(expected), device=device), expected))
        return links, metadata_rows

    def _branches(self, data, n, dtype, device):
        attributes = data[BRANCH].edge_attr
        if not isinstance(attributes, torch.Tensor) or attributes.ndim != 2 or attributes.shape[1] not in (8, 9):
            raise ValueError("CANOS branch attributes must be [M,8] or [M,9]")
        attributes = _float(attributes, attributes.shape, "branch.edge_attr", dtype, device)
        edges = data[BRANCH].edge_index
        if not isinstance(edges, torch.Tensor) or edges.dtype != torch.long or edges.shape != (2, len(attributes)):
            raise ValueError("branch.edge_index must be int64 [2,M]")
        edges = edges.to(device=device)
        if torch.any(edges < 0) or torch.any(edges >= n):
            raise ValueError("Invalid branch bus indices")
        branch_coefficients(attributes)
        return edges, attributes

    def build_inputs(self, data):
        """Accept a CANOS HeteroData or Batch; never access y/solved metadata."""
        load_official()
        from gridfm_graphkit.datasets.masking import AddPFHeteroMask

        n, kinds, index, demand, types, gen_links, metadata_rows = self._controls(data)
        dtype, device = demand.dtype, demand.device
        g = gen_links.shape[1]
        edges, attributes = self._branches(data, n, dtype, device)
        active = (attributes[:, 8] > 0) if attributes.shape[1] == 9 else torch.ones(
            len(attributes), dtype=torch.bool, device=device
        )
        active_rows = active.nonzero(as_tuple=True)[0]
        active_edges, active_attr = edges[:, active], attributes[active]
        bus = demand.new_zeros((n, 15))
        gen = demand.new_zeros((g, 6))
        bus[:, :2] = demand
        for number in (1, 2, 3):
            bus[:, 4 + number] = (types == number).to(dtype)
        bus[index["PV"], 3] = kinds["PV"][:, 1]
        bus[index["slack"], 3] = kinds["slack"][:, 1]
        bus[index["slack"], 4] = kinds["slack"][:, 0]
        bus[:, 12:14] = _float(data["bus"].shunt, (n, 2), "bus.shunt", dtype, device)
        pg = demand.new_zeros(n)
        pg[index["PV"]] = kinds["PV"][:, 0] + demand[index["PV"], 0]
        gen[:, 0] = pg[gen_links[1]]

        placeholders = []
        limits = data["bus"].get("limits")
        if limits is not None:
            bus[:, 8:10] = _float(limits, (n, 2), "bus.limits", dtype, device)
        else:
            placeholders += ["bus.Vmin", "bus.Vmax"]
        gen_store = data["gen"] if "gen" in data.node_types else {}
        if metadata_rows is None and any(key in gen_store for key in ("limits", "cost_coefficients")):
            raise ValueError("Cannot align known gen metadata without explicit gen links")
        limits = gen_store.get("limits")
        if limits is not None:
            limits = _float(limits, (g, 4), "gen.limits", dtype, device)[metadata_rows]
            gen[:, 1:3] = limits[:, :2]
            bus[gen_links[1], 10:12] = limits[:, 2:4]
        else:
            placeholders += ["gen.Pmin", "gen.Pmax", "bus.Qmin", "bus.Qmax"]
        cost = gen_store.get("cost_coefficients")
        if cost is not None:
            gen[:, 3:6] = _float(cost, (g, 3), "gen.cost_coefficients", dtype, device)[metadata_rows]
        else:
            placeholders += ["gen.c0", "gen.c1", "gen.c2"]
        kv = data["bus"].get("vn_kv")
        if kv is not None:
            bus[:, 14] = _float(kv.reshape(-1), (n,), "bus.vn_kv", dtype, device)
        else:
            placeholders += ["bus.vn_kv"]

        yff, yft, ytt, ytf = branch_coefficients(active_attr)
        count = len(active_attr)
        branch = demand.new_zeros((2 * count, 10))
        branch[:, 2:6] = torch.cat((
            torch.stack((yff.real, yff.imag, yft.real, yft.imag), dim=1),
            torch.stack((ytt.real, ytt.imag, ytf.real, ytf.imag), dim=1),
        ))
        branch[:, 6] = active_attr[:, 6].repeat(2)
        angles = data[BRANCH].get("angle_limits")
        if angles is not None:
            angles = _float(angles, (len(attributes), 2), "branch.angle_limits", dtype, device)
            branch[:, 7:9] = angles[active].repeat(2, 1)
        else:
            placeholders += ["branch.angmin", "branch.angmax"]
        rate = data[BRANCH].get("edge_limits")
        if rate is not None:
            rate = _float(rate, (len(attributes), 1), "branch.edge_limits", dtype, device)
            branch[:, 9] = rate[active, 0].repeat(2)
        else:
            placeholders += ["branch.rate_a"]
        graph = HeteroData()
        graph["bus"].num_nodes, graph["gen"].num_nodes = n, g
        graph["bus"].x, graph["gen"].x = bus, gen
        graph[CONNECTS].edge_index = torch.cat((active_edges, active_edges.flip(0)), dim=1)
        graph[CONNECTS].edge_attr = branch
        graph[GEN_BUS].edge_index, graph[BUS_GEN].edge_index = gen_links, gen_links.flip(0)
        graph = AddPFHeteroMask()(graph)
        if self.mask == PRIMARY_MASK:
            graph.mask_dict["bus"][graph.mask_dict["REF"], 3] = False
        for feature in placeholders:
            kind, column = feature.split(".")
            columns = {"bus": BUS_COLUMNS, "gen": GEN_COLUMNS, "branch": EDGE_COLUMNS}[kind]
            if not bool(graph.mask_dict[kind][:, columns.index(column)].all()):
                raise RuntimeError("A placeholder would become a known input")

        # Preserve actual known metadata before masking; never populate hidden states from labels.
        graph["bus"].unmasked_features = bus.clone()
        graph["gen"].unmasked_features = gen.clone()
        graph[CONNECTS].unmasked_features = branch.clone()
        B = self.scale.Bpu
        bus[:, [0, 1, 2, 10, 11, 12, 13]] /= B
        gen[:, :3] /= B
        branch[:, :6] /= B
        branch[:, 9] /= B
        bus.masked_fill_(graph.mask_dict["bus"], 0)
        gen.masked_fill_(graph.mask_dict["gen"], 0)
        branch.masked_fill_(graph.mask_dict["branch"], 0)
        graph[CONNECTS].physical_branch_row = active_rows.repeat(2)
        if "batch" in data["bus"]:
            graph["bus"].batch = data["bus"].batch.clone()
            graph["gen"].batch = data["bus"].batch[gen_links[1]].clone()
        graph.adapter_metadata = {
            "mask": self.mask, "normalization_sha256": self.scale.fingerprint,
            "Bpu": B, "power_scale_pu": B, "normalization_mode": self.scale.mode,
            "normalization_source_id": self.scale.source_id,
            "masked_zero_placeholders": placeholders,
            "units": "already_pu_and_radians", "labels_read": False,
            "generator_layout": self.generator_layout,
            "normalization_generator_layout": self.scale.generator_layout,
            "generator_convention": ("one_aggregate_per_PV_REF_sorted_by_bus"
                                     if self.generator_layout == "aggregate_pf"
                                     else "one_aggregate_per_PV_REF_in_CANOS_gen_row_order"),
        }
        return graph

    def build_targets(self, data):
        """Separate offline labels; not called from build_inputs or decode."""
        n, kinds, index, demand, _, links, _ = self._controls(data)
        dtype, device = demand.dtype, demand.device
        bus = demand.new_zeros((n, 5))
        bus[:, :2] = demand
        pg = demand.new_zeros(n)
        for kind in kinds:
            y = _float(data[kind].y, kinds[kind].shape, f"{kind}.y", dtype, device)
            rows = index[kind]
            if kind == "PQ":
                bus[rows, 3], bus[rows, 4] = y[:, 1], y[:, 0]
            elif kind == "PV":
                bus[rows, 2] = y[:, 0] + demand[rows, 1]
                bus[rows, 3], bus[rows, 4] = kinds[kind][:, 1], y[:, 1]
                pg[rows] = kinds[kind][:, 0] + demand[rows, 0]
            else:
                bus[rows, 2] = y[:, 1] + demand[rows, 1]
                bus[rows, 3], bus[rows, 4] = kinds[kind][:, 1], kinds[kind][:, 0]
                pg[rows] = y[:, 0] + demand[rows, 0]
        bus[:, :3] /= self.scale.Bpu
        return {"bus": bus, "gen": pg[links[1], None] / self.scale.Bpu}

    def decode(self, output, data, *, case_name="case30"):
        """Read bus physical Pg/Qg (not gen head), return original CANOS units."""
        prediction = output["bus"]
        n, _, index, demand, _, _, _ = self._controls(data, dtype=prediction.dtype)
        if prediction.shape != (n, 4) or prediction.device != demand.device:
            raise ValueError("Expected dtype/device-aligned official bus output [N,4]")
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite official prediction")
        state = prediction[:, [1, 0]]
        physical_gen = prediction[:, 2:4] * self.scale.Bpu
        edges, attributes = self._branches(data, n, state.dtype, state.device)
        pv, ref = index["PV"], index["slack"]
        return {
            "bus": state, "PQ": state[index["PQ"]],
            "PV": torch.stack((physical_gen[pv, 1] - demand[pv, 1], state[pv, 0]), dim=1),
            "slack": physical_gen[ref] - demand[ref],
            "edge_preds": analytic_branch_flows(state, edges, attributes),
            "casename": case_name,
        }
