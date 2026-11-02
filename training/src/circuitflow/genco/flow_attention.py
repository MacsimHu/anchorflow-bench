"""Source-routed electrical attention on the retained heterogeneous GENCO-FSI."""
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import softmax

from .adapter import CONNECTS
from .flat_start import construct_flat_start_tiny
from .runtime import Runtime


class SourceRoutedTransformerConv(nn.Module):
    """Keep edge-conditioned Q/K/V and beta skips; change only normalization."""

    def __init__(self, original, *, source_routing=True):
        super().__init__()
        if type(source_routing) is not bool:
            raise TypeError("source_routing must be boolean")
        if (not isinstance(original, TransformerConv) or original.flow != "source_to_target"
                or original.dropout != 0. or original.lin_edge is None):
            raise ValueError("This candidate requires source-to-target flow and zero dropout")
        self.original = original
        self.register_buffer("_source_routing", torch.tensor(source_routing))

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):
        if (not isinstance(x, Tensor) or not isinstance(edge_index, Tensor)
                or edge_index.dtype != torch.long or edge_index.ndim != 2
                or edge_index.shape[0] != 2):
            raise ValueError("Electrical routing requires homogeneous nodes and COO edges")
        if not bool(self._source_routing):
            return self.original(x, edge_index, edge_attr, return_attention_weights)
        if edge_attr is None:
            raise ValueError("Electrical edge attributes are required")
        conv = self.original
        heads, width = conv.heads, conv.out_channels
        source, target = edge_index
        query = conv.lin_query(x).view(-1, heads, width)
        key = conv.lin_key(x).view(-1, heads, width)
        value = conv.lin_value(x).view(-1, heads, width)
        edge = conv.lin_edge(edge_attr).view(-1, heads, width)
        scores = (query[target] * (key[source] + edge)).sum(-1) / math.sqrt(width)
        alpha = softmax(scores, source, num_nodes=x.shape[0])
        weights = F.dropout(alpha, p=conv.dropout, training=self.training)
        messages = (value[source] + edge) * weights.unsqueeze(-1)
        out = value.new_zeros((x.shape[0], heads, width)).index_add(0, target, messages)
        out = out.reshape(-1, heads * width) if conv.concat else out.mean(dim=1)
        if conv.root_weight:
            skip = conv.lin_skip(x)
            if conv.lin_beta is not None:
                beta = conv.lin_beta(torch.cat((out, skip, out-skip), dim=-1)).sigmoid()
                out = beta * skip + (1-beta) * out
            else:
                out = out + skip
        return (out, (edge_index, alpha)) if isinstance(return_attention_weights, bool) else out


class HeterogeneousFlowGNS(nn.Module):
    def __init__(self, core, *, source_routing=True):
        super().__init__()
        self.core = core
        for layer in core.layers:
            layer.convs[CONNECTS] = SourceRoutedTransformerConv(
                layer.convs[CONNECTS], source_routing=source_routing)

    @property
    def layer_residuals(self):
        return self.core.layer_residuals

    def parent_state_dict(self):
        return {name.replace(".original.", "."): value for name, value in self.core.state_dict().items()
                if not name.endswith("._source_routing")}

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, mask_dict):
        return self.core(x_dict, edge_index_dict, edge_attr_dict, mask_dict)


def construct_flow_tiny(*, source_routing=True) -> Runtime:
    result = construct_flat_start_tiny()
    before = {name: value.detach().clone() for name, value in result.model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    result.model = HeterogeneousFlowGNS(result.model, source_routing=source_routing)
    after = result.model.parent_state_dict()
    if (list(before) != list(after) or any(not torch.equal(before[k], after[k]) for k in before)
            or not torch.equal(rng, torch.get_rng_state())
            or sum(p.numel() for p in result.model.parameters()) != 1276611):
        raise ValueError("Flow routing changed parent tensors, RNG or parameter count")
    result.metadata = {
        **result.metadata, "name": "GENCO-FSI-HFA",
        "heterogeneous_flow_attention": {
            "source_routing": source_routing, "changed_relation": CONNECTS,
            "heads": 8, "layers": 12, "trainable_parameters_added": 0,
            "mechanism": "per-head outgoing rather than incoming softmax on electrical edges",
            "retained": "edge-conditioned Q/K/V, beta skips, bus/gen relations, masks, shared state head, analytic completion, raw feedback and native loss",
            "prior_art": "local FlowCAN source routing combined with pinned GENCO TransformerConv",
            "not_claimed": "Kirchhoff conservation, Flowformer linear attention, full CANOS fusion or SOTA",
        },
    }
    return result
