"""Small synthetic, solved four-bus example; not benchmark or research evidence."""
import torch
from torch_geometric.data import HeteroData

from .genco.adapter import BRANCH, analytic_branch_flows


def four_bus():
    graph = HeteroData()
    state = torch.tensor([[.12, 1.04], [.06, 1.02], [-.02, .98], [.04, 1.01]])
    graph["bus"].x = torch.zeros(4, 2)
    graph["bus"].num_nodes = 4
    graph["bus"].shunt = torch.tensor([[.015, .025], [.01, -.02], [0., .035], [.005, 0.]])
    graph["bus"].limits = torch.tensor([[.9, 1.1]]*4)
    graph["bus"].vn_kv = torch.tensor([132., 132., 33., 33.])
    graph[BRANCH].edge_index = torch.tensor([[0, 0, 1, 2, 3], [1, 1, 2, 3, 0]])
    graph[BRANCH].edge_attr = torch.tensor([
        [.02, .15, .001, .025, .002, .015, 1.1, .07],
        [.04, .25, 0., .01, 0., .02, 1., 0.],
        [.03, .20, .002, .03, .003, .01, 1., 0.],
        [.01, .12, 0., .02, 0., .025, .97, -.03],
        [.025, .18, 0., .01, 0., .02, 1., 0.],
    ])
    graph[BRANCH].edge_limits = torch.full((5, 1), 2.)
    flows = analytic_branch_flows(state, graph[BRANCH].edge_index, graph[BRANCH].edge_attr)
    net = torch.zeros(4, 2)
    net.index_add_(0, graph[BRANCH].edge_index[0], flows[:, :2])
    net.index_add_(0, graph[BRANCH].edge_index[1], flows[:, 2:])
    net += state[:, 1:2].square()*graph["bus"].shunt*torch.tensor([1., -1.])
    demand = torch.tensor([[.1, .03], [.2, .06], [0., 0.], [0., 0.]])
    demand[2:] = -net[2:]
    graph["bus"].bus_demand = demand
    graph["bus"].bus_gen = net+demand
    graph["bus"].bus_voltages = state
    graph["PQ"].x, graph["PQ"].y = demand[2:].clone(), state[2:].clone()
    graph["PQ"].known_generation = torch.zeros(2, 2)
    graph["PV"].x = torch.stack((net[1:2, 0], state[1:2, 1]), dim=1)
    graph["PV"].y = torch.stack((net[1:2, 1], state[1:2, 0]), dim=1)
    graph["PV"].demand, graph["PV"].generation = demand[1:2], (net+demand)[1:2]
    graph["slack"].x, graph["slack"].y = state[:1].clone(), net[:1].clone()
    graph["slack"].demand, graph["slack"].generation = demand[:1], (net+demand)[:1]
    for kind, buses in (("PQ", [2, 3]), ("PV", [1]), ("slack", [0])):
        graph[kind, f"{kind}_link", "bus"].edge_index = torch.tensor([list(range(len(buses))), buses])
    graph["bus"].y = torch.cat((graph["slack"].y, graph["PV"].y, graph["PQ"].y))
    graph[BRANCH].edge_label = flows
    return graph, state
