import os

os.environ["GENCO_SCATTER_BACKEND"] = "pyg_native_sum"

import torch

from circuitflow.api import ModelSpec, construct, forward, inference_payload, load_inference
from circuitflow.fixtures import four_bus
from circuitflow.genco.adapter import PFAdapter
from circuitflow.genco.normalization import PowerScale


def test_standalone_inference_roundtrip_and_no_label_access(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    spec = ModelSpec(max_component_buses=128)
    runtime = construct(spec)
    runtime.model.eval()
    scale = PowerScale.fixed_cpu_canary(2.5)
    adapter = PFAdapter(power_scale_pu=scale, generator_layout="aggregate_pf")
    graph, state = four_bus()
    with torch.no_grad():
        original, _ = forward(runtime, adapter, graph.clone(), "cpu", need_loss=False)
    path = tmp_path/"model.pt"
    torch.save(inference_payload(runtime, spec, scale, {"purpose": "synthetic test"}), path)
    loaded, restored, _ = load_inference(path)
    poisoned = graph.clone()
    for kind in poisoned.node_types:
        if "y" in poisoned[kind]:
            poisoned[kind].y.fill_(float("nan"))
    poisoned["bus"].bus_gen.fill_(float("nan"))
    poisoned["bus"].bus_voltages.fill_(float("nan"))
    poisoned["PV"].generation.fill_(float("nan"))
    poisoned["slack"].generation.fill_(float("nan"))
    poisoned["bus", "branch", "bus"].edge_label.fill_(float("nan"))
    with torch.no_grad():
        actual, _ = forward(loaded, restored, poisoned, "cpu", need_loss=False)
    assert torch.equal(actual["bus"], original["bus"])
    assert torch.equal(actual["bus"][[0, 1], 1], state[[0, 1], 1])
    assert torch.equal(actual["bus"][0, 0], state[0, 0])


def test_both_feedback_arms_can_backpropagate():
    torch.set_num_threads(1)
    graph, _ = four_bus()
    for mode in ("coupled", "deferred"):
        torch.manual_seed(0)
        runtime = construct(ModelSpec(coupling=mode, max_component_buses=128))
        adapter = PFAdapter(power_scale_pu=2.5, generator_layout="aggregate_pf")
        _, loss = forward(runtime, adapter, graph.clone(), "cpu")
        loss.backward()
        assert torch.isfinite(loss)
        assert any(parameter.grad is not None for parameter in runtime.model.parameters())
