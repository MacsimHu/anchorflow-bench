import copy
import json

import pytest
import torch
from torch_geometric.data import Batch

from circuitflow.api import ModelSpec, construct, forward, inference_payload, load_inference
from circuitflow.fixtures import four_bus
from circuitflow.genco.adapter import CONNECTS, PFAdapter
from circuitflow.genco.normalization import PowerScale
from circuitflow.model import electrical_coarse_space


def test_extra_allocation_has_exact_owners_without_reassigning_existing_controls():
    import case30_matched
    from case30_contract import authorize

    allocation = case30_matched.read(case30_matched.ROOT / "CASE30_EXTRA_ALLOCATION.json")
    scope = case30_matched.read(case30_matched.ROOT / "ACTIVE_SCOPE.json")
    allowed = []
    for regime, variant in (("Mixed", "Knockout"), ("Mixed", "MeanToken"), ("Intact", "MeanToken")):
        for seed in range(3):
            job = authorize(allocation, seed, variant, regime)
            assert job["owner"] == "company-A100"
            assert job["task"] == ("1.1" if regime == "Intact" else "1.3")
            allowed.append(job["id"])
    assert allowed == scope["additional_training_coordinate_allowlist"]
    for regime, variant in (("Mixed", "Deferred"), ("Mixed", "Unit"),
                            ("Intact", "Deferred"), ("Intact", "Knockout"), ("Intact", "Unit")):
        with pytest.raises(ValueError):
            authorize(allocation, 0, variant, regime)


def test_mean_token_pools_each_component_and_broadcasts_without_harmonic_solve(monkeypatch):
    adapter = PFAdapter(power_scale_pu=2.5, generator_layout="aggregate_pf")
    graph, _ = four_bus()
    inputs = adapter.build_inputs(Batch.from_data_list([graph.clone(), graph.clone()]))
    monkeypatch.setattr(torch.linalg, "solve", lambda *args: pytest.fail("Mean token used a harmonic solve"))
    coarse = electrical_coarse_space(
        inputs.edge_index_dict[CONNECTS], inputs.edge_attr_dict[CONNECTS],
        inputs.mask_dict["PV"], inputs.mask_dict["REF"], weight_mode="mean_token")
    values = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    pooled = coarse.restrict(values)
    assert torch.equal(pooled, torch.tensor([[1.5], [5.5]]))
    assert torch.equal(coarse.prolong(pooled), torch.tensor([[1.5]] * 4 + [[5.5]] * 4))
    assert coarse.anchor_graph.numel() == 2
    assert torch.equal(coarse.edge_index, torch.tensor([[0, 1], [0, 1]]))


def test_knockout_retains_modules_and_readouts_but_removes_every_global_increment():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    runtime = construct(ModelSpec(coarse_weighting="knockout", max_component_buses=128))
    full = construct(ModelSpec(max_component_buses=128))
    assert sum(p.numel() for p in runtime.model.parameters()) == sum(p.numel() for p in full.model.parameters())
    with torch.no_grad():
        runtime.model.coarse_head[-1].bias.fill_(0.25)
    graph, _ = four_bus()
    inputs = PFAdapter(power_scale_pu=2.5, generator_layout="aggregate_pf").build_inputs(graph)
    _, aux = runtime.model(
        inputs.x_dict, inputs.edge_index_dict, inputs.edge_attr_dict,
        inputs.mask_dict, return_aux=True)
    assert aux["work"]["physical_readouts"] == 25
    assert aux["work"]["coarse_layers"] == 12
    assert all(torch.count_nonzero(row["global_delta"]) == 0 for row in aux["iterations"])
    assert all(torch.equal(row["local_state"], row["combined_state"]) for row in aux["iterations"])


@pytest.mark.parametrize("weighting", ["knockout", "mean_token"])
def test_extra_control_full_adam_next_update_replay_and_inference_identity(weighting, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    spec = ModelSpec(coarse_weighting=weighting, max_component_buses=128)
    runtime = construct(spec)
    optimizer = torch.optim.AdamW(runtime.model.parameters(), lr=5e-4)
    adapter = PFAdapter(power_scale_pu=2.5, generator_layout="aggregate_pf")
    graph, controls = four_bus()

    def update(current, optim):
        optim.zero_grad(set_to_none=True)
        outputs, loss = forward(current, adapter, graph.clone(), "cpu")
        loss.backward()
        coarse_gradient = current.model.coarse_head[-1].weight.grad
        assert coarse_gradient is not None
        if weighting == "mean_token":
            assert torch.count_nonzero(coarse_gradient) > 0
        else:
            assert torch.count_nonzero(coarse_gradient) == 0
        optim.step()
        assert torch.isfinite(loss)
        assert torch.equal(outputs["bus"][[0, 1], 1], controls[[0, 1], 1])
        assert torch.equal(outputs["bus"][0, 0], controls[0, 0])

    update(runtime, optimizer)
    state_path = tmp_path / "current.pt"
    torch.save(dict(model=runtime.model.state_dict(), optimizer=optimizer.state_dict(),
                    rng=torch.get_rng_state()), state_path)
    update(runtime, optimizer)
    expected = copy.deepcopy(runtime.model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    restored = construct(spec)
    restored_optimizer = torch.optim.AdamW(restored.model.parameters(), lr=5e-4)
    state = torch.load(state_path, weights_only=True)
    restored.model.load_state_dict(state["model"], strict=True)
    restored_optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng"])
    update(restored, restored_optimizer)
    for name, value in expected.items():
        actual = restored.model.state_dict()[name]
        assert torch.equal(value, actual) if torch.is_tensor(value) else value == actual
    for index, values in expected_optimizer["state"].items():
        for key, value in values.items():
            actual = restored_optimizer.state_dict()["state"][index][key]
            assert torch.equal(value, actual) if torch.is_tensor(value) else value == actual
    path = tmp_path / "inference.pt"
    torch.save(inference_payload(restored, spec, PowerScale.fixed_cpu_canary(2.5), {}), path)
    loaded, decoder, _ = load_inference(path)
    with torch.no_grad():
        original, _ = forward(restored, adapter, graph.clone(), "cpu", need_loss=False)
        replay, _ = forward(loaded, decoder, graph.clone(), "cpu", need_loss=False)
    assert torch.equal(original["bus"], replay["bus"])
    assert loaded.model.last_work["physical_readouts"] == 25


@pytest.mark.parametrize("variant,weighting,task", [
    ("Knockout", "knockout", "1.3"), ("MeanToken", "mean_token", "1.3"),
    ("MeanToken", "mean_token", "1.1")])
def test_matched_runner_loads_original_full_tensors_into_requested_control(variant, weighting, task, tmp_path, monkeypatch):
    import case30_matched
    import circuitflow.data

    reference_path = (case30_matched.ROOT / "case30_intact_reference" / "FULL_REFERENCE.json"
                      if task == "1.1" else case30_matched.REFERENCE)
    reference = case30_matched.read(reference_path)

    class Pool:
        _data, slices = None, None

        def __init__(self, split):
            self.split = split

        def __len__(self):
            return reference["data"][self.split]["count"]

        def get(self, index):
            return None

    monkeypatch.setattr(circuitflow.data, "load_pool", lambda path, split: Pool(split))
    identities = iter(reference["data"][split]["semantic_sha256"] for split in ("train", "val"))
    monkeypatch.setattr(case30_matched, "cache_identity", lambda *args: next(identities))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(case="case30", task=task)))
    job = dict(case="case30", task=task, seed=0,
               model_spec=dict(reference["model_spec"], coarse_weighting=weighting))
    trainer = case30_matched.trainer_type(variant=variant, reference_path=reference_path)(
        tmp_path, job, manifest, "cpu", 1)
    assert trainer.spec.coarse_weighting == weighting
    assert trainer.runtime.model.config["coarse_weighting"] == weighting
    assert case30_matched.tensor_identity(trainer.runtime.model.state_dict()) == reference["initials"]["0"]["tensor_identity"]
