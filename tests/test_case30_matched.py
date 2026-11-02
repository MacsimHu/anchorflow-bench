import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import case30_matched
from case30_contract import authorize, schedule, tensor_identity


ROOT = Path(case30_matched.__file__).parent


def test_original_six_coordinates_have_exactly_three_company_jobs():
    allocation = json.loads((ROOT / "CASE30_ALLOCATION.json").read_bytes())
    allowed = json.loads((ROOT / "ACTIVE_SCOPE.json").read_bytes())["new_training_coordinate_allowlist"]
    assert [authorize(allocation, seed, "Unit")["id"] for seed in range(3)] == allowed
    for seed in range(3):
        with pytest.raises(ValueError, match="original lab"):
            authorize(allocation, seed, "Deferred")
    for seed in (-1, 11, 22):
        with pytest.raises(ValueError, match="exact coordinate"):
            authorize(allocation, seed, "Unit")
    from run_company import authorize_current_scope
    with pytest.raises(RuntimeError, match="withdrawn"):
        authorize_current_scope()


def test_original_batch_order_includes_the_24_graph_tail():
    batches = schedule(1)
    assert len(batches) == 760 and len(batches[-1]) == 24
    flattened = [item for batch in batches for item in batch]
    assert sorted(flattened) == list(range(48600))
    import hashlib
    assert hashlib.sha256(json.dumps(batches[0], separators=(",", ":")).encode()).hexdigest() == (
        "778fe2104fd6c3f1693865f5716a41142125e0540e33b19ac4dba6be1af06819")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_public_initial_is_the_original_tensor_set_with_only_unit_configuration_changed(seed):
    from circuitflow.api import ModelSpec, construct

    reference = json.loads(case30_matched.REFERENCE.read_bytes())
    item = reference["initials"][str(seed)]
    saved = torch.load(case30_matched.REFERENCE.parent / item["file"], weights_only=True)
    assert case30_matched.digest(case30_matched.REFERENCE.parent / item["file"]) == item["sha256"]
    runtime = construct(ModelSpec(**reference["model_spec"]))
    expected = dict(saved["model"]["_extra_state"], coarse_weighting="unit")
    assert runtime.model.get_extra_state() == expected
    runtime.model.load_state_dict(dict(saved["model"], _extra_state=expected), strict=True)
    assert tensor_identity(runtime.model.state_dict()) == item["tensor_identity"]
    assert sum(parameter.numel() for parameter in runtime.model.parameters()) == reference["parameters"]


def test_scheduler_uses_all_450_original_dev_cases_and_weights_the_tail(monkeypatch, tmp_path):
    import circuitflow.api
    import torch_geometric.data

    batches = []
    model = SimpleNamespace(eval=lambda: None, layer_residuals=[torch.tensor(2.)] * 12)

    def fake_forward(runtime, adapter, raw, device):
        batches.append(len(raw))
        value = float(len(batches))
        bus = torch.zeros(len(raw) * 30, 2)
        bus[:, 1] = 1.
        return {"bus": bus}, torch.tensor(value)

    monkeypatch.setattr(circuitflow.api, "forward", fake_forward)
    monkeypatch.setattr(torch_geometric.data.Batch, "from_data_list", lambda graphs: graphs)
    cls = case30_matched.trainer_type()
    trainer = cls.__new__(cls)
    trainer.root, trainer.device = tmp_path, "cpu"
    trainer.val = list(range(450))
    trainer.runtime, trainer.adapter = SimpleNamespace(model=model), None
    observed = []
    trainer.scheduler = SimpleNamespace(step=lambda value: observed.append(value))
    trainer.record_attempt = lambda *args: None
    trainer.order = []
    trainer.progress = dict(phase="val", epoch=200, val_cursor=0, val_loss_sum=0.,
                            val_physics_sum=0., val_seconds=0., val_graphs=0,
                            attempted_val_graphs=0, observations=[], best=None)
    for index in range(8):
        assert trainer.validation_step() == (index == 7)
    assert batches == [64] * 7 + [2]
    assert observed == [sum((index + 1) * count for index, count in enumerate(batches)) / 450]
    assert trainer.progress["val_graphs"] == 450
    assert trainer.progress["phase"] == "done"
    assert trainer.progress["pending_best"]["epoch"] == 200
    assert np.load(tmp_path / "dev_0200.npy", allow_pickle=False).shape == (450, 30, 2)
