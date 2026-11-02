import copy
import io
import json
import random
import tarfile
import zipfile

import numpy as np
import pytest
import torch
from torch_geometric.data import InMemoryDataset

from circuitflow import checkpoint
from circuitflow.api import ModelSpec, construct, forward
from circuitflow.cli import job_from
from circuitflow.data import CachedPool, extract_public_archive
from circuitflow.fixtures import four_bus
from circuitflow.genco.adapter import PFAdapter
from circuitflow.evaluation import evaluate, pack_results
from circuitflow.genco.normalization import PowerScale
from circuitflow.io import sha256, write_json
from circuitflow.training import Trainer, numerical_runtime, source_identity


def model_parts():
    runtime = construct(ModelSpec(max_component_buses=128))
    optimizer = torch.optim.AdamW(runtime.model.parameters(), lr=5e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.7, patience=5)
    return runtime, optimizer, scheduler


def update(runtime, optimizer):
    graph, _ = four_bus()
    adapter = PFAdapter(power_scale_pu=2.5, generator_layout="aggregate_pf")
    runtime.model.train()
    optimizer.zero_grad(set_to_none=True)
    _, loss = forward(runtime, adapter, graph, "cpu")
    loss.backward()
    optimizer.step()
    return loss.detach().clone()


def test_full_state_replays_next_update_and_rng(tmp_path):
    torch.set_num_threads(1)
    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)
    runtime, optimizer, scheduler = model_parts()
    scheduler.step(float(update(runtime, optimizer)))
    identity = dict(job="synthetic-checkpoint-test", backend="cpu")
    progress = dict(step=1, epoch=1, phase="val", val_cursor=2, val_loss_sum=1.25)
    order = [2, 0, 1]
    payload = checkpoint.capture(runtime, optimizer, scheduler, progress, order, identity, "cpu")
    path = tmp_path/"current.pt"
    checkpoint.save(path, payload, 64*1024**2)
    rng_expected = random.random(), float(np.random.random()), torch.rand(4)
    expected_loss = update(runtime, optimizer)
    expected = copy.deepcopy(runtime.model.state_dict())
    expected_moments = copy.deepcopy(optimizer.state_dict())
    restored, other_optimizer, other_scheduler = model_parts()
    actual_progress, actual_order = checkpoint.restore(
        torch.load(path, weights_only=True), restored, other_optimizer, other_scheduler, identity, "cpu")
    assert actual_progress == progress and actual_order == order
    assert random.random() == rng_expected[0]
    assert float(np.random.random()) == rng_expected[1]
    assert torch.equal(torch.rand(4), rng_expected[2])
    assert torch.equal(update(restored, other_optimizer), expected_loss)
    for key, value in expected.items():
        actual = restored.model.state_dict()[key]
        assert torch.equal(actual, value) if torch.is_tensor(value) else actual == value
    for index, state in expected_moments["state"].items():
        for key, value in state.items():
            actual = other_optimizer.state_dict()["state"][index][key]
            assert torch.equal(actual, value) if torch.is_tensor(value) else actual == value
    assert other_scheduler.state_dict() == scheduler.state_dict()
    with pytest.raises(ValueError, match="different frozen"):
        checkpoint.restore(payload, restored, other_optimizer, other_scheduler, {"job": "other"}, "cpu")


def test_memory_mapped_samples_do_not_mutate_the_shared_cache(tmp_path):
    graph, _ = four_bus()
    data, slices = InMemoryDataset.collate([graph, graph.clone()])
    path = tmp_path/"train.pt"
    torch.save((data, slices), path)
    original_bytes = path.read_bytes()
    pool = CachedPool(path, 2, case="synthetic")
    changed = pool.get(1)
    reference = changed["PQ"].x.clone()
    changed["PQ"].x.add_(7)
    assert torch.equal(pool.get(0)["PQ"].x, reference)
    assert torch.equal(pool.get(1)["PQ"].x, reference)
    assert path.read_bytes() == original_bytes


def make_archive(path, name, data):
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))


def test_public_archive_cannot_escape_or_overwrite_existing_data(tmp_path):
    destination = tmp_path/"raw"
    destination.mkdir()
    archive = tmp_path/"public.tar.gz"
    make_archive(archive, "../outside.json", b"bad")
    with pytest.raises(ValueError, match="non-local"):
        extract_public_archive(archive, destination)
    assert not (tmp_path/"outside.json").exists()
    original = destination/"sample.json"
    original.write_bytes(b"old")
    make_archive(archive, "sample.json", b"new")
    with pytest.raises(FileExistsError, match="Never overwrite"):
        extract_public_archive(archive, destination)
    assert original.read_bytes() == b"old"
    make_archive(archive, "sample.json", b"old")
    assert extract_public_archive(archive, destination)["verified_reused_files"] == 1


def test_collaboration_does_not_readmit_existing_seed_zero(tmp_path):
    path = tmp_path/"protocol.json"
    path.write_text(json.dumps(dict(
        schema="circuitflow-collaboration-v1",
        jobs=[dict(id="unauthorized-existing-job", case="case500", task="1.3", seed=0)])))
    with pytest.raises(ValueError, match="duplicate existing seed0"):
        job_from(path, "unauthorized-existing-job")


def test_partial_return_is_explicit_and_excludes_unrelated_files(tmp_path):
    run = tmp_path/"run"
    run.mkdir()
    (run/"RUN_PLAN.json").write_text('{"job":"synthetic-export-test"}')
    (run/"FAILED.json").write_text('{"type":"synthetic-test-failure"}')
    (run/"latest_a.pt").write_bytes(b"retained-synthetic-checkpoint")
    (run/"private_notes.json").write_text('{"excluded":true}')
    archive = tmp_path/"return.zip"
    result = pack_results(run, None, archive)
    assert result["status"] == "incomplete_or_failed"
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        assert "training/latest_a.pt" in names
        assert "training/FAILED.json" in names
        assert "training/private_notes.json" not in names
        assert json.loads(bundle.read("MANIFEST.json"))["status"] == "incomplete_or_failed"


def test_trainer_metadata_supports_safe_current_state_resume(tmp_path, monkeypatch):
    class Pool:
        def get(self, index):
            return four_bus()[0]

        def __len__(self):
            return 5400

    monkeypatch.setattr("circuitflow.training.load_pool", lambda *args: Pool())
    train_hash = "a"*64
    scale = PowerScale(2.5, "train_fitted", train_hash, tuple(range(48600)), "b"*64,
                       48600, generator_layout="aggregate_pf")
    write_json(tmp_path/"scale.json", scale.to_dict())
    manifest = tmp_path/"MANIFEST.json"
    write_json(manifest, dict(
        case="case30", task="1.1",
        splits=dict(train=dict(count=48600, sha256=train_hash), val=dict(count=5400)),
        power_scale=dict(file="scale.json", sha256=sha256(tmp_path/"scale.json"))))
    job = dict(case="case30", task="1.1", seed=19, maximum_active_hours=1,
               model_spec=dict(name="circuitflow", max_component_buses=128))
    trainer = Trainer(tmp_path, job, manifest, "cpu", 1)
    trainer.progress["phase"] = "train"
    trainer.train_step()
    trainer.save_current()
    restored = Trainer(tmp_path, job, manifest, "cpu", 1)
    restored.resume()
    assert restored.progress["step"] == 1
    assert restored.progress["next_batch"] == 1
    assert type(restored.identity["runtime"]["torch"]) is str
    assert torch.equal(restored.optimizer.state_dict()["state"][0]["exp_avg"],
                       trainer.optimizer.state_dict()["state"][0]["exp_avg"])


def test_readout_rejects_changed_runtime_before_loading_data(tmp_path, monkeypatch):
    runtime = numerical_runtime("cpu", 1)
    runtime["torch"] = "different-recorded-version"
    manifest = tmp_path/"MANIFEST.json"
    manifest.write_text("{}")
    (tmp_path/"inference.pt").write_bytes(b"synthetic-model")
    (tmp_path/"fixed_H200_VAL.npy").write_bytes(b"synthetic-val")
    write_json(tmp_path/"COMPLETE.json", dict(
        complete=True, progress=dict(step=152000),
        inference_sha256=sha256(tmp_path/"inference.pt"),
        fixed_H200_VAL_sha256=sha256(tmp_path/"fixed_H200_VAL.npy"),
        identity=dict(data_manifest_sha256=sha256(manifest),
                      source_sha256=source_identity(), runtime=runtime)))

    def unexpected_load(*args, **kwargs):
        pytest.fail("A changed numerical runtime must not open a data split")

    monkeypatch.setattr("circuitflow.evaluation.load_pool", unexpected_load)
    with pytest.raises(ValueError, match="declared training runtime"):
        evaluate(tmp_path, manifest, tmp_path/"readout")
    assert not (tmp_path/"readout").exists()
