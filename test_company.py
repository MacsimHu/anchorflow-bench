import json
import math
import os

import pytest

import anchortrain
from anchorbench import METRIC_SCHEMA, score_sample, write_new
from exchange import CLAIM, ingest, validate

os.environ["GENCO_SCATTER_BACKEND"] = "pyg_native_sum"


def frozen(tmp_path):
    path = tmp_path / "protocol.json"
    write_new(path, anchortrain.protocol())
    return path


def example(path):
    job = anchortrain.read_protocol(path)["jobs"][0]
    return dict(
        schema="anchorflow-public-result-v1", status="complete",
        protocol_sha256=anchortrain.digest(path), code_sha256=anchortrain.code_identity(),
        **{key: "a" * 64 for key in
           ("data_manifest_sha256", "checkpoint_sha256", "test_sha256", "readout_sha256")},
        **{key: job[key] for key in ("case", "regime", "method", "variant", "seed")},
        coordinate_sha256=job["id"], actual_updates=152000,
        checkpoint_selection_rule="fixed_H200", metric_schema=METRIC_SCHEMA,
        runtime=dict(python="3.10.13", torch="2.1.2+cu118", numpy="1.26.4",
                     torch_geometric="2.6.1", device="cuda", threads=4),
        same_inference_for_all_metrics=True, partitions_pooled=False, scientific_claim=CLAIM,
        groups=[dict(partition=partition, topology=topology,
                     cases=(2000 if partition == "feasible" else 200) * (3 if topology == "All" else 1),
                     MSE=.02, PBL=.01, rPBL_percent=.1, PBL_p95_pu=.03, PBL_p99_pu=.04,
                     PBL_worst_pu=.05, equation_worst_pu=.06,
                     equation_pass_fraction={"0.001": .1, "0.0001": .05, "1e-07": 0.})
                for partition in ("feasible", "near_infeasible") for topology in ("All", "N", "N-1", "N-2")])


def test_company_protocol_is_bounded_and_tamper_evident(tmp_path):
    path = frozen(tmp_path)
    plan = anchortrain.read_protocol(path)
    assert len(plan["jobs"]) == 100
    assert {job["seed"] for job in plan["jobs"]} == {11, 22, 33, 44, 55}
    assert {job["method"] for job in plan["jobs"]} == {"AnchorFlow", "GENCO"}
    assert {job["variant"] for job in plan["jobs"]} == {"Full", "Deferred", "Unit weights"}
    plan["jobs"][0]["seed"] = 66
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError):
        anchortrain.read_protocol(path)


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(private_note="do not publish"),
    lambda r: r.update(actual_updates=True),
    lambda r: r.update(metric_schema="load_normalized"),
    lambda r: r.update(seed=0),
    lambda r: r["runtime"].update(torch="/private/host"),
    lambda r: r["groups"].pop(),
    lambda r: r["groups"][0].update(PBL=float("nan")),
    lambda r: r["groups"][0].update(cases=30000),
    lambda r: r["groups"][0].update(MSE=2.),
    lambda r: r["groups"][0]["equation_pass_fraction"].update({"1e-07": .4}),
])
def test_aggregate_receiver_rejects_mismatches(tmp_path, mutation):
    path = frozen(tmp_path)
    result = example(path)
    mutation(result)
    with pytest.raises(ValueError):
        validate(result, path)


def test_aggregate_ingest_is_private_field_free_and_nonoverwriting(tmp_path):
    path = frozen(tmp_path)
    result = example(path)
    validate(result, path)
    source = tmp_path / "result.json"
    write_new(source, result)
    target = ingest(source, path, tmp_path / "receiver")
    assert ingest(source, path, tmp_path / "receiver") == target
    assert json.loads(target.read_text()) == result
    result["checkpoint_sha256"] = "b" * 64
    source.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(FileExistsError):
        ingest(source, path, tmp_path / "receiver")


def test_voltage_bridge_preserves_physics_masks_and_original_branch_orientation():
    import torch
    from anchor_physics import metric_input
    from circuitflow.fixtures import four_bus
    graph, reference = four_bus()
    record = metric_input(graph, reference, "synthetic", "feasible", "N")
    scored = score_sample(record)
    assert scored["MSE"] < 1e-10
    # Independent complex Y-bus oracle, rather than assuming float32 fixture labels are exact.
    edges = graph["bus", "branch", "bus"].edge_index
    attributes = graph["bus", "branch", "bus"].edge_attr.double()
    matrix = torch.zeros((4, 4), dtype=torch.complex128)
    for (first, last), row in zip(edges.t().tolist(), attributes):
        resistance, reactance, gf, bf, gt, bt, tap, shift = row.tolist()
        series = 1 / complex(resistance, reactance)
        transformer = complex(tap * math.cos(shift), tap * math.sin(shift))
        matrix[first, first] += (series + complex(gf, bf)) / tap**2
        matrix[first, last] -= series / transformer.conjugate()
        matrix[last, last] += series + complex(gt, bt)
        matrix[last, first] -= series / transformer
    matrix += torch.diag(torch.complex(graph["bus"].shunt[:, 0].double(),
                                     graph["bus"].shunt[:, 1].double()))
    voltage = reference[:, 1].double() * torch.exp(1j * reference[:, 0].double())
    power = voltage * (matrix @ voltage).conj()
    oracle = torch.zeros((4, 2), dtype=torch.float64)
    oracle[2:] = -graph["PQ"].x.double() - torch.stack((power.real[2:], power.imag[2:]), dim=1)
    oracle[1, 0] = graph["PV"].x[0, 0].double() - power.real[1]
    assert torch.allclose(torch.tensor(record["residual"], dtype=torch.float64), oracle, atol=1e-12, rtol=0.)
    assert scored["PBL"] == pytest.approx(torch.linalg.vector_norm(oracle, dim=1).mean().item(), abs=1e-12)
    assert record["residual"][0] == [0., 0.]
    assert record["residual"][1][1] == 0.
    moved = reference.clone()
    moved[2, 0] += .02
    predicted = metric_input(graph, moved, "moved", "feasible", "N-1")
    assert score_sample(predicted)["PBL"] > .001
    assert predicted["from_bus"] == graph["bus", "branch", "bus"].edge_index[0].tolist()
    assert predicted["to_bus"] == graph["bus", "branch", "bus"].edge_index[1].tolist()
    # The offline scorer may read labels, but labels cannot alter its PF residual or denominator.
    poisoned = graph.clone()
    poisoned["PQ"].y += 9.
    rescored = metric_input(poisoned, moved, "poisoned", "feasible", "N-1")
    assert rescored["residual"] == predicted["residual"]
    assert rescored["abs_s_from_pu"] == predicted["abs_s_from_pu"]
    assert score_sample(rescored)["MSE"] > score_sample(predicted)["MSE"]
    moved[0, 0] += .01
    with pytest.raises(ValueError):
        metric_input(graph, moved, "invalid controls", "feasible", "N")


def test_unit_weight_control_reaches_a_finite_backward_path():
    import torch
    from circuitflow.api import ModelSpec, construct, forward
    from circuitflow.fixtures import four_bus
    from circuitflow.genco.adapter import PFAdapter
    torch.set_num_threads(1)
    graph, _ = four_bus()
    runtime = construct(ModelSpec(coarse_weighting="unit", max_component_buses=128))
    _, loss = forward(runtime, PFAdapter(power_scale_pu=2.5, generator_layout="aggregate_pf"),
                      graph, "cpu")
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [p.grad for p in runtime.model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_readout_scorer_requires_exact_bound_endpoint_before_opening_test(tmp_path, monkeypatch):
    from anchor_physics import score_readout
    path = frozen(tmp_path)
    job = anchortrain.selected_job(path, anchortrain.read_protocol(path)["jobs"][0]["id"])
    manifest = tmp_path / "manifest.json"
    write_new(manifest, {})
    run, readout = tmp_path / "run", tmp_path / "readout"
    run.mkdir()
    readout.mkdir()
    write_new(run / "COMPLETE.json", dict(
        complete=True, identity=dict(job=job, data_manifest_sha256=anchortrain.digest(manifest)),
        progress=dict(step=151999), inference_sha256="a" * 64))
    write_new(readout / "COMPLETE.json", dict(complete=True))
    monkeypatch.setattr("anchor_physics.load_pool", lambda *args: pytest.fail("TEST must stay closed"))
    with pytest.raises(ValueError, match="complete bound endpoint"):
        score_readout(run, readout, manifest, path)
