"""Canonical local rescoring of a verified fixed-endpoint voltage readout."""
import json
from pathlib import Path

import numpy as np
import torch

from anchorbench import METRIC_SCHEMA, summarize
from anchortrain import digest, selected_job
from circuitflow.data import input_identity, load_pool
from circuitflow.genco.adapter import BRANCH, analytic_branch_flows
from circuitflow.io import read_json


def metric_input(graph, voltage, identity, partition, topology):
    graph = graph.clone().to("cpu")
    state = torch.as_tensor(np.asarray(voltage).copy(), dtype=torch.float64)
    if state.shape != (graph["bus"].num_nodes, 2) or not torch.isfinite(state).all():
        raise ValueError("Readout state must contain finite [angle_rad, magnitude_pu] per bus")
    if torch.any(state[:, 1] <= 0):
        raise ValueError("Nonpositive voltage is an invalid prediction, not an omitted case")
    indices = {kind: graph[kind, f"{kind}_link", "bus"].edge_index[1]
               for kind in ("PQ", "PV", "slack")}
    for kind in ("PV", "slack"):
        known = graph[kind].x.double()
        actual = state[indices[kind]]
        if not torch.equal(actual[:, 1], known[:, 1]) or (
                kind == "slack" and not torch.equal(actual[:, 0], known[:, 0])):
            raise ValueError("The readout did not preserve the specified voltage controls")
    edges = graph[BRANCH].edge_index
    flows = analytic_branch_flows(state, edges, graph[BRANCH].edge_attr.double())
    injection = torch.zeros_like(state)
    injection.index_add_(0, edges[0], flows[:, :2])
    injection.index_add_(0, edges[1], flows[:, 2:])
    injection += state[:, 1:2].square() * graph["bus"].shunt.double() * torch.tensor([1., -1.])
    residual = torch.zeros_like(state)
    pq, pv, ref = (indices[k] for k in ("PQ", "PV", "slack"))
    residual[pq] = -graph["PQ"].x.double() - injection[pq]
    residual[pv, 0] = graph["PV"].x[:, 0].double() - injection[pv, 0]
    predictions = dict(
        PQ=state[pq], PV=torch.stack((injection[pv, 1], state[pv, 0]), dim=1),
        reference=injection[ref], branch=flows)
    targets = dict(PQ=graph["PQ"].y, PV=graph["PV"].y,
                   reference=graph["slack"].y, branch=graph[BRANCH].edge_label)
    mse = {}
    for kind in predictions:
        target = targets[kind].double()
        if target.shape != predictions[kind].shape or not torch.isfinite(target).all():
            raise ValueError("Completed-output targets differ from the benchmark contract")
        mse[kind] = float((predictions[kind] - target).square().mean()) if target.numel() else 0.
    return dict(sample_id=identity, partition=partition, topology=topology,
                group_mse=mse, residual=residual.tolist(),
                from_bus=edges[0].tolist(), to_bus=edges[1].tolist(),
                abs_s_from_pu=torch.linalg.vector_norm(flows[:, :2], dim=1).tolist(),
                abs_s_to_pu=torch.linalg.vector_norm(flows[:, 2:], dim=1).tolist())


def score_readout(run, readout, manifest_path, protocol_path):
    run, readout, manifest_path = Path(run), Path(readout), Path(manifest_path)
    complete = read_json(run / "COMPLETE.json")
    evaluation = read_json(readout / "COMPLETE.json")
    identity = complete["identity"]
    job = selected_job(protocol_path, identity["job"]["id"])
    if (identity["job"] != job or not complete["complete"] or not evaluation["complete"]
            or complete["progress"]["step"] != 152000
            or digest(manifest_path) != identity["data_manifest_sha256"]
            or digest(run / "inference.pt") != complete["inference_sha256"]
            or evaluation["model_sha256"] != complete["inference_sha256"]
            or digest(readout / "TEST.npy") != evaluation["TEST_sha256"]
            or evaluation["TEST_graphs"] != 6600
            or not read_json(readout / "QUALIFICATION.json")["passed"]):
        raise ValueError("Only a complete bound endpoint and qualified readout can become a result")
    manifest = read_json(manifest_path)
    if manifest["case"] != job["case"] or manifest["task"] != job["task"]:
        raise ValueError("The evaluation manifest changed scientific cell")
    pool = load_pool(manifest_path, "test")
    states = np.load(readout / "TEST.npy", mmap_mode="r", allow_pickle=False)
    if states.shape != (6600, int(job["case"][4:]), 2):
        raise ValueError("The readout population/shape differs")
    identities = [json.loads(line) for line in
                  (readout / "TEST_INPUTS.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(identities) != 6600 or [r["index"] for r in identities] != list(range(6600)):
        raise ValueError("Every ordered TEST input must be accounted for")
    def samples():
        for index in range(6600):
            graph = pool.get(index)
            fingerprint = input_identity(graph)
            if fingerprint != identities[index]["input_sha256"]:
                raise ValueError("Readout and scoring inputs are not identical")
            feasible = index < 6000
            offset, count = (index, 2000) if feasible else (index - 6000, 200)
            yield metric_input(
                graph, states[index], f"{manifest['splits']['test']['sha256']}:{index}",
                "feasible" if feasible else "near_infeasible", ("N", "N-1", "N-2")[offset // count])
    scored = summarize(samples())
    runtime = identity["runtime"]
    return dict(
        schema="anchorflow-public-result-v1", status="complete",
        protocol_sha256=digest(protocol_path), code_sha256=job["company_code_sha256"],
        data_manifest_sha256=digest(manifest_path),
        checkpoint_sha256=complete["inference_sha256"],
        test_sha256=manifest["splits"]["test"]["sha256"],
        readout_sha256=evaluation["TEST_sha256"],
        case=job["case"], regime=job["regime"], method=job["method"],
        variant=job["variant"], seed=job["seed"], coordinate_sha256=job["id"],
        actual_updates=complete["progress"]["step"],
        checkpoint_selection_rule="fixed_H200", metric_schema=METRIC_SCHEMA,
        runtime={key: runtime[key] for key in
                 ("python", "torch", "numpy", "torch_geometric", "device", "threads")},
        same_inference_for_all_metrics=True, partitions_pooled=False,
        groups=scored["groups"], scientific_claim="one run, not complete campaign or SOTA")
