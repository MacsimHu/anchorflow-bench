import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

import circuitflow.data as benchmark_data
from circuitflow._benchmark_source import PublicCaseDataset
from circuitflow.data import BenchmarkDataset


def test_combined_preparation_preserves_splits_without_redundant_caches(tmp_path):
    dataset = PublicCaseDataset.__new__(PublicCaseDataset)
    dataset.root = str(tmp_path)
    dataset.task, dataset.model = 1.1, "CANOS"
    dataset.task_config = {1.1: {"feasible": {"n": 10, "n-1": 0, "n-2": 0}}}
    dataset.test_config = {"feasible": {"n": 2, "n-1": 0, "n-2": 0}}
    raw = tmp_path/"n"/"raw"
    raw.mkdir(parents=True)
    shuffle = tmp_path/"shuffle_files"/"n"
    shuffle.mkdir(parents=True)
    (shuffle/"raw_shuffle.json").write_text(json.dumps({i: 11-i for i in range(12)}))
    for i in range(12):
        (raw/f"sample_{i+1}.json").write_text(json.dumps({"index": i+1}))

    def convert(sample, is_cpf_sample=False):
        graph = HeteroData()
        graph["bus"].x = torch.tensor([[sample["index"]]])
        return graph

    dataset.build_heterodata = convert
    expected = {"train": list(range(12, 3, -1)), "val": [3], "test": [2, 1]}
    for split, indices in expected.items():
        graphs = dataset.shuffle_split_and_save_data(str(tmp_path), split, save_subsplits=False)
        assert [int(graph["bus"].x[0, 0]) for graph in graphs] == indices
    assert not list(tmp_path.rglob("*.pt"))
    for split, indices in expected.items():
        graphs = dataset.shuffle_split_and_save_data(str(tmp_path), split)
        assert [int(graph["bus"].x[0, 0]) for graph in graphs] == indices
    assert len(list(tmp_path.rglob("*.pt"))) == 3


def test_failed_combined_write_never_publishes_a_partial_cache(tmp_path, monkeypatch):
    dataset = BenchmarkDataset.__new__(BenchmarkDataset)
    dataset.task, dataset.case_name = 1.1, "case30"
    dataset.root = str(tmp_path)
    dataset._custom_processed_dir = str(tmp_path/"processed")
    graph = HeteroData()
    graph["bus"].x = torch.ones(2, 1)

    def split_graphs(root, split, *, save_subsplits):
        assert save_subsplits is False
        return [graph, graph.clone()]

    def failed_write(value, stream):
        stream.write(b"incomplete")
        raise OSError("synthetic disk write failure")

    dataset.shuffle_split_and_save_data = split_graphs
    monkeypatch.setitem(benchmark_data.COUNTS, "train", 2)
    monkeypatch.setattr(torch, "save", failed_write)
    with pytest.raises(OSError, match="synthetic disk write failure"):
        dataset.process()
    assert not (Path(dataset.processed_dir)/"train.pt").exists()
    assert (Path(dataset.processed_dir)/"train.pt.pending").read_bytes() == b"incomplete"


def test_existing_combined_cache_is_rejected_before_parsing(tmp_path, monkeypatch):
    dataset = BenchmarkDataset.__new__(BenchmarkDataset)
    dataset.task, dataset.case_name = 1.1, "case30"
    dataset.root = str(tmp_path)
    dataset._custom_processed_dir = str(tmp_path)
    target = tmp_path/"train.pt"
    target.write_bytes(b"existing-cache")

    def unexpected_parse(*args, **kwargs):
        pytest.fail("Existing caches must be rejected before any parsing")

    monkeypatch.setattr(dataset, "shuffle_split_and_save_data", unexpected_parse)
    with pytest.raises(FileExistsError, match="must not be overwritten"):
        dataset.process()
    assert target.read_bytes() == b"existing-cache"
