"""Public benchmark preparation and on-demand graph separation."""
import gc
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import time
import urllib.request

from filelock import FileLock
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.separate import separate

from ._benchmark_source import PublicCaseDataset
from .genco.adapter import BRANCH
from .genco.normalization import TrainingPool, fit_power_scale
from .io import file_record, read_json, sha256, write_json


CASES = ("case30", "case118", "case500")
COUNTS = dict(train=48600, val=5400, test=6600)
REPOSITORY = "pfdelta/pfdelta"
API = "https://huggingface.co/api/datasets/"+REPOSITORY


def public_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def extract_public_archive(archive, destination):
    destination = Path(destination).resolve()
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        names = set()
        pending_members = []
        reused = 0
        for member in members:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or ".." in name.parts or "\\" in member.name
                    or not (member.isfile() or member.isdir())
                    or not (destination/member.name).resolve().is_relative_to(destination)):
                raise ValueError("The public archive contains a non-local or non-regular member")
            if member.isfile() and member.name in names:
                raise ValueError("The public archive repeats a file member")
            names.add(member.name)
            target = destination/member.name
            if member.isfile() and target.exists():
                if not target.is_file() or target.stat().st_size != member.size:
                    raise FileExistsError("An existing raw-data path differs; use a separate workspace")
                expected = hashlib.sha256()
                with source.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(4*1024**2), b""):
                        expected.update(block)
                if sha256(target) != expected.hexdigest():
                    raise FileExistsError("Never overwrite a different pre-existing raw data file")
                reused += 1
            else:
                pending_members.append(member)
        required = sum(member.size for member in pending_members if member.isfile())
        if shutil.disk_usage(destination).free < required+5*1024**3:
            raise OSError("Archive extraction would consume the five-GiB write/stop margin")
        source.extractall(destination, members=pending_members)
    return dict(members=len(members), verified_reused_files=reused, newly_extracted_bytes=required)


def download_public(case_root, case, revision=None):
    from huggingface_hub import hf_hub_download

    case_root = Path(case_root)
    receipt = case_root/"PUBLIC_DOWNLOAD.json"
    if receipt.exists():
        recorded = read_json(receipt)
        if recorded["case"] != case or (revision is not None and recorded["revision"] != revision):
            raise ValueError("An existing raw dataset belongs to a different source revision")
        return recorded
    info = public_json(API+("/revision/"+revision if revision else ""))
    revision = info["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("The public source did not provide an immutable revision")
    entries = public_json(API+"/tree/"+revision+"?recursive=false&expand=false")
    records = []
    for name in ("shuffle_files.tar.gz", case+".tar.gz"):
        entry, = [item for item in entries if item.get("path") == name]
        digest = entry.get("lfs", {}).get("oid")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("The official file lacks a verified LFS content identifier")
        if shutil.disk_usage(case_root).free < entry["size"]+5*1024**3:
            raise OSError("Downloading this archive would exhaust the write/stop margin")
        path = Path(hf_hub_download(
            REPOSITORY, filename=name, repo_type="dataset", revision=revision,
            local_dir=case_root/"downloads", token=False))
        if path.stat().st_size != entry["size"] or sha256(path) != digest:
            raise ValueError("The downloaded public archive differs from its immutable source")
        extraction = extract_public_archive(path, case_root)
        records.append(dict(file=name, sha256=digest, bytes=entry["size"], extraction=extraction))
    result = dict(repository=REPOSITORY, revision=revision, case=case, files=records,
                  license_metadata=info.get("cardData", {}).get("license"),
                  targets_used_for_selection=False)
    write_json(receipt, result)
    return result


def prune(data):
    keep = {"bus", "PQ", "PV", "slack"}
    if not keep.issubset(data.node_types):
        raise ValueError("The fixed-control benchmark graph lacks required node types")
    for kind in list(data.node_types):
        if kind not in keep:
            del data[kind]
    for relation in list(data.edge_types):
        if relation[0] not in keep or relation[2] not in keep:
            del data[relation]
    return data


class BenchmarkDataset(PublicCaseDataset):
    def download(self):
        if not (Path(self.root)/"PUBLIC_DOWNLOAD.json").is_file():
            raise RuntimeError("Download and bind the public source before preprocessing")

    def build_heterodata(self, pm_case, is_cpf_sample=False):
        return prune(super().build_heterodata(pm_case, is_cpf_sample=is_cpf_sample))

    def process(self):
        if self.task not in (1.1, 1.3) or self.case_name not in CASES:
            raise ValueError("This handoff only admits the declared single-grid tasks")
        output = Path(self.processed_dir)
        output.mkdir(parents=True, exist_ok=True)
        with FileLock(str(output/"PREPARE.lock"), timeout=0):
            for split in ("train", "val", "test"):
                target = output/(split+".pt")
                if target.exists():
                    raise FileExistsError("An existing processed split must not be overwritten")
                graphs = self.shuffle_split_and_save_data(self.root, split, save_subsplits=False)
                if len(graphs) != COUNTS[split]:
                    raise ValueError(f"The official {split} population changed; never truncate or resample")
                data, slices = self.collate(graphs)
                pending = target.with_name(target.name+".pending")
                with pending.open("xb") as stream:
                    torch.save((data, slices), stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(pending, target)
                del graphs, data, slices
                gc.collect()

    def load(self, split):
        self.data, self.slices = torch.load(
            str(Path(self.processed_dir)/(split+".pt")), map_location="cpu", mmap=True, weights_only=False)


class CachedPool(InMemoryDataset):
    def __init__(self, path, count, *, case):
        super().__init__(root=None)
        self.case_name = case
        self._data, self.slices = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
        if len(self) != count:
            raise ValueError("The complete frozen cache population differs")

    def get(self, index):
        data = separate(type(self._data), self._data, index, self.slices, decrement=False).clone()
        return prune(data)


def load_pool(manifest_path, split):
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    if manifest["schema"] != "circuitflow-public-benchmark-v1" or split not in COUNTS:
        raise ValueError("Use an explicitly prepared benchmark manifest and split")
    item = manifest["splits"][split]
    path = (manifest_path.parent/item["file"]).resolve()
    if not path.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError("The cache must remain within its declared data package")
    if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
        raise ValueError("The locally prepared cache changed")
    return CachedPool(path, item["count"], case=manifest["case"])


def input_identity(raw):
    value = hashlib.sha256()
    fields = [("bus", "bus_demand"), ("bus", "shunt"), (BRANCH, "edge_index"),
              (BRANCH, "edge_attr")]
    for kind in ("PQ", "PV", "slack"):
        fields.extend([(kind, "x"), ((kind, f"{kind}_link", "bus"), "edge_index")])
    for kind, name in fields:
        tensor = raw[kind][name].detach().cpu().contiguous()
        value.update(repr((kind, name, str(tensor.dtype), tuple(tensor.shape))).encode())
        value.update(tensor.numpy().tobytes())
    return value.hexdigest()


def prepare(data_root, case, task, *, revision=None):
    import psutil

    if case not in CASES or str(task) not in ("1.1", "1.3"):
        raise ValueError("Unsupported scientific cell")
    case_root = Path(data_root).resolve()/case
    case_root.mkdir(parents=True, exist_ok=True)
    available = psutil.virtual_memory().available
    required_ram = {"case30": 4, "case118": 10, "case500": 24}[case]*1024**3
    if available < required_ram:
        raise MemoryError("Insufficient currently available RAM for complete public preprocessing")
    if shutil.disk_usage(case_root).free < (32 if case == "case500" else 8)*1024**3:
        raise OSError("Reserve raw-data, collation-cache and checkpoint space before preprocessing")
    started = time.monotonic()
    source = download_public(case_root, case, revision)
    dataset = BenchmarkDataset(root_dir=str(Path(data_root).resolve()), case_name=case,
                               task=float(task), model="CANOS", split="train", add_bus_type=True)
    output = Path(dataset.processed_dir)
    del dataset
    gc.collect()
    return bind_existing(output, case, task, public_source=source, started=started)


def bind_existing(output, case, task, *, scale_file=None, public_source=None, started=None):
    if case not in CASES or str(task) not in ("1.1", "1.3"):
        raise ValueError("Unsupported scientific cell")
    output = Path(output).resolve(strict=True)
    manifest_path = output/"MANIFEST.json"
    if manifest_path.exists():
        raise FileExistsError("The data manifest is already frozen; reuse it, do not refit its scale")
    started = time.monotonic() if started is None else started
    source = public_source or dict(kind="explicitly_owned_existing_caches",
                                   upstream_revision="unverified_existing_cache",
                                   original_archive_identity_not_reconstructed=True)
    manifest = dict(schema="circuitflow-public-benchmark-v1", case=case, task=str(task),
                    public_source=source, splits={}, train_only_scaling=True,
                    deterministic_preprocessing_reads_all_three_official_splits=True,
                    TEST_predictions_or_selection=0)
    for split, count in COUNTS.items():
        path = output/(split+".pt")
        pool = CachedPool(path, count, case=case)
        if pool.get(0)["bus"].num_nodes != int(case[4:]):
            raise ValueError("A cached graph does not belong to the declared grid")
        manifest["splits"][split] = dict(file=path.name, count=count, **file_record(path))
        del pool
    gc.collect()
    if scale_file is None:
        train = CachedPool(output/"train.pt", COUNTS["train"], case=case)
        scale = fit_power_scale(TrainingPool(train, manifest["splits"]["train"]["sha256"]),
                                range(COUNTS["train"]), generator_layout="aggregate_pf")
    else:
        from .genco.normalization import PowerScale
        scale = PowerScale.from_dict(read_json(scale_file))
        if (scale.mode != "train_fitted" or scale.source_id != manifest["splits"]["train"]["sha256"]
                or scale.indices != tuple(range(COUNTS["train"])) or scale.generator_layout != "aggregate_pf"):
            raise ValueError("The existing scale does not bind this complete TRAIN cache")
    destination = output/"power_scale.json"
    if destination.exists():
        from .genco.normalization import PowerScale
        if PowerScale.from_dict(read_json(destination)).fingerprint != scale.fingerprint:
            raise FileExistsError("A different scale already occupies the target path")
    else:
        write_json(destination, scale.to_dict())
    manifest["power_scale"] = dict(file="power_scale.json", **file_record(output/"power_scale.json"))
    manifest["preparation_seconds"] = time.monotonic()-started
    write_json(manifest_path, manifest)
    return manifest_path
