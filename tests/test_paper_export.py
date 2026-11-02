import copy

import pytest

import anchortrain
from anchorbench import write_new
from exchange import ingest
from paper_export import export
from test_company import example, frozen


def cohort(tmp_path, count=5):
    protocol = frozen(tmp_path)
    jobs = [job for job in anchortrain.read_protocol(protocol)["jobs"]
            if (job["case"], job["regime"], job["method"], job["variant"])
            == ("case30", "Mixed", "AnchorFlow", "Full")]
    for job in jobs[:count]:
        value = copy.deepcopy(example(protocol))
        value.update({key: job[key] for key in ("case", "regime", "method", "variant", "seed")})
        value["coordinate_sha256"] = job["id"]
        source = tmp_path / (job["id"] + ".json")
        write_new(source, value)
        ingest(source, protocol, tmp_path / "repository")
    return protocol


def test_partial_cohort_cannot_become_a_paper_mean(tmp_path):
    protocol = cohort(tmp_path, 4)
    result = export(tmp_path / "repository", protocol)
    selected = next(c for c in result["cohorts"] if c["observed_seeds"])
    assert selected["status"] == "pending"
    assert selected["groups"] is None
    assert selected["observed_seeds"] == [11, 22, 33, 44]


def test_complete_cohort_preserves_partitions_seeds_and_quantile_estimand(tmp_path):
    protocol = cohort(tmp_path)
    result = export(tmp_path / "repository", protocol)
    selected = next(c for c in result["cohorts"] if c["status"] == "complete")
    assert selected["declared_seeds"] == [11, 22, 33, 44, 55]
    assert len(selected["groups"]) == 8
    assert selected["groups"][0]["metrics"]["PBL_p99_pu"]["seed_values"] == [.04] * 5
    assert selected["groups"][4]["cases_per_run"] == 600
    assert result["historical_results_merged"] is False


def test_incompatible_test_populations_are_not_pooled(tmp_path):
    protocol = cohort(tmp_path)
    source = next((tmp_path / "repository" / "results").glob("*/*/metrics.json"))
    import json
    value = json.loads(source.read_text())
    value["test_sha256"] = "b" * 64
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="identical input populations"):
        export(tmp_path / "repository", protocol)
