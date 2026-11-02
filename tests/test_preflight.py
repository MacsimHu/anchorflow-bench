import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import preflight
import run_company
from anchortrain import ROOT


class PreflightTests(unittest.TestCase):
    def test_existing_newer_torch_is_not_formal_environment(self):
        response = dict(python=[3, 10, 21],
                        packages={"torch": "2.6.0+cu124", "numpy": "1.26.4",
                                  "torch-geometric": "2.6.1"})
        with patch("preflight.subprocess.run", return_value=SimpleNamespace(stdout=json.dumps(response))):
            report = preflight.inspect_environment(Path("python"))
        self.assertFalse(report["formal_versions_match"])
        self.assertFalse(report["torch_import_attempted"])

    def test_formal_versions_do_not_claim_cuda_qualification(self):
        response = dict(python=[3, 10, 21],
                        packages={"torch": "2.1.2+cu118", "numpy": "1.26.4",
                                  "torch-geometric": "2.6.1"})
        with patch("preflight.subprocess.run", return_value=SimpleNamespace(stdout=json.dumps(response))):
            report = preflight.inspect_environment(Path("python"))
        self.assertTrue(report["formal_versions_match"])
        self.assertFalse(report["CUDA_verified"])

    def test_full_system_disk_is_rejected_before_writing(self):
        with patch("shutil.disk_usage", return_value=SimpleNamespace(free=6 * preflight.GIB)):
            with self.assertRaises(OSError):
                preflight.check_work_root(Path.cwd())

    def test_busy_gpu_is_never_probed(self):
        responses = [SimpleNamespace(stdout="GPU-owned\n"),
                     SimpleNamespace(stdout="GPU-owned, 1234\n")]
        with patch("preflight.subprocess.run", side_effect=responses) as run:
            with self.assertRaises(RuntimeError):
                preflight.probe_cuda(Path("python"), "GPU-owned", 120)
        self.assertEqual(run.call_count, 2)

    def test_driver_selects_exact_initial_seed_block(self):
        jobs = run_company.select_wave(ROOT / "COMPANY_QUEUE.json", ROOT / "company_h200.json",
                                       "case30", "Mixed", [11])
        self.assertEqual([(job["method"], job["variant"]) for job in jobs], [
            ("AnchorFlow", "Full"), ("AnchorFlow", "Deferred"),
            ("AnchorFlow", "Unit weights"), ("GENCO", "Full")])
        with self.assertRaises(ValueError):
            run_company.select_wave(ROOT / "COMPANY_QUEUE.json", ROOT / "company_h200.json",
                                    "case30", "Mixed", [11, 66])

    def test_driver_rejects_changed_queue_before_launch(self):
        queue = json.loads((ROOT / "COMPANY_QUEUE.json").read_text())
        queue["ready_order"][0]["seed"] = 66
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.json"
            path.write_text(json.dumps(queue))
            with self.assertRaises(ValueError):
                run_company.select_wave(path, ROOT / "company_h200.json", "case30", "Mixed", [11])


if __name__ == "__main__":
    unittest.main()
