import copy
import unittest

from anchorbench import campaign, score_sample, summarize


def sample(identity="one", partition="feasible", topology="N"):
    return dict(sample_id=identity, partition=partition, topology=topology,
                group_mse=dict(PQ=1., PV=2., reference=3., branch=4.),
                residual=[[3., 4.], [0., 0.], [0., 2.]],
                from_bus=[0, 0, 2], to_bus=[1, 2, 0],
                abs_s_from_pu=[2., 3., 1.], abs_s_to_pu=[1., 1., 4.])


class CampaignTests(unittest.TestCase):
    def test_matrix_has_finite_unique_unadmitted_coordinates(self):
        result = campaign()
        self.assertEqual(len(result["jobs"]), 386)
        self.assertEqual(sum(result["blocks"].values()), 386)
        self.assertEqual(len({j["coordinate_sha256"] for j in result["jobs"]}), 386)
        self.assertTrue(all(not j["training_enabled"] for j in result["jobs"]))
        self.assertFalse(result["frozen_launch_authority"])
        self.assertEqual(result["blocks"]["main_native"], 120)
        self.assertEqual(result["blocks"]["mechanism_core"], 80)
        self.assertEqual(result["tuning_coordinate_cap"], 72)

    def test_one_rpbl_uses_recorded_orientation_max_not_incident_sum(self):
        row = score_sample(sample())
        self.assertEqual(row["MSE"], 10.)
        self.assertAlmostEqual(row["PBL"], 7. / 3)
        self.assertAlmostEqual(row["rPBL_percent"], 100.)
        self.assertEqual(row["equation_max_abs_pu"], 4.)

    def test_threshold_uses_zero_contribution_and_percent_exactly_once(self):
        row = sample()
        row.update(residual=[[1., 0.]], from_bus=[0], to_bus=[0],
                   abs_s_from_pu=[1e-10], abs_s_to_pu=[0.])
        self.assertEqual(score_sample(row)["rPBL_percent"], 0.)
        row["abs_s_from_pu"] = [2.]
        self.assertEqual(score_sample(row)["rPBL_percent"], 50.)

    def test_partitions_are_separate_and_all_weights_actual_cases(self):
        a = sample("a")
        b = sample("b", topology="N-1")
        b["residual"] = [[0., 0.]] * 3
        c = sample("c", partition="near_infeasible")
        result = summarize([a, b, c])
        index = {(g["partition"], g["topology"]): g for g in result["groups"]}
        self.assertEqual(index[("feasible", "All")]["cases"], 2)
        self.assertAlmostEqual(index[("feasible", "All")]["PBL"], 7. / 6)
        self.assertEqual(index[("near_infeasible", "All")]["cases"], 1)
        self.assertFalse(result["partitions_pooled"])
        self.assertEqual(result["status"], "scored_not_admitted")
        self.assertEqual(index[("feasible", "All")]["equation_pass_fraction"]["0.001"], .5)

    def test_invalid_inputs_and_duplicate_samples_fail_explicitly(self):
        for problem in ("nonfinite", "bool", "negative", "endpoint", "length", "missing_group"):
            row = copy.deepcopy(sample())
            if problem == "nonfinite":
                row["residual"][0][0] = float("nan")
            elif problem == "bool":
                row["group_mse"]["PQ"] = True
            elif problem == "negative":
                row["abs_s_from_pu"][0] = -1.
            elif problem == "endpoint":
                row["to_bus"][0] = 3
            elif problem == "length":
                row["to_bus"].pop()
            else:
                del row["group_mse"]["PV"]
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                score_sample(row)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize([sample(), sample()])
        with self.assertRaisesRegex(ValueError, "empty"):
            summarize([])


if __name__ == "__main__":
    unittest.main()
