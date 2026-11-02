# MIT; exact class bodies extracted from PFDelta pf_losses_utils.py.
import torch
from torch.nn.functional import mse_loss

class constraint_violations_loss_pf:
    """
    Constraint-violation metrics for PF-model outputs.

    This class computes several auxiliary constraint-violation metrics used to
    monitor or regularize models that predict bus and branch flows (e.g., CANOS).
    It aggregates per-bus real/reactive mismatches and per-branch flow mismatches
    into a single scalar loss while preserving the individual metrics for
    inspection.

    Attributes
    ----------
    constraint_loss : torch.Tensor or None
        Last computed aggregated scalar constraint loss.
    bus_real_mismatch : torch.Tensor or None
        Mean absolute real power mismatch per bus (last call).
    bus_reactive_mismatch : torch.Tensor or None
        Mean absolute reactive power mismatch per bus (last call).
    """

    def __init__(
        self,
    ):
        """Create a constraint_violations_loss_pf instance."""
        self.constraint_loss = None
        self.bus_real_mismatch = None
        self.bus_reactive_mismatch = None

    def __call__(self, output_dict, data):
        """
        Compute the constraint-violation loss and store component metrics.

        Parameters
        ----------
        output_dict : dict
            Model outputs containing predicted tensors for "bus", "PV", "slack",
            and "edge_preds" as produced by the model forward pass.
        data : torch_geometric.data.HeteroData or batch
            Ground-truth hetero data containing targets and structural metadata
            (edge_index, edge_label, shunts, bus-level aggregation tensors, etc.).

        Returns
        -------
        torch.Tensor
            Aggregated scalar loss equal to the sum of:
              mean(|real bus mismatch|) +
              mean(|reactive bus mismatch|) +
              mean(|real branch flow mismatch|) +
              mean(|reactive branch flow mismatch|)

        """
        device = data["bus"].x.device
        casename = output_dict["casename"]
        # Get the predictions and edge features
        bus_pred = output_dict["bus"]
        PV_pred = output_dict["PV"]
        slack_pred = output_dict["slack"]
        slack_demand = data["slack"].demand
        PV_generation = data["PV"].generation
        PV_demand = data["PV"].demand
        edge_pred = output_dict["edge_preds"]
        edge_indices = data["bus", "branch", "bus"].edge_index
        va, vm = bus_pred.T

        # # Unnormalize slack predictions
        # mean = canos_pfdelta_stats[casename]["mean"]["slack"]["y"].to(device)
        # std = canos_pfdelta_stats[casename]["std"]["slack"]["y"].to(device)
        # unnormalized_slack_pred = (slack_pred * std) + mean

        # Get the branch flows from the edge predictions
        n = data["bus"].x.shape[0]
        sum_branch_flows = torch.zeros(n, dtype=torch.cfloat, device=device)
        flows_rev = edge_pred[:, 2] + 1j * edge_pred[:, 3]
        flows_fwd = edge_pred[:, 0] + 1j * edge_pred[:, 1]
        sum_branch_flows.scatter_add_(0, edge_indices[0], flows_fwd)
        sum_branch_flows.scatter_add_(0, edge_indices[1], flows_rev)

        # Generator flows (already aggregated per bus)
        bus_gen = data["bus"].bus_gen.to(device)
        slack_idx = data["slack", "slack_link", "bus"].edge_index[1]
        pv_idx = data["PV", "PV_link", "bus"].edge_index[1]
        slack_pg = slack_pred[:, 0] + slack_demand[:, 0]
        slack_qg = slack_pred[:, 1] + slack_demand[:, 1]
        pv_pg = PV_generation[:, 0]
        pv_qg = PV_pred[:, 0] + PV_demand[:, 1]
        bus_gen[slack_idx] = torch.stack([slack_pg, slack_qg], dim=1)
        bus_gen[pv_idx] = torch.stack([pv_pg, pv_qg], dim=1)
        gen_flows = bus_gen[:, 0] + 1j * bus_gen[:, 1]

        # Demand flows (already aggregated per bus)
        bus_demand = data["bus"].bus_demand.to(device)
        demand_flows = bus_demand[:, 0] + 1j * bus_demand[:, 1]

        # Shunt admittances
        bus_shunts = data["bus"].shunt.to(device)
        shunt_flows = (torch.abs(vm) ** 2) * (
            bus_shunts[:, 0] - 1j * bus_shunts[:, 1]
        )  # (g_shunt - j*b_shunt)

        power_balance = gen_flows - demand_flows - shunt_flows - sum_branch_flows
        real_power_mismatch = torch.abs(torch.real(power_balance))
        reactive_power_mismatch = torch.abs(torch.imag(power_balance))

        # power: real and imaginary mismatches
        violation_degree_real_mismatch = real_power_mismatch.mean()
        violation_degree_imag_mismatch = reactive_power_mismatch.mean()

        # branch flows: ground truth mismatch, real
        p_flows_true = data["bus", "branch", "bus"].edge_label[
            :, 0
        ]  # forward active flow
        p_flows_mismatch = torch.real(flows_fwd) - p_flows_true
        violation_degree_real_flow_mismatch = torch.abs(p_flows_mismatch).mean()

        # branch flows: ground truth mismatch, reactive
        q_flows_true = data["bus", "branch", "bus"].edge_label[
            :, 1
        ]  # forward reactive flow
        q_flows_mismatch = torch.imag(flows_fwd) - q_flows_true
        violation_degree_imag_flow_mismatch = torch.abs(q_flows_mismatch).mean()

        # loss
        loss_c = (
            violation_degree_real_mismatch
            + violation_degree_imag_mismatch
            + violation_degree_real_flow_mismatch
            + violation_degree_imag_flow_mismatch
        )

        self.constraint_loss = loss_c
        self.bus_real_mismatch = violation_degree_real_mismatch
        self.bus_reactive_mismatch = violation_degree_imag_mismatch
        self.real_flow_mismatch_violation = violation_degree_real_flow_mismatch
        self.imag_flow_mismatch_violation = violation_degree_imag_flow_mismatch

        return loss_c

class CANOS_PF_MSE:
    """
    Mean-squared error loss for CANOS power-flow outputs.

    Computes the sum of mean-squared errors across the CANOS prediction
    components:
      - PV node outputs,
      - PQ node outputs,
      - Slack node outputs,
      - Branch/edge predictions.

    The last computed scalar loss is stored in `self.loss`.
    """

    def __init__(
        self,
    ):
        """Initialize storage for the last computed loss."""

        self.loss = None

    def __call__(self, output_dict, data):
        """
        Compute the CANOS MSE loss as the sum of per-component MSEs.

        Parameters
        ----------
        output_dict : dict
            Model outputs containing tensors for "PV", "PQ", "slack", and
            "edge_preds".
        data : torch_geometric.data.HeteroData or batched HeteroData
            Ground-truth hetero data containing the target tensors:
            data["PV"].y, data["PQ"].y, data["slack"].y and the branch
            targets at data["bus", "branch", "bus"].edge_label.

        Returns
        -------
        torch.Tensor
            Scalar tensor equal to MSE(PV_pred, PV_target) +
            MSE(PQ_pred, PQ_target) + MSE(slack_pred, slack_target) +
            MSE(edge_preds, branch_target).
        """
        PV_pred, PQ_pred, slack_pred = (
            output_dict["PV"],
            output_dict["PQ"],
            output_dict["slack"],
        )
        edge_preds = output_dict["edge_preds"]

        # gather targets
        PV_target, PQ_target, slack_target = data["PV"].y, data["PQ"].y, data["slack"].y
        branch_target = data["bus", "branch", "bus"].edge_label

        # calculate L2 loss
        pv_loss = mse_loss(PV_pred, PV_target)
        pq_loss = mse_loss(PQ_pred, PQ_target)
        slack_loss = mse_loss(slack_pred, slack_target)
        edge_loss = mse_loss(edge_preds, branch_target)

        total_loss = pv_loss + pq_loss + slack_loss + edge_loss
        self.loss = total_loss

        return total_loss
