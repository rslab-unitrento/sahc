import typing

import torch
import torch.nn as nn
from torch.nn import functional as F

import hrlc_utils
from ._utils import BaseBackbone, VectorizedLevelHead

class VectorizedSAHCHead(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        lut: dict[hrlc_utils.Level, tuple[torch.Tensor, list[str]]],
        level_weights: dict[hrlc_utils.Level, float],
        mappings: dict[tuple[hrlc_utils.Level, hrlc_utils.Level], torch.Tensor],
        ensemble_mode: typing.Literal["geometric"] = "geometric",
        init_pos_bias: float = 0.5,
        init_neg_bias: float = -5.0,
        init_noise_std: float = 0.01,
        train_joints: bool = True,
        divergence_type: typing.Literal["jsd"] = "jsd",
        **kwargs,
    ):
        super().__init__()
        if ensemble_mode != "geometric":
            raise ValueError("Only geometric consensus is supported")
        if divergence_type != "jsd":
            raise ValueError("Only JSD consistency is supported")
        self.level_head = VectorizedLevelHead(
            n_features,
            n_hidden,
            lut,
            level_weights,
        )
        self.train_joints = train_joints
        self.divergence_type = divergence_type
        self.init_pos_bias = init_pos_bias
        self.init_neg_bias = init_neg_bias
        self.init_noise_std = init_noise_std
        self.ensemble_mode: typing.Literal["geometric"] = ensemble_mode
        self.head_info = self.level_head.head_info
        self.head_map = self.level_head.head_map
        self.num_heads = self.level_head.num_heads
        self.max_classes = self.level_head.max_classes
        valid_mask = self.level_head.valid_mask
        n_classes_tensor = self.level_head.n_classes_tensor

        self.log_temp = nn.Parameter(torch.zeros(self.num_heads))

        self.valid_mask: torch.Tensor
        self.register_buffer("valid_mask", valid_mask)
        self.n_classes_tensor: torch.Tensor
        self.register_buffer("n_classes_tensor", n_classes_tensor)

        edges_level, self.unique_joints_level, gather_level, trans_level = (
            self._build_graph(mappings)
        )
        if not self.train_joints:
            self.unique_joints_level.requires_grad_(False)
        self.edges_level: torch.Tensor
        self.register_buffer("edges_level", edges_level)
        self.gather_level: torch.Tensor
        self.register_buffer("gather_level", gather_level)
        self.trans_level: torch.Tensor
        self.register_buffer("trans_level", trans_level)

        self.level_counts: torch.Tensor
        self.register_buffer("level_counts", self._get_counts(self.edges_level))

    def _get_counts(self, edge_index: torch.Tensor) -> torch.Tensor:
        # Create a container of zeros, one slot for every Head ID
        counts = torch.zeros(self.num_heads, dtype=torch.float)

        if edge_index.numel() > 0:
            target_indices = edge_index[0]

            counts.index_add_(
                0, target_indices, torch.ones_like(target_indices, dtype=torch.float)
            )

        return counts

    def _build_graph(
        self,
        mappings: dict[tuple[hrlc_utils.Level, hrlc_utils.Level], torch.Tensor],
    ) -> tuple[torch.Tensor, nn.Parameter, torch.Tensor, torch.Tensor]:
        edges = []
        unique_joints_list = []
        pair_to_idx = {}

        gather_indices = []
        transpose_flags = []

        for t_idx, target_info in enumerate(self.head_info):
            for s_idx, source_info in enumerate(self.head_info):
                if t_idx == s_idx:
                    continue

                t_lvl = target_info["level"]
                s_lvl = source_info["level"]
                if t_lvl == s_lvl:
                    continue

                key_pair = tuple(sorted((t_lvl, s_lvl)))
                mapping = mappings.get(key_pair)
                if mapping is None:
                    continue

                low_idx, high_idx = sorted((t_idx, s_idx))
                pair_key = (low_idx, high_idx)

                if pair_key not in pair_to_idx:
                    canonical_target = self.head_info[high_idx]["level"]
                    full_joint = torch.full(
                        (self.max_classes, self.max_classes), 255.0
                    )

                    mat = mapping.t().float()
                    if canonical_target != key_pair[1]:
                        mat = mat.t()

                    vals = torch.zeros_like(mat)
                    vals[mat == 1] = self.init_pos_bias
                    vals[mat == 0] = self.init_neg_bias
                    vals += torch.randn_like(vals) * self.init_noise_std

                    h, w = mat.shape
                    full_joint[:h, :w] = vals
                    pair_to_idx[pair_key] = len(unique_joints_list)
                    unique_joints_list.append(full_joint)

                gather_indices.append(pair_to_idx[pair_key])
                edges.append([t_idx, s_idx])
                transpose_flags.append(s_idx == high_idx)

        if not edges:
            return (
                torch.empty((2, 0), dtype=torch.long),
                nn.Parameter(torch.empty(0)),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.bool),
            )

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        unique_joints = torch.stack(unique_joints_list).view(
            -1, self.max_classes, self.max_classes, 1, 1
        )
        gather_tensor = torch.tensor(gather_indices, dtype=torch.long)
        trans_tensor = torch.tensor(transpose_flags, dtype=torch.bool)

        return edge_index, nn.Parameter(unique_joints), gather_tensor, trans_tensor

    def _compute_vectorized_jsd(
        self,
        p_logits: torch.Tensor,
        q_logits: torch.Tensor,
        mask: torch.Tensor,
        dims_to_sum: list[int],
        divergence_type: str = "jsd",
    ) -> torch.Tensor:
        """
        Computes divergence between p and q, summing over specified dimensions.
        The consistency divergence is JSD.
        """
        p_log = p_logits
        q_log = q_logits
        p = p_log.exp()
        q = q_log.exp()

        if divergence_type != "jsd":
            raise ValueError("Only JSD consistency is supported")
        log_m = torch.clamp(0.5 * (p + q), min=1e-7).log()
        kl1 = F.kl_div(log_m, p_log, reduction="none", log_target=True)
        kl2 = F.kl_div(log_m, q_log, reduction="none", log_target=True)
        div = 0.5 * (kl1 + kl2)

        div = div.masked_fill(~mask, 0.0)

        return div.sum(dim=dims_to_sum)

    def _apply_consensus(
        self,
        logits: torch.Tensor,
        edges: torch.Tensor,
        joints: torch.Tensor,
        counts: torch.Tensor,
        compute_jsd: bool = True,
    ):
        B, n_heads, max_c, H, W = logits.shape

        if edges.numel() > 0:
            target_idx, source_idx = edges[0], edges[1]

            source_preds = torch.index_select(logits, 1, source_idx)

            joints_log_prob = joints.log_softmax(dim=2)

            source_lse = torch.logsumexp(source_preds, dim=2, keepdim=True)
            term = source_preds.unsqueeze(2) + joints_log_prob
            projected = torch.logsumexp(term, dim=3)
            projected = projected - source_lse  # [B, N_Edges, MaxC, H, W]

            consensus_sum = torch.zeros_like(logits)
            consensus_sum.index_add_(1, target_idx, projected.to(consensus_sum.dtype))

            denom = (counts + 1.0).view(1, n_heads, 1, 1, 1)
            consensus = ((logits + consensus_sum) / denom).log_softmax(2)

        else:
            consensus = logits
            projected = None

        jsd_per_head = torch.zeros([self.num_heads], device=logits.device)

        if compute_jsd and projected is not None:
            mask_heads = self.valid_mask.view(1, n_heads, max_c, 1, 1)

            dims_sum = [0, 2, 3, 4]
            term_self = self._compute_vectorized_jsd(
                logits,
                consensus,
                mask_heads,
                dims_sum,
                divergence_type=self.divergence_type,
            )

            if edges.numel() > 0:
                cons_edges = torch.index_select(consensus, 1, target_idx)

                mask_edges_flat = torch.index_select(self.valid_mask, 0, target_idx)
                mask_edges = mask_edges_flat.view(1, -1, max_c, 1, 1)

                term_edges_raw = self._compute_vectorized_jsd(
                    projected,  # type: ignore[arg-type]
                    cons_edges,
                    mask_edges,
                    dims_sum,
                    divergence_type=self.divergence_type,
                )

                term_neighbors = torch.zeros_like(term_self)
                term_neighbors.index_add_(
                    0, target_idx, term_edges_raw.to(term_neighbors.dtype)
                )
            else:
                term_neighbors = torch.zeros_like(term_self)

            total_jsd = term_self + term_neighbors

            avg_jsd = total_jsd / (counts + 1.0)

            norm_factor = B * H * W * torch.log(self.n_classes_tensor)

            jsd_per_head = avg_jsd / norm_factor

        return consensus, jsd_per_head

    def _reconstruct_joints(
        self,
        unique_joints: torch.Tensor,
        gather_idx: torch.Tensor,
        trans_flags: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct full joints tensor from unique parameters + metadata."""
        if gather_idx.numel() == 0:
            return unique_joints

        unique_joints = (
            unique_joints.view(unique_joints.shape[0], -1)
            .log_softmax(1)
            .view(unique_joints.shape)
        )

        joints = torch.index_select(unique_joints, 0, gather_idx)

        mask = trans_flags.view(-1, 1, 1, 1, 1)

        joints = torch.where(mask, joints.transpose(1, 2), joints)

        return joints.unsqueeze(0)

    def forward(self, x: torch.Tensor, compute_jsd: bool = True):
        logits_5d: torch.Tensor = self.level_head(x)

        T = self.log_temp.exp().clamp(min=0.1, max=10.0)
        T_reshaped = T.view(1, self.num_heads, 1, 1, 1)

        scaled_logits = logits_5d / T_reshaped
        log_probs = scaled_logits.log_softmax(2)

        full_joints_level = self._reconstruct_joints(
            self.unique_joints_level, self.gather_level, self.trans_level
        )
        level_cons, level_jsd = self._apply_consensus(
            log_probs,
            self.edges_level,
            full_joints_level,
            self.level_counts,
            compute_jsd,
        )

        outputs = dict(
            OG=logits_5d,
            SAHC=level_cons,
        )

        return outputs, level_jsd


class SAHCModel(nn.Module):
    def __init__(
        self,
        backbone: BaseBackbone,
        head: VectorizedSAHCHead,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor, compute_jsd: bool = True):
        feats = self.backbone(x)
        return self.head(feats, compute_jsd)
