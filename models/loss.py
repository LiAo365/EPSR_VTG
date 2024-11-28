# Copyright (c) Ye Liu. Licensed under the BSD 3-Clause License.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from nncore.nn import LOSSES, Parameter, build_loss, InfoNCELoss, TripletLoss


@LOSSES.register()
class C3LLoss(nn.Module):
    def __init__(self, loss_weight=0.01, temperature=0.07):
        super(C3LLoss, self).__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.infonce_loss = InfoNCELoss(temperature=temperature)
        self.triplet_loss = TripletLoss(margin=0.5)

    def _diversity_loss_(self, samples, lambda_center=0.2, lambda_diversity=0.05):
        center = samples[-1]  # using the last query embedding as the center.
        center = (center + samples.mean(dim=0)) / 2.0  # using the mean of the samples as the center.
        center = center.unsqueeze(0)

        loss_center = F.mse_loss(samples, center.expand_as(samples))

        cos_similarities = F.cosine_similarity(samples.unsqueeze(1), samples.unsqueeze(0), dim=-1)
        loss_diversity = 1.0 - cos_similarities

        loss_value = lambda_center * loss_center.mean() + lambda_diversity * loss_diversity.mean()

        return loss_value

    def _triplet_loss_(self, pos_samples, neg_samples, temperature=None, margin=0.5):
        anchor = pos_samples[-1].unsqueeze(0)  # shape: 1 * C, the last one is the query embedding.
        pos_samples = pos_samples[:-1]
        n, m = pos_samples.size(0), neg_samples.size(0)
        num = max(n, m)

        anchor = anchor.repeat_interleave(num, dim=0)  # shape: num * C
        # extend the positive samples and negative samples to the same number.
        if n < num:
            pos_samples = pos_samples.repeat_interleave(num // n, dim=0)
            pos_samples = torch.cat((pos_samples, pos_samples[: num % n]), dim=0)
        else:
            pos_samples = pos_samples[:num]

        if m < num:
            neg_samples = neg_samples.repeat_interleave(num // m, dim=0)
            neg_samples = torch.cat((neg_samples, neg_samples[: num % m]), dim=0)
        else:
            neg_samples = neg_samples[:num]

        loss_value = self.triplet_loss(pos_samples, neg_samples, anchor)
        return loss_value

    def _infonce_loss_(self, pos_samples, neg_samples, temperature=None):
        all_samples = torch.cat([pos_samples, neg_samples], dim=0)  # shape: (N+M) x C
        # The cosine similarity is scaled by the temperature
        sim_matrix = F.cosine_similarity(pos_samples.unsqueeze(1), all_samples.unsqueeze(0), dim=-1)  # / temperature  # shape: N x (N+M)
        # Create the target labels: the positive samples should match only themselves
        target = torch.arange(pos_samples.size(0), device=pos_samples.device)  # shape: N (correct class indices)
        # Apply cross-entropy loss over the softmax-normalized similarity matrix
        loss_value = F.cross_entropy(sim_matrix, target)
        return loss_value

    def _pull_push_loss_(self, pos_samples, neg_samples, temperature=None, margin=0.5):
        # calculate the loss value using pull-push loss.
        # calculate the cosine similarity among the positive samples.
        pos_sim = F.cosine_similarity(pos_samples.unsqueeze(1), pos_samples.unsqueeze(0), dim=-1) / temperature  # shape N * N
        # calculate the cosine similarity between the positive samples and the negative samples.
        neg_sim = F.cosine_similarity(pos_samples.unsqueeze(1), neg_samples.unsqueeze(0), dim=-1) / temperature  # shape N * M
        # Loss calculation
        # 1. Maximize positive similarity by pulling positive samples closer
        # pos_loss = F.relu(1.0 - pos_sim)  # Minimize 1 - cos_sim for positive samples pairs
        # 2. Maximize negative similarity by pushing negative samples away
        # neg_loss = F.relu(neg_sim - margin)  # Minimize max(0, cos_sim - margin) for negative samples pairs
        # calculate the loss value.
        loss_value = -torch.log(torch.exp(pos_sim).sum(dim=1) / (torch.exp(pos_sim).sum(dim=1) + torch.exp(neg_sim).sum(dim=1))).mean()
        # Total loss
        # loss_value = pos_loss.mean() + neg_loss.mean()
        return loss_value

    def forward(self, pos_samples, neg_samples, temperature=None, margin=0.5):
        """
        This loss is used to pull the positive samples closer and push the negative samples away.
        Args:
            pos_samples: the positive samples, shape: N * C, N is the number of samples, C is the dimension of the samples.
            neg_samples: the negative samples, shape: M * C, M is the number of samples.
            temperature: the temperature of the loss. if None, use the default temperature.
        Returns:
            loss_value: the loss value.
        """
        # make sure the shape is correct. if not, add the unsqueeze operation.
        if pos_samples.dim() == 1:
            pos_samples = pos_samples.unsqueeze(0)
        if neg_samples.dim() == 1:
            neg_samples = neg_samples.unsqueeze(0)

        if temperature is None:
            temperature = self.temperature

        # 1. using pull-push loss to calculate the loss value. (Fewer Improvement)
        loss_value = self._pull_push_loss_(pos_samples, neg_samples, temperature, margin)

        # 2. using infonce loss to calculate the loss value. ()
        # loss_value = self._infonce_loss_(pos_samples, neg_samples, temperature)

        # 3. using triplet loss to calculate the loss value. (Fewer Improvement)
        # loss_value = self._triplet_loss_(pos_samples, neg_samples, temperature, margin)

        # 4. using diversity loss to calculate the loss value.
        # loss_diversity = self._diversity_loss_(neg_samples)

        # return (loss_value + loss_diversity) * self.loss_weight
        return loss_value * self.loss_weight


@LOSSES.register()
class SampledNCELoss(nn.Module):

    def __init__(self, temperature=0.07, max_scale=100, learnable=False, direction=("row", "col"), loss_weight=1.0):
        super(SampledNCELoss, self).__init__()

        scale = torch.Tensor([math.log(1 / temperature)])

        if learnable:
            self.scale = Parameter(scale)
        else:
            self.register_buffer("scale", scale)

        self.temperature = temperature
        self.max_scale = max_scale
        self.learnable = learnable
        self.direction = (direction,) if isinstance(direction, str) else direction
        self.loss_weight = loss_weight

    def extra_repr(self):
        return "temperature={}, max_scale={}, learnable={}, direction={}, loss_weight={}".format(
            self.temperature, self.max_scale, self.learnable, self.direction, self.loss_weight
        )

    def forward(self, video_emb, query_emb, video_msk, saliency, pos_clip):
        batch_inds = torch.arange(video_emb.size(0), device=video_emb.device)

        pos_scores = saliency[batch_inds, pos_clip].unsqueeze(-1)
        loss_msk = (saliency <= pos_scores) * video_msk

        scale = self.scale.exp().clamp(max=self.max_scale)
        i_sim = F.cosine_similarity(video_emb, query_emb, dim=-1) * scale
        i_sim = i_sim + torch.where(loss_msk > 0, 0.0, float("-inf"))

        loss = 0

        if "row" in self.direction:
            i_met = F.log_softmax(i_sim, dim=1)[batch_inds, pos_clip]
            loss = loss - i_met.sum() / i_met.size(0)

        if "col" in self.direction:
            j_sim = i_sim.t()
            j_met = F.log_softmax(j_sim, dim=1)[pos_clip, batch_inds]
            loss = loss - j_met.sum() / j_met.size(0)

        loss = loss * self.loss_weight
        return loss


@LOSSES.register()
class BundleLoss(nn.Module):

    def __init__(self, sample_radius=1.5, loss_cls=None, loss_reg=None, loss_sal=None, loss_video_cal=None, loss_layer_cal=None, loss_motion=None):
        super(BundleLoss, self).__init__()

        self._loss_cls = build_loss(loss_cls)
        self._loss_reg = build_loss(loss_reg)
        self._loss_sal = build_loss(loss_sal)
        self._loss_video_cal = build_loss(loss_video_cal)
        self._loss_layer_cal = build_loss(loss_layer_cal)
        self._loss_motion = build_loss(loss_motion)

        self.sample_radius = sample_radius

    def get_target_single(self, point, gt_bnd, gt_cls):
        num_pts, num_gts = point.size(0), gt_bnd.size(0)

        lens = gt_bnd[:, 1] - gt_bnd[:, 0]  # calculate the length of all the ground truth boundaries.
        lens = lens[None, :].repeat(num_pts, 1)  # repeat the lens to the same size of the point.

        gt_seg = gt_bnd[None].expand(num_pts, num_gts, 2)  # expand the ground truth boundary: num_pts * num_gts * 2, 2 means the start and end.
        s = point[:, 0, None] - gt_seg[:, :, 0]  # calculate the distance between the start of the point and the start of the ground truth boundary.
        e = gt_seg[:, :, 1] - point[:, 0, None]  # calculate the distance between the end of the point and the end of the ground truth boundary.
        r_tgt = torch.stack((s, e), dim=-1)  # shape: num_pts * num_gts * 2

        if self.sample_radius > 0:
            center = (gt_seg[:, :, 0] + gt_seg[:, :, 1]) / 2  # calculate the center of the ground truth boundary.
            t_mins = center - point[:, 3, None] * self.sample_radius
            t_maxs = center + point[:, 3, None] * self.sample_radius
            dist_s = point[:, 0, None] - torch.maximum(t_mins, gt_seg[:, :, 0])
            dist_e = torch.minimum(t_maxs, gt_seg[:, :, 1]) - point[:, 0, None]
            center = torch.stack((dist_s, dist_e), dim=-1)  # shape: num_pts * num_gts * 2
            cls_msk = center.min(-1)[0] >= 0  # num_pts * num_gts
        else:
            cls_msk = r_tgt.min(-1)[0] >= 0

        reg_dist = r_tgt.max(-1)[0]  # regression distance
        reg_msk = torch.logical_and((reg_dist >= point[:, 1, None]), (reg_dist <= point[:, 2, None]))

        lens.masked_fill_(cls_msk == 0, float("inf"))  # fill the lens with inf if the cls_msk is 0.
        lens.masked_fill_(reg_msk == 0, float("inf"))  # fill the lens with inf if the reg_msk is 0.
        min_len, min_len_inds = lens.min(dim=1)

        min_len_mask = torch.logical_and((lens <= (min_len[:, None] + 1e-3)), (lens < float("inf"))).to(r_tgt.dtype)

        label = F.one_hot(gt_cls[:, 0], 2).to(r_tgt.dtype)
        c_tgt = torch.matmul(min_len_mask, label).clamp(min=0.0, max=1.0)[:, 1]
        r_tgt = r_tgt[range(num_pts), min_len_inds] / point[:, 3, None]

        # c_tgt: num_pts, represents the cls target, 1 means the positive sample, 0 means the negative sample.
        # r_tgt: num_pts * 2, represents the reg target, the first column is the start distance with the nearest monment bou
        return c_tgt, r_tgt

    def get_target(self, data):
        cls_tgt, reg_tgt = [], []

        for i in range(data["boundary"].size(0)):
            gt_bnd = data["boundary"][i] * data["fps"][i]
            gt_cls = gt_bnd.new_ones(gt_bnd.size(0), 1).long()  # create a tensor filled with 1, with the same size of gt_bnd.

            c_tgt, r_tgt = self.get_target_single(data["point"], gt_bnd, gt_cls)

            cls_tgt.append(c_tgt)
            reg_tgt.append(r_tgt)

        cls_tgt = torch.stack(cls_tgt)
        reg_tgt = torch.stack(reg_tgt)

        return cls_tgt, reg_tgt

    def loss_cls(self, data, output, cls_tgt):
        src = data["out_class"].squeeze(-1)
        msk = torch.cat(data["pymid_msk"], dim=1)

        loss_cls = self._loss_cls(src, cls_tgt, weight=msk, avg_factor=msk.sum())

        output["loss_cls"] = loss_cls
        return output

    def loss_reg(self, data, output, cls_tgt, reg_tgt):
        src = data["out_coord"]
        msk = cls_tgt.unsqueeze(2).repeat(1, 1, 2).bool()

        loss_reg = self._loss_reg(src, reg_tgt, weight=msk, avg_factor=msk.sum())

        output["loss_reg"] = loss_reg
        return output

    def loss_sal(self, data, output):
        video_emb = data["video_emb"]
        query_emb = data["query_emb"]
        video_msk = data["video_msk"]

        saliency = data["saliency"]
        pos_clip = data["pos_clip"][:, 0]

        output["loss_sal"] = self._loss_sal(video_emb, query_emb, video_msk, saliency, pos_clip)
        return output

    def loss_cal(self, data, output):
        pos_clip = data["pos_clip"][:, 0]

        batch_inds = torch.arange(pos_clip.size(0), device=pos_clip.device)

        coll_v_emb, coll_q_emb = [], []
        for v_emb, q_emb in zip(data["coll_v"], data["coll_q"]):
            v_emb_pos = v_emb[batch_inds, pos_clip]
            q_emb_pos = q_emb[:, 0]

            coll_v_emb.append(v_emb_pos)
            coll_q_emb.append(q_emb_pos)

        v_emb = torch.stack(coll_v_emb)
        q_emb = torch.stack(coll_q_emb)
        output["loss_video_cal"] = self._loss_video_cal(v_emb, q_emb)

        v_emb = torch.stack(coll_v_emb, dim=1)
        q_emb = torch.stack(coll_q_emb, dim=1)
        output["loss_layer_cal"] = self._loss_layer_cal(v_emb, q_emb)

        return output

    def _loss_motion_by_saliency_(self, data, output):
        temporal_saliency = data[
            "saliency"
        ]  # shape: B * T, a binary tensor, 1 means the salient moment(positive sample), 0 means the non-salient moment(negative sample).
        loss_values = []
        for v_emb, q_emb in zip(data["coll_v"], data["coll_q"]):  # different layer cls token embeddings.
            video_cl_loss = []
            for idx, saliency in enumerate(temporal_saliency):  # different video instance in the batch.
                pos_samples = v_emb[idx][saliency.bool()]
                pos_samples = torch.cat((q_emb[idx], pos_samples), dim=0)
                neg_samples = v_emb[idx][~saliency.bool()]
                video_cl_loss.append(self._loss_motion(pos_samples.squeeze(0), neg_samples.squeeze(0)))
            loss_values.append(torch.stack(video_cl_loss).mean())

        loss_value = torch.stack(loss_values).mean()
        if "loss_motion" in output:
            output["loss_motion"] = output["loss_motion"] + loss_value
        else:
            output["loss_motion"] = loss_value
        return output

    def _loss_motion_by_boundary_(self, data, output):
        # boundary is the original boundary annotation, Must multiply by fps to get the correct boundary.
        boundary = data["boundary"]
        boundary_idxs = []
        for idx in range(boundary.size(0)):
            boundary[idx] = boundary[idx] * data["fps"][idx]
            # cast boundary to int type, for start, using the floor, for end, using the ceil.
            boundary_idxs.append((math.floor(boundary[idx][0][0].item()), math.ceil(boundary[idx][0][1].item())))

        # contrastive learning loss should be calculated by the video cls token embeddings and corresponding query embeddings.
        cl_loss = []
        for v_emb, q_emb in zip(data["coll_v"], data["coll_q"]):  # different layer cls token embeddings.
            video_cl_loss = []
            for idx, (start, end) in enumerate(boundary_idxs):  # different video instance in the batch.
                pos_samples = v_emb[idx][start:end]
                pos_samples = torch.cat((q_emb[idx], pos_samples), dim=0)

                neg_samples = torch.cat((v_emb[idx][:start], v_emb[idx][end:]), dim=0)
                video_cl_loss.append(self._loss_motion(pos_samples.squeeze(0), neg_samples.squeeze(0)))
            cl_loss.append(torch.stack(video_cl_loss).mean())

        loss_value = torch.stack(cl_loss).sum()
        if "loss_motion" in output:
            output["loss_motion"] = output["loss_motion"] + loss_value
        else:
            output["loss_motion"] = loss_value
        return output

    # def _video_score_loss_(self, data, output):
    #     saliency = data["saliency"]
    #     video_score = data["video_score"]

    def loss_motion(self, data, output):
        if "saliency" in data and "boundary" in data:
            output = self._loss_motion_by_boundary_(data, output)
        elif "saliency" in data:
            output = self._loss_motion_by_saliency_(data, output)
        elif "boundary" in data:
            output = self._loss_motion_by_boundary_(data, output)
        else:
            pass

        # if "saliency" in data and "video_score" in data:

        return output

    def forward(self, data, output):
        if self._loss_reg is not None:
            cls_tgt, reg_tgt = self.get_target(data)
            # here change the cls_tgt to the saliency, different the origin code.
            # cls_tgt = data["saliency"][:, data["point"][:, 0].int()]
            output = self.loss_reg(data, output, cls_tgt, reg_tgt)

        else:
            cls_tgt = data["saliency"]

        if self._loss_cls is not None:
            output = self.loss_cls(data, output, cls_tgt)

        if self._loss_sal is not None:
            output = self.loss_sal(data, output)

        if self._loss_video_cal is not None or self._loss_layer_cal is not None:
            output = self.loss_cal(data, output)

        if self._loss_motion is not None:
            output = self.loss_motion(data, output)

        return output
