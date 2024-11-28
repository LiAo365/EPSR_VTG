# Copyright (c) Ye Liu. Licensed under the BSD 3-Clause License.

import torch
import torch.nn as nn
from nncore.nn import MODELS, build_model
import torch.nn.functional as F


def calculate_norm(feature):
    f_norm = torch.norm(feature, p=2, dim=-1, keepdim=True)
    feature = feature / (f_norm + 1e-9)
    return feature


def random_walk(query_emb, video_emb, weight=0.5):
    """
    Random Walk Algorithm to realize the knowledge propagation from visual modal to language modal.
    Args:
        query_emb: the language embedding tensor with shape (B, L, C).
        video_emb: the visual knowledge tensor with shape (B, T*P, C).
        weight: the weight of visual knowledge, default is 0.5.
    """
    # normalize the visual and language embedding.
    # query_norm = calculate_norm(query_emb)
    # video_norm = calculate_norm(video_emb)
    query_norm = torch.nn.functional.normalize(query_emb, p=2, dim=-1)
    video_norm = torch.nn.functional.normalize(video_emb, p=2, dim=-1)
    eye_q = torch.eye(query_emb.size(1)).float().to(query_emb.device)

    # calculate the affinity matrix between visual and language embedding.
    latent_z = F.softmax(torch.einsum("bkd,btd->bkt", [video_norm, query_norm]) * 5.0, 1)
    norm_latent_z = latent_z / (latent_z.sum(dim=-1, keepdim=True) + 1e-9)
    affinity_mat = torch.einsum("bkt,bkd->btd", [latent_z, norm_latent_z])
    mat_inv_q = torch.inverse(eye_q - (weight**2) * affinity_mat)
    video2query_sum_q = weight * torch.einsum("bkt,bkd->btd", [latent_z, video_emb]) + query_emb
    refined_query = (1 - weight) * torch.einsum("btk,bkd->btd", [mat_inv_q, video2query_sum_q])

    return refined_query


@MODELS.register()
class R2Block(nn.Module):
    def __init__(self, dims, in_dims, k=4, dropout=0.5, use_tef=True, pos_cfg=None, tem_cfg=None):
        super(R2Block, self).__init__()

        # yapf:disable
        self.video_map = nn.Sequential(
            nn.LayerNorm((in_dims[0] + 2) if use_tef else in_dims[0]),
            nn.Dropout(dropout),
            nn.Linear((in_dims[0] + 2) if use_tef else in_dims[0], dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(dims),
            nn.Dropout(dropout),
            nn.Linear(dims, dims),
        )

        self.query_map = nn.Sequential(
            nn.LayerNorm(in_dims[1]),
            nn.Dropout(dropout),
            nn.Linear(in_dims[1], dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(dims),
            nn.Dropout(dropout),
            nn.Linear(dims, dims),
        )
        # yapf:enable

        if k > 1:
            # learnable gate for fusing the hidden states of different time steps.
            self.gate = nn.Parameter(torch.zeros([k - 1]))

        # just a map layers, to make up for video_map and query_map.
        self.v_map = nn.Linear(dims, dims)
        self.q_map = nn.Linear(dims, dims)
        # if the visual and query features are mapped to the same space, using same Linear layer may be better, if not, using Linear layer respectively.
        # self.v_q_map = nn.Linear(dims, dims)

        self.scale = nn.Parameter(torch.zeros([k]))

        self.pos = build_model(pos_cfg, dims=dims)
        self.tem = build_model(tem_cfg, dims=dims)  # nn.TransformerDecoderLayer

        self.dims = dims
        self.in_dims = in_dims
        self.k = k
        self.dropout = dropout
        self.use_tef = use_tef

    def forward(self, video_emb, query_emb, video_msk, query_msk, video_score=None):
        video_emb = video_emb[-self.k :]  # pick the last k layers features
        query_emb = query_emb[-self.k :]  # pick the last k layers features

        _, b, t, p, _ = video_emb.size()

        if self.use_tef:  # temporal positional encoding features
            tef_s = torch.arange(0, 1, 1 / t, device=video_emb.device)
            tef_e = tef_s + 1.0 / t
            tef = torch.stack((tef_s, tef_e), dim=1)
            tef = tef.unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(self.k, b, 1, p, 1)  # K * B * T * P * 2
            video_emb = torch.cat(
                (video_emb, tef[:, :, : video_emb.size(2)]), dim=-1
            )  # add temporal positional encoding features on the last dimension

        coll_v, coll_q, last = [], [], None
        for i in range(self.k - 1, -1, -1):
            # map visual and query features to the same dimension space using linear layers (MLP)
            v_emb = self.video_map(video_emb[i])  # B * T * P * C
            q_emb = self.query_map(query_emb[i])  # B * L * C

            coll_v.append(v_emb[:, :, 0])  # the first token of visual features is cls token, which is used for classification. B * T * C
            coll_q.append(q_emb)

            if video_score is None:
                # q_emb_max, _ = torch.max(q_emb, dim=1, keepdim=True)  # B * 1 * C
                q_emb_max = torch.mean(q_emb, dim=1, keepdim=True)  # B * 1 * C
                t_similarity = torch.bmm(v_emb[:, :, 0], q_emb_max.transpose(1, 2)) / self.dims**0.5  # B * T * 1
                video_score = t_similarity.softmax(-1)  # B * T * 1

            v_pool = v_emb.view(b * t, -1, self.dims)  # BT * P * C
            q_pool = q_emb.repeat_interleave(t, dim=0)  # BT * L * C

            # bidirectional random walk algorithm to realize the knowledge propagation between visual and language modal.
            # if video_score is not None:
            #     v_pool_map_select = v_pool * video_score.view(-1, 1, 1)
            #     q_pool_map_select = q_pool * video_score.view(-1, 1, 1)
            # v_pool_rk = random_walk(v_pool, q_pool_map_select)  # language to visual
            # q_pool_rk = random_walk(q_pool, v_pool_map_select)  # visual to language
            # v_pool = v_pool + v_pool_rk
            # q_pool = q_pool + q_pool_rk

            # map visual and query features to the same space using linear layers (MLP)
            v_pool_map = self.v_map(v_pool)  # BT * P * C
            q_pool_map = self.q_map(q_pool)  # BT * L * C

            # v_pool_map = self.v_q_map(v_pool_map)  # BT * P * C
            # q_pool_map = self.v_q_map(q_pool_map)  # BT * L * C

            # else:
            #     # q_emb_max, _ = torch.max(q_emb, dim=1, keepdim=True)  # B * 1 * C
            #     q_emb_max = torch.mean(q_emb, dim=1, keepdim=True)  # B * 1 * C
            #     t_similarity = torch.bmm(v_emb[:, :, 0], q_emb_max.transpose(1, 2))  # B * T * 1
            #     t_similarity = t_similarity.softmax(-1)  # B * T * 1
            #     v_pool_map_select = v_pool_map * t_similarity.view(-1, 1, 1)
            #     q_pool_map_select = q_pool_map * t_similarity.view(-1, 1, 1)

            # bidirectional random walk algorithm to realize the knowledge propagation between visual and language modal.
            if video_score is not None:
                v_pool_map_select = v_pool_map * video_score.view(-1, 1, 1)
                q_pool_map_select = q_pool_map * video_score.view(-1, 1, 1)
            v_pool_map = random_walk(v_pool_map, q_pool_map_select)  # language to visual
            q_pool_map = random_walk(q_pool_map, v_pool_map_select)  # visual to language

            # v_pool_map = v_pool_map + v_pool_map_rk
            # q_pool_map = q_pool_map + q_pool_map_rk

            # calculate the similarities for path-token pairs using normalized embedded Gaussian.
            att = torch.bmm(q_pool_map, v_pool_map.transpose(1, 2)) / self.dims**0.5
            att = att.softmax(-1)  # BT * L * P

            # pooling visual features into each token with residual connection
            o_pool = torch.bmm(att, v_pool) + q_pool  # BT * L * C
            o_pool = o_pool.amax(dim=1, keepdim=True)  # BT * 1 * C

            # combine the spatial pooling token with CLS token to generate query-modulated spatial features.
            v_emb = v_pool[:, 0, None] + o_pool * self.scale[i].tanh()
            v_emb = v_emb.view(b, t, self.dims)  # B * T * C

            if i < self.k - 1:
                # pooled visual features is fused with the previous hidden state using a gating mechanism.
                gate = self.gate[i].sigmoid()
                v_emb = gate * v_emb + (1 - gate) * last

            v_pe = self.pos(v_emb)
            # update the hidden state using a transformer decoder layer.
            last = self.tem(v_emb, q_emb, q_pe=v_pe, q_mask=video_msk, k_mask=query_msk)

        return last, q_emb, coll_v, coll_q
