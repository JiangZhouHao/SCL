import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage



# CoLA Pipeline
class CoLA(nn.Module):
    def __init__(self, cfg):
        super(CoLA, self).__init__()
        self.len_feature = cfg.FEATS_DIM
        self.num_classes = cfg.NUM_CLASSES

        self.f_embed = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )

        self.snippet_cls = nn.Conv1d(in_features=self.len_feature, out_features=20,kernel_size=1)



        self.softmax = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)

        self.r_easy = cfg.R_EASY
        self.r_hard = cfg.R_HARD
        self.m = cfg.m
        self.M = cfg.M

        self.dropout = nn.Dropout(p=0.6)


    def select_topk_embeddings(self, scores, embeddings, k):

        _, idx_DESC = scores.sort(descending=True, dim=1)
        idx_topk = idx_DESC[:, :k]
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])

        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings

    def easy_snippets_mining(self, actionness, embeddings, k_easy):

        select_idx = torch.ones_like(actionness).cuda()
        select_idx = self.dropout(select_idx)

        actionness_drop = actionness * select_idx

        actionness_rev = torch.max(actionness, dim=1, keepdim=True)[0] - actionness
        actionness_rev_drop = actionness_rev * select_idx

        easy_act = self.select_topk_embeddings(actionness_drop, embeddings, k_easy)
        easy_bkg = self.select_topk_embeddings(actionness_rev_drop, embeddings, k_easy)

        return easy_act, easy_bkg

    def hard_snippets_mining(self, actionness, embeddings, k_hard):


        aness_np = actionness.cpu().detach().numpy()

        aness_median = np.median(aness_np, 1, keepdims=True)
        aness_bin = np.where(aness_np > aness_median, 1.0, 0.0)



        erosion_M = ndimage.binary_erosion(aness_bin, structure=np.ones((1,self.M))).astype(aness_np.dtype)
        erosion_m = ndimage.binary_erosion(aness_bin, structure=np.ones((1,self.m))).astype(aness_np.dtype)
        idx_region_inner = actionness.new_tensor(erosion_m - erosion_M)
        aness_region_inner = actionness * idx_region_inner
        hard_act = self.select_topk_embeddings(aness_region_inner, embeddings, k_hard)

        dilation_m = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.m))).astype(aness_np.dtype)
        dilation_M = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.M))).astype(aness_np.dtype)
        idx_region_outer = actionness.new_tensor(dilation_M - dilation_m)
        aness_region_outer = actionness * idx_region_outer
        hard_bkg = self.select_topk_embeddings(aness_region_outer, embeddings, k_hard)


        return hard_act, hard_bkg

    def get_video_cls_scores(self, cas, k_easy):
        sorted_scores, _= cas.sort(descending=True, dim=1)
        topk_scores = sorted_scores[:, :k_easy, :]
        video_scores = self.softmax(topk_scores.mean(1))
        return video_scores

    def forward(self, x):
        device = x.device

        batch_size,num_segments = x.shape[0],x.shape[1]
        k_easy = num_segments // self.r_easy
        k_hard = num_segments // self.r_hard

        embeddings, cas, actionness = self.actionness_module(x)

        actionness_before1 = torch.roll(actionness, shifts=-1, dims=1)
        actionness_after1 = torch.roll(actionness,shifts=1,dims=1)
        actionness_before2 = torch.roll(actionness,shifts=-2,dims=1)
        actionness_after2 = torch.roll(actionness,shifts=2,dims=1)
        actionness = actionness + 0.1*actionness_before1 + 0.1*actionness_after1 + 0.02 * actionness_before2 + 0.02*actionness_after2


        easy_act, easy_bkg = self.easy_snippets_mining(actionness, embeddings, k_easy)
        hard_act, hard_bkg = self.hard_snippets_mining(actionness, embeddings, k_hard)

        video_scores = self.get_video_cls_scores(cas, k_easy)

        contrast_pairs = {
            'EA': easy_act,
            'EB': easy_bkg,
            'HA': hard_act,
            'HB': hard_bkg
        }

        return video_scores, contrast_pairs, actionness, cas
