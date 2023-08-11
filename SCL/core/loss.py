import torch
import torch.nn as nn
import faiss
import numpy as np


class ActionLoss(nn.Module):
    def __init__(self):
        super(ActionLoss, self).__init__()
        self.bce_criterion = nn.BCELoss()
        self.ce_criterion = nn.CrossEntropyLoss()

    def cls_criterion(self, inputs, label, uncertainty=None):
        if not uncertainty is None:
            loss1 = -torch.mean(torch.sum(torch.exp(-uncertainty) * torch.log(inputs.clamp(min=1e-7)) * label, dim=-1)) #(B, T, C) -> (B, T) -> 1
            loss2 = self.beta * torch.mean(uncertainty) #(B, T, 1) -> 1
            return loss1 + loss2
        else:
            return -torch.mean(torch.sum(torch.log(inputs.clamp(min=1e-7)) * label, dim=-1))

    def forward(self, video_scores, label):

        label = label / torch.sum(label, dim=1, keepdim=True)

        loss = self.bce_criterion(video_scores, label)
        return loss

class SniCoLoss(nn.Module):
    def __init__(self):
        super(SniCoLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def run_kmeans(self,X):

        print('performing kmeans clustering')
        results = {'im2cluster': [], 'centroids': [], 'density': []}
        for i in range(X.shape[0]):
            d = X.shape[2]
            k = 2
            #k设置成2,代表一个features里面可分为2大类,动作类和背景类
            clus = faiss.Clustering(d, k)
            clus.verbose = True
            clus.niter = 10
            clus.nredo = 4

            clus.max_points_per_centroid = 350
            clus.min_points_per_centroid = 10

            res = faiss.StandardGpuResources()
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False

            index = faiss.GpuIndexFlatL2(res, d, cfg)

            x = X[i].detach().cpu().numpy()

            clus.train(x, index)

            D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
            im2cluster = [int(n[0]) for n in I]

            # get cluster centroids
            centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

            # sample-to-centroid distances for each cluster
            Dcluster = [[] for c in range(k)]
            for im, i in enumerate(im2cluster):
                Dcluster[i].append(D[im][0])

            # concentration estimation (phi)
            density = np.zeros(k)
            for i, dist in enumerate(Dcluster):
                if len(dist) > 1:
                    d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                    density[i] = d

                    # if cluster only has one point, use the max to estimate its concentration
            dmax = density.max()
            for i, dist in enumerate(Dcluster):
                if len(dist) <= 1:
                    density[i] = dmax

            density = density.clip(np.percentile(density, 10),
                                   np.percentile(density, 90))  # clamp extreme values for stability
            density = 1 * density / density.mean()  # scale the mean to temperature

            # convert to cuda Tensors for broadcast
            centroids = torch.Tensor(centroids).cuda()
            centroids = nn.functional.normalize(centroids, p=2, dim=1)

            im2cluster = torch.LongTensor(im2cluster).cuda()
            density = torch.Tensor(density).cuda()

            results['centroids'].append(centroids)
            results['density'].append(density)
            results['im2cluster'].append(im2cluster)

        return results
    def NCE(self, q, k, neg, cluster_result,index,T=0.07):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss = self.ce_criterion(logits, labels)

        if cluster_result is not None:
            proto_labels = []
            proto_logits = []
            for n, (im2cluster, prototypes, density) in enumerate(
                    zip(cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density'])):

                # 统计cluster前面150个的标签.在f_all当中前150个tensor是动作
                t = torch.sum(im2cluster[:150])
                # 如果0多则0为动作
                if t < 75:
                    act = 0
                    bag = 1
                # 如果1多则1为动作
                else:
                    act = 1
                    bag = 0

                # 有了index和act之后进行比较
                # index = 0,明白此时是动作作为正样本,所以动作应该为positive,
                if index == 0:
                    pos_proto_id = im2cluster[[act]]
                    neg_proto_id = im2cluster[[bag]]
                else:
                    pos_proto_id = im2cluster[[bag]]
                    neg_proto_id = im2cluster[[act]]

                pos_prototypes = prototypes[pos_proto_id]

                neg_prototypes = prototypes[neg_proto_id]

                proto_selected = torch.cat([pos_prototypes, neg_prototypes])

                # compute prototypical logits
                logits_proto = torch.mm(q, proto_selected.t())

                # targets for prototype assignment，一共两个原型，所以prototype是0，1,2
                labels_proto = torch.linspace(0, 1, steps=q.size(0)).long().cuda()

                # scaling temperatures for the selected prototypes

                temp_proto = density[[0, 1]]

                logits_proto /= temp_proto

                proto_labels.append(labels_proto)
                proto_logits.append(logits_proto)

            # ProtoNCE loss

        proto_logits = None
        if proto_logits is not None:
            loss_proto = 0
            for proto_out, proto_target in zip(proto_logits, proto_labels):
                loss_proto += self.ce_criterion(proto_out, proto_target)
            # average loss across all sets of prototypes
            loss_proto /= 2
            loss += loss_proto


        return loss

    def forward(self, contrast_pairs,label):

        #拼接所有的向量，就是整个视频的feature
        f_all = torch.cat([contrast_pairs['EA'], contrast_pairs['EB'], contrast_pairs['HA'], contrast_pairs['HB']],dim=1)
        #cluster_result = self.run_kmeans(f_all)
        cluster_result = None

        HA_refinement = self.NCE(
            torch.mean(contrast_pairs['HA'], 1), 
            torch.mean(contrast_pairs['EA'], 1), 
            contrast_pairs['EB'],
            cluster_result,
            0
        )

        HB_refinement = self.NCE(
            torch.mean(contrast_pairs['HB'], 1),
            torch.mean(contrast_pairs['EB'], 1),
            contrast_pairs['EA'],
            cluster_result,
            1
        )



        loss = HA_refinement + HB_refinement
        return loss
        

class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()

    def forward(self, video_scores, label, contrast_pairs,video_scores_supp,fore_weights):

        loss_cls = self.action_criterion(video_scores, label,video_scores_supp,fore_weights,contrast_pairs)
        loss_snico = self.snico_criterion(contrast_pairs,label)
        loss_total = loss_cls + 0.01 * loss_snico

        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict
