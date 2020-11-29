import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class SAGELayer(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(SAGELayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.w = Parameter(torch.FloatTensor(2 * in_feat, out_feat))
        nn.init.xavier_uniform_(self.w, gain=nn.init.calculate_gain('relu'))

    def forward(self, x, adj):
        if len(x.size()) == 1:
            h_mean_neighbors = adj.mm(x.view(-1, 1))
            h_old = torch.cat((x.view(-1, 1), h_mean_neighbors), dim=1)
        else:
            h_mean_neighbors = adj.mm(x)
            h_old = torch.cat((x, h_mean_neighbors), dim=1)

        h_new = torch.sigmoid(h_old.mm(self.w))
        if 0 in torch.norm(h_new, p=2, dim=1).view(-1, 1):
            print('\t HERE')
        h_new2 = h_new / torch.norm(h_new, p=2, dim=1).view(-1, 1)
        return h_new2


class TopologyAwareGSSPoolingLayer(nn.Module):
    def __init__(self, pooling_ratio, adj_bar_hop_order, variation_hop_order, loss=False):
        super(TopologyAwareGSSPoolingLayer, self).__init__()
        self.pooling_ratio = pooling_ratio
        self.adj_bar_hop_order = adj_bar_hop_order
        self.variation_hop_order = variation_hop_order
        self.loss = loss

    def forward(self, x, adj):
        n = adj.size()[0]
        adj_hat = torch.eye(n) + adj
        sample_size = int(self.pooling_ratio * n) if int(self.pooling_ratio * n) >= 2 else 1
        d = torch.diag(adj.sum(1))
        l = d - adj
        sampled_nodes = self.sampler(sample_size, x, adj, l, self.adj_bar_hop_order, self.variation_hop_order)
        c = self.compute_coarsening_matrix(sampled_nodes, sampled_nodes.size()[0], n)
        adj_s = torch.matrix_power(adj, 2)
        adj_tilda = torch.where(adj_s > 0, torch.tensor(1.), torch.tensor(0.)) - torch.diag(torch.diag(adj_s))
        new_adj = c.matmul(adj_tilda).matmul(c.T)
        new_x = adj_hat.mm(x)
        new_x = c.matmul(new_x)

        return new_x, new_adj

    def sampler(self, sample_size, signal, adj, l, adj_bar_hop_order, variation_hop_order):
        adj_bar = self.compute_adj_bar(adj, adj_bar_hop_order)

        variation_score = torch.flatten(self.compute_variation_score(signal, l, variation_hop_order))
        sample_matrix = torch.diag(variation_score).matmul(adj_bar).matmul(torch.diag(variation_score))

        sampled_nodes = set()

        for i in range(adj.size()[0]):
            quotient = int(sample_matrix.argmax().item() / adj.size()[1])
            remainder = torch.remainder(sample_matrix.argmax(), torch.tensor(adj.size()[1]))
            idx = [quotient, int(remainder.item())]

            if len(sampled_nodes) == 0:
                sampled_nodes.add(idx[0])
                sampled_nodes.add(idx[1])
                sample_matrix[idx[0], idx[1]] = -1.
                sample_matrix[idx[1], idx[0]] = -1.
            else:
                if 0 not in adj_bar[list(sampled_nodes), idx[0]]:
                    sampled_nodes.add(idx[0])
                if len(sampled_nodes) == sample_size:
                    break
                if 0 not in adj_bar[list(sampled_nodes), idx[1]]:
                    sampled_nodes.add(idx[1])
                if len(sampled_nodes) == sample_size:
                    break
                sample_matrix[idx[0], idx[1]] = -1.
                sample_matrix[idx[1], idx[0]] = -1.

        sampled_nodes = torch.tensor(list(sampled_nodes))

        return sampled_nodes

    @staticmethod
    def compute_adj_bar(adj, hop_order):
        new_adj = torch.zeros_like(adj)
        for i in range(hop_order):
            new_adj = new_adj + torch.matrix_power(adj, i + 1)

        new_adj = torch.ones_like(adj) - torch.clamp(new_adj, min=0, max=1)

        return new_adj

    @staticmethod
    def compute_variation_score(signal, l, hop_order):
        l = torch.matrix_power(l, hop_order)
        variations = torch.norm(torch.abs(l.matmul(signal)), p=2, dim=1)  # highest variations
        # zeroed = torch.where(variations == 0, torch.tensor(0.001, dtype=torch.float32), variations)
        # variation_score = 1. / zeroed
        return variations

    @staticmethod
    def compute_coarsening_matrix(sampled_nodes, k, n):
        c = torch.zeros((k, n), dtype=torch.float32)
        for i in range(c.size()[0]):
            for j in range(c.size()[1]):
                if j == sampled_nodes[i]:
                    c[i, j] = 1
        return c


class ReadoutLayer(nn.Module):
    def __init__(self, mode):
        super(ReadoutLayer, self).__init__()
        self.mode = mode

    def forward(self, x):
        if self.mode == 'sum':
            return x.sum(dim=0)


class BatchReadoutLayer(nn.Module):
    def __init__(self, mode='sum'):
        super(BatchReadoutLayer, self).__init__()
        self.mode = mode

    def forward(self, x):
        if self.mode == 'sum':
            return x.sum(dim=1)


class ClassifierLayer(nn.Module):
    def __init__(self, n_feat, n_hid1, n_hid2, n_classes):
        super(ClassifierLayer, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(n_feat, n_hid1),
                                        nn.ReLU(),
                                        nn.Linear(n_hid1, n_hid2),
                                        nn.ReLU(),
                                        nn.Linear(n_hid2, n_classes))

    def forward(self, x):
        return self.classifier(x)
