import torch
import torch.nn as nn
from layers import ClassifierLayer, ReadoutLayer, TopologyAwareGSSPoolingLayer, SAGELayer


class TopologyAwareGSSGC(nn.Module):
    def __init__(self, n_feat, n_hid, n_fc1, n_fc2, n_class, pooling_ratio, adj_bar_hop_order, variation_hop_order, loss=False):
        super(TopologyAwareGSSGC, self).__init__()
        self.gc1_1 = SAGELayer(n_feat, n_hid)
        self.gc1_2 = SAGELayer(n_hid, n_hid)
        self.gc1_3 = SAGELayer(n_hid, n_hid)

        self.pool = TopologyAwareGSSPoolingLayer(pooling_ratio, adj_bar_hop_order, variation_hop_order, loss=loss)

        self.gc2_1 = SAGELayer(3 * n_hid, n_hid)
        self.gc2_2 = SAGELayer(n_hid, n_hid)
        self.gc2_3 = SAGELayer(n_hid, n_hid)

        self.readout = ReadoutLayer('sum')

        self.classifier = ClassifierLayer(3 * n_hid, n_fc1, n_fc2, n_class)

    def forward(self, x, adj):
        x1 = self.gc1_1(x, adj)
        x2 = self.gc1_2(x1, adj)
        x3 = self.gc1_3(x2, adj)

        x_conv1 = torch.cat((x1, x2, x3), dim=1)

        x_conv1_pooled, adj_pooled = self.pool(x_conv1, adj)

        x4 = self.gc2_1(x_conv1_pooled, adj_pooled)
        x5 = self.gc2_2(x4, adj_pooled)
        x6 = self.gc2_3(x5, adj_pooled)

        x_conv2 = torch.cat((x4, x5, x6), dim=1)

        x_ro1 = self.readout(x_conv1)
        x_ro2 = self.readout(x_conv2)

        features = x_ro1 + x_ro2

        out = self.classifier(features)

        return out



