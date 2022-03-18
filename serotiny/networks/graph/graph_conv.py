"""Basic Neural Net for Graph Classification."""

import torch
from torch.nn import functional as F

from torch_geometric.nn import GraphConv


class GraphNet(torch.nn.Module):
    def __init__(
        self, hidden_channels, out_channels, in_channels, task, add_self_loops
    ):
        super(GraphNet, self).__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        # self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=add_self_loops)  # noqa
        # self.conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=add_self_loops)  # noqa
        # self.conv3 = GCNConv(hidden_channels, hidden_channels, add_self_loops=add_self_loops)  # noqa
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.task = task

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        self.conv3.aggr = aggr

    def forward(self, x0, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x0, edge_index, edge_weight))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
        x3 = F.dropout(x3, p=0.2, training=self.training)

        if self.task == "classification":
            # x = torch.cat([x1, x2, x3], dim=-1)
            x = self.lin(x3)
            return x.log_softmax(dim=-1)
        else:
            return self.lin(x3)


class MLPNet(torch.nn.Module):
    def __init__(
        self, hidden_channels, out_channels, in_channels, task, add_self_loops
    ):
        super(MLPNet, self).__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.conv1 = torch.nn.Linear(in_channels, hidden_channels)
        self.conv2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.task = task

    def forward(self, x0, edge_index=None, edge_weight=None):
        x1 = F.relu(self.conv1(x0))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.relu(self.conv2(x1))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x3 = F.relu(self.conv3(x2))
        x3 = F.dropout(x3, p=0.2, training=self.training)

        if self.task == "classification":
            # x = torch.cat([x1, x2, x3], dim=-1)
            x = self.lin(x3)
            return x.log_softmax(dim=-1)
        else:
            return self.lin(x3)
