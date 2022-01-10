from torch import nn

from serotiny.networks.weight_init import weight_init
from torch_geometric.nn import GraphConv


def _make_block(input_dim, output_dim):
    # return nn.Sequential(
    #     GraphConv(input_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU()
    block = []
    block.append(GraphConv(input_dim, output_dim))
    block.append(nn.BatchNorm1d(output_dim))
    block.append(nn.ReLU())
    return block


class GCN(nn.Module):
    def __init__(
        self, *dims, hidden_layers=[256],
    ):
        super().__init__()

        assert len(dims) >= 2

        self.input_dims = dims[:-1]
        self.output_dim = dims[-1]

        self.hidden_layers = hidden_layers

        net = [_make_block(sum(self.input_dims), hidden_layers[0])]

        net += [
            _make_block(input_dim, output_dim)
            for (input_dim, output_dim) in zip(hidden_layers[1:], hidden_layers[2:])
        ]
        net = [item for sublist in net for item in sublist]
        # import ipdb
        # ipdb.set_trace()
        net += [GraphConv(hidden_layers[-1], self.output_dim)]
        self.test = GraphConv(11, 256)
        self.net = nn.Sequential(*net)
        self.net.apply(weight_init)

    def forward(self, x1, edge_index, edge_weight=None):
        for index, layer in enumerate(self.net):
            if index % 3 == 0:
                x1 = layer(x1, edge_index, edge_weight)
            else:
                x1 = layer(x1)
        return x1

