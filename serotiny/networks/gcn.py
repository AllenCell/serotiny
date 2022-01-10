# from torch import nn

# from serotiny.networks.weight_init import weight_init
# from torch_geometric.nn import GraphConv


# def _make_block(input_dim, output_dim):
#     return nn.Sequential(
#         GraphConv(input_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU()
#     )


# class GCN(nn.Module):
#     def __init__(
#         self, *dims, hidden_layers=[256],
#     ):
#         super().__init__()

#         assert len(dims) >= 2

#         self.input_dims = dims[:-1]
#         self.output_dim = dims[-1]

#         self.hidden_layers = hidden_layers

#         net = [_make_block(sum(self.input_dims), hidden_layers[0])]

#         net += [
#             _make_block(input_dim, output_dim)
#             for (input_dim, output_dim) in zip(hidden_layers[1:], hidden_layers[2:])
#         ]

#         net += [GraphConv(hidden_layers[-1], self.output_dim)]

#         self.net = nn.Sequential(*net)
#         self.net.apply(weight_init)

#     def forward(self, x1, edge_index, edge_weight=False):
#         if edge_weight is not None:
#             return self.net(x1, edge_index, edge_weight)
#         else:
#             return self.net(x1, edge_index)
