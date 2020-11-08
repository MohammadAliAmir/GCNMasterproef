import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt


def dicttotensor(d):
    tensors = []
    for i in d:
        t = torch.tensor(d[i])
        t = torch.unsqueeze(t, -1)
        tensors.append(t)
    return tensors


class Net(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        self.conv1 = GCNConv(1, hidden_size)
        self.conv2 = GCNConv(hidden_size, 1)

    def forward(self, x, edge_index):
        # edge_index = torch.from_numpy(preprocess_adj(edge_index)).float()
        x = x.double()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


nodes = np.arange(5)
edges = [(0, 1), (1, 3), (3, 1), (3, 4), (4, 2), (2, 4), (1, 2)]
graph = nx.DiGraph()
graph.add_nodes_from(nodes)
graph.add_edges_from(edges)
nx.draw(graph, with_labels=True)
plt.show()
data = {0: np.arange(24) + 0,
        1: np.arange(24) + 1,
        2: np.arange(24) + 2,
        3: np.arange(24) + 3,
        4: np.arange(24) + 4,
        5: np.arange(24) + 5}
val_data = {6: np.arange(24) + 6,
            7: np.arange(24) + 7}
# dense_adjacency = nx.to_pandas_adjacency(graph)
sparse_adj = nx.to_scipy_sparse_matrix(graph).tocoo()
sparse_adj_in_coo_format = np.stack([sparse_adj.row, sparse_adj.col])
sparse_adj_in_coo_format_tensor = torch.tensor(sparse_adj_in_coo_format, dtype=torch.long).cuda()

training_tensors = dicttotensor(data)
val_tensors = dicttotensor(val_data)
model = Net(training_tensors[0].shape[0])
learning_rate = 1e-3
model = model.double().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
criterion = nn.MSELoss().cuda()
model.train()

epoch_losses = []
for epoch in range(1000):
    for i in range(len(training_tensors)):
        if i + 1 < len(training_tensors):
            input = training_tensors[i].double().cuda()
            target = training_tensors[i + 1].double().cuda()
            output = model(input, sparse_adj_in_coo_format_tensor).double().cuda()
            # t_output=output.view(-1)
            loss = criterion(output, target)
            loss.backward()
            print(loss.item())
            epoch_losses.append(loss.item())

            for p in model.parameters():
                p.data.add_(p.grad.data, alpha=-learning_rate)

    optimizer.step()
model.eval()
with torch.no_grad():
    val_input = val_tensors[0].double().cuda()
    val_target = val_tensors[1].double().cuda()
    val_output = model(val_input, sparse_adj_in_coo_format_tensor)
    val_loss = criterion(val_output, val_target)
    print("Loss: " + str(val_loss.item()))
