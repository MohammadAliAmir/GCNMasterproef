import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch, DataLoader


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

    def forward(self, sample):
        x, edge_index = sample.x, sample.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x #  F.log_softmax(x, dim=0)


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

frame_data = pd.DataFrame.from_dict(data)
data_graphs = []
for i in range(len(frame_data)-1):
    x = torch.tensor([frame_data.iloc[i]], dtype=torch.double).cuda()
    x = x.permute(1, 0)  # nodes, features
    y = torch.tensor([frame_data.iloc[i+1]], dtype=torch.double).cuda()
    y = y.permute(1, 0)  # nodes, features
    data_entry = Data(x=x, y=y, edge_index=sparse_adj_in_coo_format_tensor)
    data_graphs.append(data_entry)
loader = DataLoader(data_graphs, batch_size=1)


#training_tensors = dicttotensor(data)
#val_tensors = dicttotensor(val_data)
model = Net(data_graphs[0]['x'].shape[0])
learning_rate = 1e-4
model = model.double().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
criterion = nn.MSELoss().cuda()



model.train()

epoch_losses = 0
for epoch in range(1000):
    for data_entry in loader:
        output = model(data_entry)
        # t_output=output.view(-1)
        loss = criterion(output, data_entry.y)
        loss.backward()
        optimizer.step()
        #print(loss.item())
        epoch_losses += loss

        # for p in model.parameters():
        #     p.data.add_(p.grad.data, alpha=-learning_rate)
    print(epoch_losses)
    epoch_losses = 0


model.eval()
with torch.no_grad():
    val_input = val_tensors[0].double().cuda()
    val_target = val_tensors[1].double().cuda()
    val_output = model(val_input, sparse_adj_in_coo_format_tensor)
    val_loss = criterion(val_output, val_target)
    print("Loss: " + str(val_loss.item()))
