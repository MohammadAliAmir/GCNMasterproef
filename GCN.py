import numpy as np
import pandas as pd
import networkx as nx
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
import random
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch,lib
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

np.random.seed(42)
def loaddata():
    if os.path.exists(r'graphwithwayid.csv'):
        Location = r'graphwithwayid.csv'
    else:
        Location = 'E:\PycharmProjects\graph_convolution\Data\Xian_city/raw/graph_withWayId_StreetType_CorrectDirs.csv'

    df = pd.read_csv(Location)
    return df

def load_avgspeed():
    if os.path.isfile('avg_speed.csv'):
        avg_speed = pd.read_csv('avg_speed.csv', sep=',', index_col=0)
    else:
        avg_speed = pd.read_csv('E:\PycharmProjects\graph_convolution\Data\Xian_city/raw/avg_speed.csv', sep=';',
                                index_col=0)
    return avg_speed



def savemodel(model):
    torch.save(model.state_dict(), './GCNModel.ckpt')

def loadmodel(m):
    m.load_state_dict(torch.load('./GCNModel.ckpt'))
    return m
# class GCNConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super(GCNConv, self).__init__(aggr='add')
#         self.lin = torch.nn.Linear(in_channels, out_channels)
#
#     def forward(self, x, edge_index):
#         # Step 1: Add self-loops
#         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
#
#         # Step 2: Multiply with weights
#         x = self.lin(x)
#
#         # Step 3: Calculate the normalization
#         row, col = edge_index
#         deg = degree(row, x.size(0), dtype=x.dtype)
#         deg_inv_sqrt = deg.pow(-0.5)
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
#
#         # Step 4: Propagate the embeddings to the next layer
#         return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
#                               norm=norm)
#
#     def message(self, x_j, norm):
#         # Normalize node features.
#         return norm.view(-1, 1) * x_j

class Net(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        self.conv1 = GCNConv(1, hidden_size)
        self.conv2 = GCNConv(hidden_size, 1)

    def forward(self, sample):
        x, edge_index = sample.x.double(), sample.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x #  F.log_softmax(x, dim=0)
def train(loader):
    model.train()
    print("In training")

    epoch_losses=[]
    for data_graph in loader:
        print(" ")
        model.zero_grad()
        output = model(data_graph)
        loss = criterion(output,data_graph.y).double()
        loss.backward()
        print("Loss: " + str(loss.item()))
        epoch_losses.append(loss.item())

        for p in model.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)

    return output , sum(epoch_losses)/len(epoch_losses)

def evaluate(set):
    val_outputs=[]
    losses=[]
    for entry in set:
        output=model(entry)
        target=entry.y
        val_loss=criterion(output,target).double()
        val_outputs.append(output)
        print("Validation Loss: "+str(val_loss.item()))
        losses.append(val_loss)

    return sum(losses)/len(losses)



if __name__ == '__main__':

    #Logger
    writer = SummaryWriter()

    #Load data
    data=loaddata()
    avg_speed=load_avgspeed()

    #Generate graph
    G= lib.generate_graph_nodes_roads(data)
    # Initial graph drawing parameters
    pos, widths, colors = lib.get_graph_params(G)
    #Update colors
    # c= lib.speedtocolor("2016-10-01T00:50:00", avg_speed,data)

    # Plot graph
    # nx.draw(G, pos=pos, node_size=5, width=widths, edge_color=c, with_labels=False)
    # plt.show()

    #Sort speed data to road order
    road_order = [str(x) for x in list(G.nodes)]
    avg_speed = avg_speed[road_order]
    # eventueel hiervoor al self loops introduceren

    #Get Adjacency Matrix
    adjacency_matrix = lib.get_adjacency_from_graph(G)

    #Define training and test set
    batch_size=1
    trainingset=avg_speed.loc["2016-10-01":"2016-11-01"]
    train_data, train_graph, scaler = lib.df_to_data(trainingset, adjacency_matrix, batch_size)
    testset=avg_speed.loc["2016-11-07":"2016-11-14"]
    test_data, test_graph = lib.df_to_data_val(testset, scaler, adjacency_matrix, batch_size)
    valset=avg_speed.loc["2016-11-18":"2016-11-19"]
    val_data, val_graph = lib.df_to_data_val(valset, scaler, adjacency_matrix, batch_size)

    #Initialize model
    inputsize = avg_speed.shape[1]
    training_losses=[]
    val_losses=[]
    test_losses=[]
    learning_rate= 1e-4
    accuracy_check=0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(train_graph[0]['x'].shape[0]).to(device)

    #Load model
    model=loadmodel(model)
    model=model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    criterion = nn.MSELoss().cuda()


    #Start training
    print("Start training")
    model.train()
    for epoch in range(15):
        print("Epoch: " + str(epoch))
        optimizer.zero_grad()
        out,epoch_loss = train(train_data)
        training_losses.append(epoch_loss)
        optimizer.step()

        # if (epoch % accuracy_check ==0):
        print("Evaluate")
        with torch.no_grad():
            model.eval()
            validation_loss=evaluate(val_data)
            val_losses.append(validation_loss)
            test_loss = evaluate(test_data)
            test_losses.append(test_loss)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), 'checkpoint/epoch_' + str(epoch) + '.tar')

    #Save model after training
    savemodel(model)

    # #Test Model
    # model.eval()
    # with torch.no_grad():
    #
    #Save losses
    for i in range(len(training_losses)):
        writer.add_scalar('Loss/train',training_losses[i],i)
    for i in range(len(val_losses)):
        writer.add_scalar('Loss/validation',val_losses[i],i)
    for i in range(len(test_losses)):
        writer.add_scalar('Loss/test',test_losses[i],i)



