import numpy as np
import pandas as pd
import networkx as nx
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv


np.random.seed(42)
def loaddata():
    Location = 'E:\PycharmProjects\graph_convolution\Data\Xian_city/raw/graph_withWayId_StreetType_CorrectDirs.csv'
    # Location = r'graphwithwayid.csv'
    df = pd.read_csv(Location)
    return df

def loadavgspeed():
    Location = 'E:\PycharmProjects\graph_convolution\Data\Xian_city/raw/avg_speed.csv'
    # Location = r'avg_speed.csv'
    df = pd.read_csv(Location, sep=';')
    return df

def convert_index_to_datetime(feature):
        indices = pd.to_datetime(feature.index)
        feature.index = indices
        return feature

def preprocess_adj(A):
    '''
    Pre-process adjacency matrix
    :param A: adjacency matrix
    :return:
    '''
    I = np.eye(A.shape[0])
    A_hat = A + I # add self-loops
    D_hat_diag = np.sum(A_hat, axis=1)
    D_hat_diag_inv_sqrt = np.power(D_hat_diag, -0.5)
    D_hat_diag_inv_sqrt[np.isinf(D_hat_diag_inv_sqrt)] = 0.
    D_hat_inv_sqrt = np.diag(D_hat_diag_inv_sqrt)
    return np.dot(np.dot(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
def speedtocolor(date):
    # speed=time_dict[date]
    speed=avg_speed.loc[date]
    keys=speed.keys()
    colors1=[]
    df=data['edgeId'].tolist()
    for i in keys:
        if int(i) in df:
            j=speed.get(i)
            if j>49:
                colors1.append('green')
            elif speed.get(i)>27:
                colors1.append('orange')
            elif speed.get(i)>=5:
                colors1.append('red')
            else:
                colors1.append('k')
    return colors1
def dataframetotensor(df):
    tensor=[]
    for index, row in df.iterrows():
        tensor.append(row)
    return tensor
def tensortoinputoutput(set,index):
    i=torch.zeros(2)
    o=torch.zeros(2)
    boolean=True
    if len(set)>=index+1:
        i=torch.tensor(set[index].values)
        o=torch.tensor(set[index+1].values)
        boolean=False
    return i,o,boolean


supported_road_types = ['primary', 'secondary', 'trunk', 'trunk_link', 'primary_link', 'secondary_link']
supported_road_types_dict = {type: idx for idx, type in enumerate(supported_road_types)}
onehots = {idx:[1 if idx == n else 0 for n in range(len(supported_road_types))] for idx in range(len(supported_road_types))}
from itertools import chain

def type_to_one_hot(type: str):
    return onehots[supported_road_types_dict[type.lower()]]


def type_to_float_idx(type: str):
    return float(supported_road_types_dict[type.lower()])



def calc_node_pos(dataframe: pd.DataFrame):
    new_frame = dataframe.copy()
    col_lat = dataframe.loc[:, ['latStart', 'latEnd']]
    col_lon = dataframe.loc[:, ['lonStart', 'lonEnd']]
    new_frame['mean_lat'] = col_lat.mean(axis=1)
    new_frame['mean_lon'] = col_lon.mean(axis=1)
    return new_frame


def generate_graph_nodes_roads(graph_data_frame: pd.DataFrame, include_all=False, color=False):
    if not include_all:
        search = ['primary', 'secondary', 'trunk']  # ,'tertiary'
        found = [graph_data_frame['type'].str.contains(x) for x in search]
        found = found[0] | found[1] | found[2]
        graph_data_frame = graph_data_frame[found]

    traffic_graph = nx.Graph()
    graph_data_frame = calc_node_pos(graph_data_frame)

    for _, values in graph_data_frame.iterrows():
        attributes = {'max_speed': values['maxSpeed'],
                      'distance': values['distance'],
                      'type': type_to_float_idx(values['type']),
                      'pos': [values['mean_lon'], values['mean_lat']]}
        traffic_graph.add_node(values['edgeId'], **attributes)

    for _, values in graph_data_frame.iterrows():
        start_of_road = values['startNode']
        end_of_road = values['endNode']
        node_id = values['edgeId']
        iterable = [['startNode', 'endNode', 'endNode', 'startNode'],
                    [end_of_road, end_of_road, start_of_road, start_of_road]]
        neighbours = [list(graph_data_frame[graph_data_frame[x] == y]['edgeId'].values) for x, y in zip(*iterable)]
        neighbours = list(set(chain(*neighbours)))
        for neighbour in neighbours:
            traffic_graph.add_edge(node_id, neighbour)
    return traffic_graph
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
    def __init__(self, input_nodes,output_size,hidden_size):
        super(Net, self).__init__()
        self.conv1 = GCNConv(1, hidden_size)
        self.conv2 = GCNConv(hidden_size, 1)

    def forward(self, x,edge_index):
        #edge_index = torch.from_numpy(preprocess_adj(edge_index)).float()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
def train(set):
    for t in range(len(set)):
        input , target , IsEnd= tensortoinputoutput(set,t)
        if not IsEnd:
            input=input.cuda()
            target=target.cuda()
            model.zero_grad()
            input = torch.unsqueeze(input, -1).float().cuda()
            output = model(input.float(), edge_index=sparse_adj_in_coo_format_tensor)
            loss = criterion(output.view(-1),target)
            loss.backward()

            for p in model.parameters():
                p.data.add_(p.grad.data, alpha=-learning_rate)

    return output , loss.item()

if __name__ == '__main__':
    data=loaddata()
    # data.itertuples(index=False,)
    pos={}
    colors=[]
    widths=[]
    for index, row in data.iterrows():
        pos.update( {row.startNode : (row.lonStart,row.latStart)})
        pos.update({row.endNode : (row.lonEnd,row.latEnd)})
        if row.maxSpeed>49:
            colors.append('green')
        elif row.maxSpeed>27:
            colors.append('orange')
        elif row.maxSpeed>=5:
            colors.append('red')
        else:
            colors.append('k')
        if row.type == 'primary' or 'primary_link':
            widths.append(3.0)
        elif row.type == 'secondary' or 'secondary_link':
            widths.append(1.5)
        elif row.type == 'tertiary' or 'tertiary_link':
            widths.append(1.0)
        elif row.type == 'trunk' or 'trunk_link':
            widths.append(2.0)
        else:
            widths.append(0.75)

    # G = nx.from_pandas_edgelist(data, source='startNode', target='endNode',edge_attr=True)
    #
    #
    # # Plot it
    # nx.draw(G,pos=pos,node_size=5,width=widths,edge_color=colors, with_labels=False)
    # plt.show(

    avg_speed = pd.read_csv('E:\PycharmProjects\graph_convolution\Data\Xian_city/raw/avg_speed.csv', sep=';', index_col=0)
    avg_speed = convert_index_to_datetime(avg_speed)
    c=speedtocolor("2016-10-01T00:50:00")
    # print(c)
    G = generate_graph_nodes_roads(data)
    # G = nx.from_pandas_edgelist(data, source='startNode', target='endNode', edge_attr=True, create_using=nx.DiGraph)

    # Plot it
    # nx.draw(G, pos=pos, node_size=5, width=widths, edge_color=c, with_labels=False)
    # plt.show()
    # adj=nx.adjacency_matrix(G)
    adj = nx.to_pandas_adjacency(G)
    road_order = [str(x) for x in list(G.nodes)]

    avg_speed = avg_speed[road_order]
    # eventueel hiervoor al self loops introduceren


    sparse_adj = nx.to_scipy_sparse_matrix(G).tocoo()
    sparse_adj_in_coo_format = np.stack([sparse_adj.row, sparse_adj.col])
    sparse_adj_in_coo_format_tensor = torch.tensor(sparse_adj_in_coo_format, dtype=torch.long).cuda()
    # adj=adj.tocoo()
    # adj = torch.sparse.LongTensor(torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),
    #                               torch.LongTensor(adj.data.astype(np.int32)))
    # adj_tensor = torch.tensor(adj)
    tensors=[]
    for index, row in avg_speed.iterrows():
        tensors.append(row.values)
        inputsize=len(row)
    trainingset=avg_speed.loc["2016-10-01":"2016-11-01"]
    trainingtensor=dataframetotensor(trainingset)

    learning_rate=0.01
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(inputsize,inputsize,inputsize).to(device)
    model=model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    criterion = nn.MSELoss()

    training_losses=[]
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = train(trainingtensor)
        # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        # loss.backward()
        optimizer.step()




