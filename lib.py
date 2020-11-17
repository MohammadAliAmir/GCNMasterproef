import os

import networkx as nx
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from sklearn import preprocessing
import torch
from torch_geometric.data import Data, Batch, DataLoader
import numpy as np
import matplotlib.pyplot as plt

supported_road_types = ['primary', 'secondary', 'trunk', 'trunk_link', 'primary_link', 'secondary_link']
supported_road_types_dict = {type: idx for idx, type in enumerate(supported_road_types)}
onehots = {idx:[1 if idx == n else 0 for n in range(len(supported_road_types))] for idx in range(len(supported_road_types))}
from itertools import chain

def convert_index_to_datetime(feature):
    indices = pd.to_datetime(feature.index)
    feature.index = indices
    return feature


def df_to_data(df, edge_matrix, batch_size):
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    x_inv=min_max_scaler.inverse_transform(x_scaled)
    df = pd.DataFrame(x_scaled)
    data_graphs = []
    # timedelta = datetime.timedelta(minutes=5)
    for i in range(len(df) - 1):
        # j=i.to_pydatetime()
        # next=j+timedelta
        x = torch.tensor([df.iloc[i]], dtype=torch.double).cuda()
        x = torch.transpose(x, 0, 1)
        # x = x.permute(df.shape[1], 1)  # nodes, features
        y = torch.tensor([df.iloc[i + 1]], dtype=torch.double).cuda()
        y = torch.transpose(y, 0, 1)
        # y = y.permute(df.shape[1], 1)  # nodes, features
        data_entry = Data(x=x, y=y, edge_index=edge_matrix)
        data_graphs.append(data_entry)
    loader = DataLoader(data_graphs, batch_size=batch_size)
    return loader, data_graphs, min_max_scaler


def df_to_data_val(df, scaler, edge_matrix, batch_size):
    x = df.values  # returns a numpy array
    x_scaled = scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    data_graphs = []
    # timedelta = datetime.timedelta(minutes=5)
    for i in range(len(df) - 1):
        # j=i.to_pydatetime()
        # next=j+timedelta
        x = torch.tensor([df.iloc[i]], dtype=torch.double).cuda()
        x = torch.transpose(x, 0, 1)
        # x = x.permute(df.shape[1], 1)  # nodes, features
        y = torch.tensor([df.iloc[i + 1]], dtype=torch.double).cuda()
        y = torch.transpose(y, 0, 1)
        # y = y.permute(df.shape[1], 1)  # nodes, features
        data_entry = Data(x=x, y=y, edge_index=edge_matrix)
        data_graphs.append(data_entry)
    loader = DataLoader(data_graphs, batch_size=batch_size)
    return loader, data_graphs


# def preprocess_adj(A):
#     '''
#     Pre-process adjacency matrix
#     :param A: adjacency matrix
#     :return:
#     '''
#     I = np.eye(A.shape[0])
#     A_hat = A + I  # add self-loops
#     D_hat_diag = np.sum(A_hat, axis=1)
#     D_hat_diag_inv_sqrt = np.power(D_hat_diag, -0.5)
#     D_hat_diag_inv_sqrt[np.isinf(D_hat_diag_inv_sqrt)] = 0.
#     D_hat_inv_sqrt = np.diag(D_hat_diag_inv_sqrt)
#     return np.dot(np.dot(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)


def speedtocolor(date, speed_df, edgelist_df):
    # speed=time_dict[date]
    speed = speed_df.loc['2016-10-01 00:00:00']
    keys = speed.keys()
    colors1 = []
    df = edgelist_df['edgeId'].tolist()
    for i in keys:
        if int(i) in df:
            j = speed.get(i)
            if j > 49:
                colors1.append('green')
            elif speed.get(i) > 27:
                colors1.append('orange')
            elif speed.get(i) >= 5:
                colors1.append('red')
            else:
                colors1.append('k')
    return colors1
def update_colors(df):
    colors=[]
    for i in df[0]:
        # t=df.loc[i]
        if i > 49:
            colors.append('green')
        elif i > 27:
            colors.append('orange')
        elif i >= 5:
            colors.append('red')
        else:
            colors.append('k')
    return colors

# def speed_df_to_color(df, edgelist_df):
#     # speed=time_dict[date]
#     speed = df
#     keys = speed.keys()
#     colors1 = []
#     df = edgelist_df['edgeId'].tolist()
#     for i in keys:
#         if int(i) in df:
#             j = speed.get(i)
#             if j > 49:
#                 colors1.append('green')
#             elif speed.get(i) > 27:
#                 colors1.append('orange')
#             elif speed.get(i) >= 5:
#                 colors1.append('red')
#             else:
#                 colors1.append('k')
#     return colors1


# def dataframetotensor(df):
#     tensor = []
#     for index, row in df.iterrows():
#         t = torch.tensor((row.values))
#         t = torch.unsqueeze(t, -1)
#         tensor.append(t)
#     return tensor


# def tensortoinputoutput(set, index):
#     i = torch.zeros(2)
#     o = torch.zeros(2)
#     boolean = True
#     if len(set) > index + 1:
#         i = set[index]
#         o = set[index + 1]
#         boolean = False
#     return i, o, boolean

def get_graph_params(graph):
    pos = nx.get_node_attributes(graph,'pos')
    speed = nx.get_node_attributes(graph,'max_speed')
    road_types = nx.get_node_attributes(graph,'type')
    widths=[]
    colors=[]
    for type in road_types:
        if type == 'primary' or 'primary_link':
            widths.append(3.0)
        elif type == 'secondary' or 'secondary_link':
            widths.append(1.5)
        elif type == 'tertiary' or 'tertiary_link':
            widths.append(1.0)
        elif type == 'trunk' or 'trunk_link':
            widths.append(2.0)
        else:
            widths.append(0.75)
    for v in speed:
        if v>49:
            colors.append('green')
        elif v>27:
            colors.append('orange')
        elif v>=5:
            colors.append('red')
        else:
            colors.append('k')
    return pos,widths,colors

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

def calc_node_pos(dataframe: pd.DataFrame):
    new_frame = dataframe.copy()
    col_lat = dataframe.loc[:, ['latStart', 'latEnd']]
    col_lon = dataframe.loc[:, ['lonStart', 'lonEnd']]
    new_frame['mean_lat'] = col_lat.mean(axis=1)
    new_frame['mean_lon'] = col_lon.mean(axis=1)
    return new_frame

def type_to_one_hot(type: str):
    return onehots[supported_road_types_dict[type.lower()]]


def type_to_float_idx(type: str):
    return float(supported_road_types_dict[type.lower()])

def get_adjacency_from_graph(graph):
    sparse_adj = nx.to_scipy_sparse_matrix(graph).tocoo()
    sparse_adj_in_coo_format = np.stack([sparse_adj.row, sparse_adj.col])
    sparse_adj_in_coo_format_tensor = torch.tensor(sparse_adj_in_coo_format, dtype=torch.long).cuda()
    return sparse_adj_in_coo_format_tensor

def predict_1hour(data, model):
    loss_function = nn.MSELoss().cuda()
    losses=[]
    batch = next(iter(data))
    output=batch.x
    for i in range(12):

        batch.x=output
        output = model(batch)
        loss = loss_function(output,batch.y)
        losses.append(loss)
        target=batch.y
        batch = next(iter(data))
    return output, target, sum(losses)/len(losses)

def tensor_to_plot(output, target, scaler, graph, pos, widths):
    output = torch.transpose(output, 0, 1)
    target = torch.transpose(target, 0, 1)
    output = output.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    unnormalized_output = scaler.inverse_transform(output)
    unnormalized_target = scaler.inverse_transform(target)
    colors_output = update_colors(unnormalized_output)
    colors_target = update_colors(unnormalized_target)

    fig = plt.figure()
    fig.suptitle('Output graph')
    nx.draw(graph, pos=pos, node_size=5, width=widths, edge_color=colors_output, with_labels=False)
    plt.show()

    fig = plt.figure()
    fig.suptitle('Target graph')
    nx.draw(graph, pos=pos, node_size=5, width=widths, edge_color=colors_target, with_labels=False)
    plt.show()


