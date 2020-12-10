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
import math
import matplotlib
from datetime import datetime

supported_road_types = ['primary', 'secondary', 'trunk', 'trunk_link', 'primary_link', 'secondary_link']
supported_road_types_dict = {type: idx for idx, type in enumerate(supported_road_types)}
onehots = {idx:[1 if idx == n else 0 for n in range(len(supported_road_types))] for idx in range(len(supported_road_types))}
from itertools import chain

def convert_index_to_datetime(feature):
    indices = pd.to_datetime(feature.index)
    feature.index = indices
    return feature


def df_to_data(df, edge_matrix, batch_size,used_scaler="MinMax"):
    x = df.values  # returns a numpy array
    if used_scaler =="MinMax":
        min_max_scaler = preprocessing.MinMaxScaler()
    else:
        min_max_scaler = preprocessing.StandardScaler()

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
    loader = DataLoader(data_graphs, batch_size=batch_size,pin_memory=True)
    # loader = DataLoader(data_graphs, batch_size=1)
    return loader, data_graphs , min_max_scaler


def df_to_data_val(df,scaler, edge_matrix, batch_size):
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
    # loader = DataLoader(data_graphs, batch_size=1)
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

def predict_1hour(data, model, scaler, df,edge):
    loss_function = nn.MSELoss().cuda()
    losses=[]
    batch = next(iter(data))
    output=batch.x
    outputs=[]
    targets=[]
    for i in range(12):

        batch.x=output
        output = model(batch)
        unnormalized_output = torch.tensor(np.transpose(unnormalize_tensor(output, scaler), [0, 1]))
        unnormalized_target = torch.tensor(np.transpose(unnormalize_tensor(batch.y, scaler), [0, 1]))
        loss = loss_function(unnormalized_output,unnormalized_target)
        losses.append(loss)
        outputs.append(unnormalized_output)
        targets.append(unnormalized_target)
        batch = next(iter(data))

    #Plot
    predictions_speed=[]
    targets_speed=[]
    for i in outputs:
        predictions_speed.append(i[0][edge])
    for i in targets:
        targets_speed.append(i[0][edge])

    datetimes= df.axes[0][:12]
    new_datetimes=[]

    for i in datetimes:
        date_obj=datetime.strptime(i,"%Y-%m-%d %H:%M:%S")
        new_datetimes.append(date_obj.strftime("%d-%m %H:%M"))
    fig = plt.figure()
    fig.suptitle("Speed over time")
    plt.ylabel("Speed")
    plt.xlabel("Time")
    plt.ylim((0,70))
    plt.plot(new_datetimes,predictions_speed,label="Prediction")
    fig.autofmt_xdate()
    plt.plot(new_datetimes,targets_speed,label="Target")
    plt.legend()

    return outputs, targets, losses
def predict(data, model, scaler,hours,df,edge,name):
    loss_function = nn.MSELoss().cuda()
    losses = []
    batch = next(iter(data))
    output = batch.x
    outputs = []
    targets = []
    rounds=int(hours*12)
    t=data.dataset[1]
    # zoek voor moving window
    count=0
    for batch in data.dataset:
        batch.x = output
        output = model(batch)
        unnormalized_output = torch.tensor(np.transpose(unnormalize_tensor(output, scaler), [0, 1]))
        unnormalized_target = torch.tensor(np.transpose(unnormalize_tensor(batch.y, scaler), [0, 1]))
        loss = loss_function(unnormalized_output, unnormalized_target)
        losses.append(loss)
        outputs.append(unnormalized_output)
        targets.append(unnormalized_target)
        count+=1
        if count>=rounds:
            break
    # for i in range(rounds):
    #     batch.x = output
    #     output = model(batch)
    #     unnormalized_output = torch.tensor(np.transpose(unnormalize_tensor(output, scaler), [0, 1]))
    #     unnormalized_target = torch.tensor(np.transpose(unnormalize_tensor(batch.y, scaler), [0, 1]))
    #     loss = loss_function(unnormalized_output, unnormalized_target)
    #     losses.append(loss)
    #     outputs.append(unnormalized_output)
    #     targets.append(unnormalized_target)
    #     batch = next(iter(data))

    # Plot
    predictions_speed = []
    targets_speed = []
    for i in outputs:
        predictions_speed.append(i[0][edge])
    for i in targets:
        targets_speed.append(i[0][edge])

    datetimes = df.axes[0][:rounds]
    new_datetimes = []
    import matplotlib.dates as mdates
    for i in datetimes:
        date_obj = datetime.strptime(i, "%Y-%m-%d %H:%M:%S")
        new_datetimes.append(date_obj.strftime("%d-%m %H:%M"))
    if hours >=24:
        loc=mdates.DayLocator()
    elif hours>1:
        loc = mdates.HourLocator()
    else:
        loc=mdates.MinuteLocator(30)


    # fig, ax = plt.subplots()
    # ax.plot(new_datetimes, predictions_speed)
    # ax.xaxis.set_major_locator(mdates.HourLocator(byhour=1))
    # # ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    # # ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=10))
    # fig.autofmt_xdate()

    fig=plt.figure()
    fig.suptitle("Speed over time for "+str(hours)+"h prediction road 160")
    plt.ylabel("Speed")
    plt.xlabel("Time")
    plt.ylim((0, 70))

    # ax.xaxis.set_minor_locator(mdates.MinuteLocator())
    # xaxis=plt.axes
    # print(plt.axes)
    # plt.axes[0].xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    plt.plot(new_datetimes, predictions_speed, label="Prediction")
    fig.autofmt_xdate()
    plt.plot(new_datetimes, targets_speed, label="Target")
    # ax.xaxis.set_major_locator(mdates.HourLocator(byhour=1))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    # ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=10))
    # plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    # [i.xaxis.set_major_locator(mdates.HourLocator(1)) for i in fig.axes]
    # [i.xaxis.set_minor_locator(mdates.MinuteLocator(30)) for i in fig.axes]

    plt.legend()
    # plt.tight_layout()


    plt.savefig('./plots/speed '+datetime.today().strftime("%d-%m-%Y %H-%M")+name+'.png',format='png',bbox_inches='tight')
    plt.close(fig)
    return outputs, targets, losses

def predict2(data, model, scaler,rounds,df,roads,name):
    road_2748 = roads.index('2748')
    road_160 = roads.index('160')
    # road_161 = roads.index('161')
    # road_753 = roads.index('753')
    # road_92053 = roads.index('92053')
    # road_274 = roads.index('274')
    # road_752 = roads.index('752')


    loss_function = nn.MSELoss().cuda()
    losses = []
    # batch = next(iter(data))
    # output = batch.x
    outputs = []
    targets = []
    # rounds=int(hours*12)
    # rounds=1
    # rolling=0
    # zoek voor moving window
    count=0
    for rolling in range(len(data.dataset)-rounds):
        output=data.dataset[rolling].x
        batch=data.dataset[rolling]
        for i in range(rolling,len(data.dataset)-rounds):
            batch.x = output
            output = model(batch)
            unnormalized_output = torch.tensor(np.transpose(unnormalize_tensor(output, scaler), [0, 1]))
            unnormalized_target = torch.tensor(np.transpose(unnormalize_tensor(data.dataset[i].y, scaler), [0, 1]))
            loss = loss_function(unnormalized_output, unnormalized_target)
            losses.append(loss)
            count+=1
            if count>=rounds:
                outputs.append(unnormalized_output)
                targets.append(unnormalized_target)
                # rolling+=1
                count=0
                break

    # Plot
    predictions_speed_160 = []
    targets_speed_160 = []
    predictions_speed_2748 = []
    targets_speed_2748 = []
    for i in outputs:
        predictions_speed_160.append(i[0][road_160])
        predictions_speed_2748.append(i[0][road_2748])
    for i in targets:
        targets_speed_160.append(i[0][road_160])
        targets_speed_2748.append(i[0][road_2748])

    # if rounds==1:
    #     datetimes = df.axes[0][:rolling+1]
    # else:
    datetimes= df.axes[0][:len(outputs)]
    new_datetimes = []
    # import matplotlib.dates as mdates
    for i in datetimes:
        date_obj = datetime.strptime(i, "%Y-%m-%d %H:%M:%S")
        new_datetimes.append(date_obj.strftime("%d-%m %H:%M"))
    # if hours >=24:
    #     loc=mdates.DayLocator()
    # elif hours>1:
    #     loc = mdates.HourLocator()
    # else:
    #     loc=mdates.MinuteLocator(30)


    # fig, ax = plt.subplots()
    # ax.plot(new_datetimes, predictions_speed)
    # ax.xaxis.set_major_locator(mdates.HourLocator(byhour=1))
    # # ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    # # ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=10))
    # fig.autofmt_xdate()

    duration = rounds * 5
    if duration>=60:
        label_duration=str(int(duration/60))+"h "
    else:
        label_duration=str(int(duration))+ "min "

    fig=plt.figure()
    fig.suptitle("Speed over time for "+label_duration+"prediction road 160")
    plt.ylabel("Speed")
    plt.xlabel("Time")
    plt.ylim((0, 70))

    # ax.xaxis.set_minor_locator(mdates.MinuteLocator())
    # xaxis=plt.axes
    # print(plt.axes)
    # plt.axes[0].xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    plt.plot(new_datetimes, targets_speed_160, label="Target")


    plt.plot(new_datetimes, predictions_speed_160, label="Prediction "+label_duration)
    fig.autofmt_xdate()

    # ax.xaxis.set_major_locator(mdates.HourLocator(byhour=1))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    # ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=10))
    # plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    # [i.xaxis.set_major_locator(mdates.HourLocator(1)) for i in fig.axes]
    # [i.xaxis.set_minor_locator(mdates.MinuteLocator(30)) for i in fig.axes]

    plt.legend()
    # plt.tight_layout()


    plt.savefig('./plots/160/speed '+datetime.today().strftime("%d-%m-%Y %H-%M")+name+'.png',format='png',bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    fig.suptitle("Speed over time for " + label_duration + "prediction road 2748")
    plt.ylabel("Speed")
    plt.xlabel("Time")
    plt.ylim((0, 70))

    plt.plot(new_datetimes, targets_speed_2748, label="Target")
    plt.plot(new_datetimes, predictions_speed_2748, label="Prediction " +label_duration)
    fig.autofmt_xdate()
    plt.legend()


    plt.savefig('./plots/2748/speed ' + datetime.today().strftime("%d-%m-%Y %H-%M") + name + '.png', format='png',
                bbox_inches='tight')
    plt.close(fig)

    return outputs, targets, losses

def tensor_to_plot(output, target, scaler, graph, pos, widths,name,Unnormalize=True ):

    if Unnormalize:
        unnormalized_output = unnormalize_tensor(output,scaler)
        unnormalized_target = unnormalize_tensor(target,scaler)
    else:
        unnormalized_output = output
        unnormalized_target = target

    #Update graph with new speeds
    # i=next(iter(graph.nodes()))
    # n = graph.nodes[i]
    # count=0
    # for i in graph.nodes():
    #     graph.nodes[i]['max_speed'] = unnormalized_output[0][count]
    #     count+=1

    #Resize
    output_u_s=np.squeeze(unnormalized_output)
    target_u_s=np.squeeze(unnormalized_target)
    g_min=0.0
    g_max=70.0

    #Output Tensor to RGBA
    norm=matplotlib.colors.Normalize(vmin=g_min,vmax=g_max)
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
    colorvalues = []
    for i in range(len(output_u_s)):
        colorVal = sm.to_rgba(output_u_s[i])
        colorvalues.append(colorVal)
    #Plot output
    fig = plt.figure()
    fig.suptitle('Prediction')
    nx.draw(graph, pos=pos, node_size=5, width=widths, edge_color=colorvalues, edge_cmap="RdYlGn" , with_labels=False)
    fig.colorbar(sm)
    # plt.tight_layout()
    plt.pause(0.05)
    plt.show(block=False)
    now_str=datetime.today().strftime("%d-%m-%Y %H-%M")
    plt.savefig('./graphs/Output'+ now_str+name+'.png',format='png',bbox_inches='tight')
    plt.close(fig)

    #Target To RGBA
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
    colorvalues = []
    for i in range(len(target_u_s)):
        colorVal = sm.to_rgba(target_u_s[i])
        colorvalues.append(colorVal)

    #Plot Target
    fig = plt.figure()
    fig.suptitle('Actual')
    nx.draw(graph, pos=pos, node_size=5, width=widths, edge_color=colorvalues, edge_cmap="RdYlGn", with_labels=False)
    fig.colorbar(sm)
    # plt.tight_layout()
    plt.pause(0.05)
    plt.show(block=False)
    plt.savefig('./graphs/Target' + now_str+name+'.png',format='png',bbox_inches='tight')
    plt.close(fig)

def unnormalize_tensor(tensor, scaler):
    tensor=torch.transpose(tensor,0,1)
    tensor=tensor.detach().cpu().numpy()
    unnormalized = scaler.inverse_transform(tensor)
    return unnormalized

def tensor_to_avg_speed(tensor, scaler):
    unnormalized_tensor = unnormalize_tensor(tensor,scaler)
    average_speed = sum(unnormalized_tensor)/len(unnormalized_tensor)
    return average_speed

def get_avg_speed_from_edge(df,edge):
    return df[edge].mean()
def unnormalize_loss(loss, scaler):
    return scaler.inverse_transform(loss.cpu())
def scale_data(df):
    x = df.values  # returns a numpy array
    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df, scaler

def generate_dataset_LSTM(df,seq_len,edge_matrix,batch_size):
    data_graphs = []
    x_list=[]
    y_list=[]
    count=0
    for i in range(len(df) - 1):

        x = torch.tensor([df.iloc[i]], dtype=torch.double).cuda()
        x = torch.transpose(x, 0, 1)
        # x = x.permute(df.shape[1], 1)  # nodes, features
        y = torch.tensor([df.iloc[i + 1]], dtype=torch.double).cuda()
        y = torch.transpose(y, 0, 1)
        # y = y.permute(df.shape[1], 1)  # nodes, features
        x_list.append(x)
        y_list.append(y)
        count+=1
        if count>=seq_len:
            data_entry = Data(x=x_list, y=y_list, edge_index=edge_matrix)
            data_graphs.append(data_entry)
            x_list=[]
            y_list=[]
            count=0
    loader = DataLoader(data_graphs, batch_size=batch_size)
    return loader, data_graphs  # , min_max_scaler