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
from datetime import datetime
import Model
import stgcn

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



def savemodel(model, epoch,model_no,loss,optimizer, criterion):
    torch.save({"model":model.state_dict(),"loss": loss, "optimizer": optimizer.state_dict(), "epoch":epoch, "criterion" : criterion }, './Models/GCNModel'+str(model_no)+'.ckpt'+datetime.today().strftime("%d-%m-%Y %H-%M")+" "+str(epoch)+comment)

def savelstm(model,epoch,loss,optimizer,criterion):
    torch.save({"model": model.state_dict(), "loss": loss, "optimizer": optimizer.state_dict(), "epoch": epoch,
                "criterion": criterion},
               './Models/LSTM/lstmModel.ckpt ' + datetime.today().strftime("%d-%m-%Y %H-%M") + " " + str(
                   epoch) + comment)


def loadmodel(path='./Models/GCNModel.ckpt'):
    checkpoint = torch.load(path)
    split=path.split("Models/")
    split = split[1].split(".ckpt")
    split1 = split[1].split(" ")
    if split[0].__contains__("2"):
        model = Model.Net2(int(split1[13]))
        # model = Model.Net2(2121)
    elif split[0].__contains__("4"):
        model = Model.Net4(int(split1[13]))
    else :
        model = Model.Net(int(split1[13]))
    learning_rate= float(split1[5])
    # learning_rate = 1e-4
    batch_size=int(split1[7])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]
    losses = checkpoint["loss"]
    criterion = checkpoint["criterion"]
    # losses =None
    # epoch=None
    # criterion=None


    return model, optimizer, epoch, losses, criterion ,batch_size


def train(loader):
    model.train()
    print("In training")

    epoch_losses=[]
    for data_graph in loader:
        # print(" ")
        model.zero_grad()
        output = model(data_graph)
        loss = criterion(output,data_graph.y).double()
        loss.backward()
        print("Loss: " + str(loss.item()))
        epoch_losses.append(loss.item())

        for p in model.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)

    return  sum(epoch_losses)/len(epoch_losses)

def evaluate(set):

    losses=[]
    losses2=[]
    for entry in set:
        output=model(entry)
        target=entry.y
        val_loss = criterion(output, target).double()

        # print("MAE Loss: " + str(val_loss.item()))
        val_loss2 = criterion2(output, target).double()

        # print("MSE Loss: " + str(val_loss2.item()))
        losses.append(val_loss)
        losses2.append(val_loss2)
        if loss_function =="MAE":
            print("MAE Loss: " + str(val_loss.item()))
            print("MSE Loss: " + str(val_loss2.item()))
        else:
            print("MSE Loss: " + str(val_loss.item()))
            print("MAE Loss: " + str(val_loss2.item()))

    return sum(losses)/len(losses), sum(losses2)/len(losses2)



if __name__ == '__main__':

    model_no = 2
    model_hidden_size_1 = [ 30]
    withLstm = True
    debugging = False
    seq_len = 10
    lstm_hidden_size = 32
    lstm_layers = 1
    learning_rate = 1e-4
    batch_size = 250
    epochs = 50
    used_scaler = "MinMax"
    loss_function = "MAE"
    path = "./training/" + datetime.today().strftime("%d-%m-%Y %H-%M")

    # Load data
    data = loaddata()
    avg_speed = load_avgspeed()

    # Generate graph
    G = lib.generate_graph_nodes_roads(data)
    # Initial graph drawing parameters
    pos, widths, colors = lib.get_graph_params(G)

    # Sort speed data to road order
    road_order = [str(x) for x in list(G.nodes)]
    avg_speed = avg_speed[road_order]
    # eventueel hiervoor al self loops introduceren

    # Scale data
    # avg_speed, scaler = lib.scale_data(avg_speed)
    # Get Adjacency Matrix
    adjacency_matrix = lib.get_adjacency_from_graph(G)

    # Define training and test set
    trainingset = avg_speed.loc["2016-10-01":"2016-11-01"]
    train_data, train_graph, scaler = lib.df_to_data(trainingset, adjacency_matrix, batch_size, used_scaler)
    testset = avg_speed.loc["2016-11-07":"2016-11-14"]
    test_data, test_graph = lib.df_to_data_val(testset, scaler, adjacency_matrix, batch_size)
    valset = avg_speed.loc["2016-11-18":"2016-11-19"]
    val_data, val_graph = lib.df_to_data_val(valset, scaler, adjacency_matrix, batch_size)

    for model_hidden in model_hidden_size_1:
        if withLstm:
            comment = " epochs lr " + str(learning_rate) + " batch " + str(
                batch_size) + " lossfunc " + loss_function + " GCN hidden size " + str(
                model_hidden) + " scaler " + used_scaler + " seq_len " + str(seq_len) + " lstm_size " + str(
                lstm_hidden_size) + " lstm_layers " + str(lstm_layers)
        else:
            comment = " epochs lr " + str(learning_rate) + " batch " + str(
                batch_size) + " lossfunc " + loss_function + " GCN hidden size " + str(
                model_hidden) + " scaler " + used_scaler

        load_previous_model = False
        if load_previous_model:
            model_path = "./Models/GCNModel.ckpt01-12-2020 03-43 200 epochs lr 0.0001 batch 1 lossfunc MSE"
        # Logger
        if debugging:
            writer = SummaryWriter()
        else:
            writer = SummaryWriter(log_dir=path, comment=str(epochs) + comment)

        #
        # train_data_lstm, train_graph_lstm = lib.generate_dataset_LSTM(trainingset,seq_len,adjacency_matrix,batch_size)
        # test_data_lstm, test_graph_lstm = lib.generate_dataset_LSTM(testset, seq_len, adjacency_matrix, batch_size)
        # val_data_lstm, val_graph_lstm = lib.generate_dataset_LSTM(valset, seq_len, adjacency_matrix, batch_size)

        # Initialize model
        inputsize = avg_speed.shape[1]
        training_losses = []
        val_losses = []
        val2_losses = []
        test_losses = []
        test2_losses = []
        accuracy_check = 5
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model = Model.GCRN(train_graph[0]['x'].shape[0],1,batch_size=batch_size).to(device)
        if withLstm:
            Lstm = Model.RNN(lstm_hidden_size, lstm_layers=lstm_layers, batch_size=batch_size,seq_len=seq_len,input_size=2121).to(device)
        # model1=stgcn.STGCN(train_graph[0]['x'].shape[0],1,10,10)
        if model_no == 1:
            model = Model.Net(model_hidden).to(device)
            # model = Model.Net3(model_hidden, seq_len).to(device)
        elif model_no == 2:
            model = Model.Net2(model_hidden).to(device)
        else:
            model = Model.Net4(model_hidden).to(device)
            # model = Model.Net3(model_hidden, batch_size).to(device)

        model = model.double()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
        if loss_function == "MAE":
            criterion = nn.L1Loss().cuda()
            criterion2 = nn.MSELoss().cuda()
        else:
            criterion = nn.MSELoss().cuda()
            criterion2 = nn.L1Loss().cuda()
        # Load model
        if load_previous_model:
            model, optimizer, epoch, training_losses, criterion, batch_size = loadmodel(model_path)

        # Start training
        print("Start training")
        model.train()
        for epoch in range(epochs):
            print("Epoch: " + str(epoch))
            optimizer.zero_grad()
            if withLstm:
                epoch_loss = Model.train_GCRN(model=model, Lstm=Lstm, loader=train_data, criterion=criterion,
                                              learning_rate=learning_rate, seq_len=seq_len, batch_size=batch_size)
            else:
                epoch_loss = train(train_data)
            # epoch_loss = Model.train2(model1,train_data,criterion,learning_rate,seq_len)
            training_losses.append(epoch_loss)
            optimizer.step()

            # if (epoch % accuracy_check ==0):
            if not withLstm:
                print("Evaluate")
                with torch.no_grad():
                    model.eval()
                    if withLstm:
                        criterion_loss, criterion2_loss = Model.evaluate_GCRN(val_data, model, Lstm, criterion, criterion2,
                                                                              loss_function, seq_len)
                        test_criterion_loss, test_criterion2_loss = Model.evaluate_GCRN(test_data, model, Lstm, criterion,
                                                                                        criterion2, loss_function, seq_len)
                    else:
                        criterion_loss, criterion2_loss = evaluate(val_data)
                        test_criterion_loss, test_criterion2_loss = evaluate(test_data)
                    val_losses.append(criterion_loss)
                    val2_losses.append(criterion2_loss)
                    test_losses.append(test_criterion_loss)
                    test2_losses.append(test_criterion2_loss)
                # if epoch % 5 == 0:
                #     # torch.save(model.state_dict(), 'checkpoint/epoch_' + str(epoch) + '.tar')
                #     savemodel(model,epoch,model_no,training_losses,optimizer,criterion)

        # Save model after training
        savemodel(model, epochs, model_no, training_losses, optimizer, criterion)
        if withLstm:
            savelstm(model,epochs,training_losses,optimizer,criterion)
        # Save losses
        print("training losses length: " + str(len(training_losses)))
        print("valdiation losses length: " + str(len(val_losses)))
        print("test losses length: " + str(len(test_losses)))
        for i in range(len(training_losses)):
            writer.add_scalars('Loss', {'train': training_losses[i]}, i)
        if loss_function == "MAE":
            for i in range(len(val_losses)):
                writer.add_scalars('Validation',
                                   {'Mean Absolute error': val_losses[i], 'Mean Square error': val2_losses[i],
                                    'Test MAE': test_losses[i], 'Test MSE': test2_losses[i]}, i)
        else:
            for i in range(len(val_losses)):
                writer.add_scalars('Validation',
                                   {'Mean Square error': val_losses[i], 'Mean Absolute error': val2_losses[i],
                                    'Test MSE': test_losses[i], 'Test MAE': test2_losses[i]}, i)

        for i in range(len(training_losses)):
            writer.add_scalars('Loss/training',{'train' :training_losses[i]},i)
        for i in range(len(val_losses)):
            writer.add_scalars('Loss/training',{'validation' : val_losses[i]},i)
        for i in range(len(test_losses)):
            writer.add_scalars('Loss/training',{'test' : test_losses[i]},i)
        writer.close()