import networkx as nx
import numpy as np
from GCN import loaddata, load_avgspeed, loadmodel
from GCN import Net
from torch.utils.tensorboard import SummaryWriter
import torch,lib
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd

if __name__ == '__main__':

    #Logger
    writer = SummaryWriter()

    #Load data
    data=loaddata()
    avg_speed=load_avgspeed()

    # Generate graph
    G= lib.generate_graph_nodes_roads(data)
    pos, widths, colors = lib.get_graph_params(G)
    # Update colors
    c = lib.speedtocolor('2016-10-01 00:50:00', avg_speed,data)

    # Plot it
    # nx.draw(G, pos=pos, node_size=5, width=widths, edge_color=c, with_labels=False)
    # plt.show()

    # Sort speed data to road order
    road_order = [str(x) for x in list(G.nodes)]
    avg_speed = avg_speed[road_order]

    # Get Adjacency Matrix
    adjacency_matrix = lib.get_adjacency_from_graph(G)

    # Define training and test set
    batch_size=1
    trainingset = avg_speed.loc["2016-10-01":"2016-11-01"]
    train_data, train_graph, scaler = lib.df_to_data(trainingset, adjacency_matrix, batch_size)
    testset = avg_speed.loc["2016-11-07":"2016-11-14"]
    test_data, test_graph = lib.df_to_data_val(testset, scaler, adjacency_matrix, batch_size)
    valset = avg_speed.loc["2016-11-18":"2016-11-19"]
    val_data, val_graph = lib.df_to_data_val(valset, scaler, adjacency_matrix, batch_size)
    # val_data_numpy=next(iter(val_data))[0].numpy()

    #Define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(train_graph[0]['x'].shape[0]).to(device)
    model = loadmodel(model)
    model=model.double()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    criterion = nn.MSELoss().cuda()

    #Evaluate model
    model.eval()
    torch.set_grad_enabled(False)
    entry = next(iter(test_data))
    output = model(entry)
    val_loss = criterion(output,entry.y)
    target = entry.y
    lib.tensor_to_plot(output,target,scaler,G,pos,widths)


    MSE=nn.MSELoss()
    # NLL=nn.NLLLoss().cuda()
    MSE_losses=[]
    # NLL_losses=[]
    MAE_losses=[]

    for entry in test_data:
        output = model(entry)
        val_loss = MSE(output, entry.y)
        MSE_losses.append(val_loss)
        # val_loss = NLL(output,entry.y.long())
        # NLL_losses.append(val_loss)
        target = entry.y
        output = torch.transpose(output, 0, 1)
        target = torch.transpose(target, 0, 1)
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        unnormalized_output = scaler.inverse_transform(output)
        unnormalized_target = scaler.inverse_transform(target)
        mae = np.mean(np.absolute(unnormalized_output-unnormalized_target))
        MAE_losses.append(mae)

    #Predict 1 hour in future
    predicted, target, losses = lib.predict_1hour(test_data,model)
    lib.tensor_to_plot(predicted,target,scaler,G,pos,widths)
    print("Prediction loss: "+ str(losses))
    #Save losses
    for i in range(len(MSE_losses)):
        writer.add_scalar('Mean Square Error loss:',MSE_losses[i],i)
    # for i in range(len(NLL_losses)):
    #     writer.add_scalar('Near Logarithmic Likelihood loss:',NLL_losses[i],i)
    for i in range(len(MAE_losses)):
        writer.add_scalar('Mean Absolute Error loss:',MAE_losses[i],i)
