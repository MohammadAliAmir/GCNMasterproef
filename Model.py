import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h

class RNN(torch.nn.Module):
    def __init__(self, hidden_size,lstm_layers,batch_size,seq_len,input_size):
        super(RNN, self).__init__()
        self.hidden_size=hidden_size
        self.lstm_layers = lstm_layers
        self.batch_size=batch_size
        self.input_size=input_size
        self.lstm = torch.nn.LSTM(input_size,hidden_size,num_layers=lstm_layers, batch_first=True)
        self.linear=torch.nn.Linear(hidden_size, input_size)
        self.seq_len=seq_len


    def forward(self, input, hidden ):

        out, hidden = self.lstm(input, hidden)
        out= self.linear(out)

        return out,hidden

    def init_hidden(self):
        h0= torch.zeros(1,1,self.hidden_size).cuda()
        c0= torch.zeros(1,1,self.hidden_size).cuda()
        # hidden=[t for t in (h0,c0)]
        return [t for t in (h0, c0)]

class Net(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        self.conv1 = GCNConv(1, hidden_size)
        self.conv2 = GCNConv(hidden_size, 1)

    def forward(self, sample):
        x, edge_index = sample.x.double(), sample.edge_index
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x #  F.log_softmax(x, dim=0)

class Net2(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Net2, self).__init__()
        self.conv1 = GCNConv(1, hidden_size)
        self.conv2 = GCNConv(hidden_size,hidden_size)
        self.conv3 = GCNConv(hidden_size, 1)

    def forward(self, sample):
        x, edge_index = sample.x.double(), sample.edge_index
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv3(x,edge_index)

        return x #  F.log_softmax(x, dim=0)
    def get_parameters(self):
        return self.parameters()

class Net3(torch.nn.Module):
    def __init__(self, hidden_size,seq_len):
        super(Net3, self).__init__()
        self.conv1 = GCNConv(1, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.conv3 = GCNConv(hidden_size, 1)

    def forward(self, sample):
        x, edge_index = sample.x.double(), sample.edge_index
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x #  F.log_softmax(x, dim=0)

class Net4(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Net4, self).__init__()
        self.conv1 = GCNConv(1, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.conv3 = GCNConv(hidden_size, 1)

    def forward(self, sample):
        x, edge_index = sample.x.double(), sample.edge_index
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x #  F.log_softmax(x, dim=0)

def train_GCRN(model,Lstm,loader,criterion,learning_rate,seq_len,batch_size):
    model.train()
    Lstm.train()
    hidden = RNN.init_hidden(Lstm)
    print("In training")
    # hidden=model.init_hidden()
    epoch_losses = []
    # for data in loader:
    #     model.zero_grad()
    #     target_stack= torch.stack(data.y,1)
    #     for input in data.x:
    #         output=model(input)
    for rolling in range(0,len(loader)-seq_len):
        output_stack=[]
        target_stack=[]
        for i in range(rolling,rolling+seq_len):


            model.zero_grad()
            target = loader.dataset[i].y
            output = model(loader.dataset[i])
            output_stack.append(output)
            target_stack.append(target)
            # target=loader.dataset[i].y
            # output = model(loader.dataset[i])
            # if i < int(seq_len/2):
            #     output_stack.append(output)
            # else:
            #     target_stack.append(target)

        # test= loader.dataset[0]
        temp_stack_o = torch.stack(output_stack,1).cuda().float()
        temp_stack_t = torch.stack(target_stack,1).cuda().float()
        temp_stack_o = torch.transpose(temp_stack_o,0,2).cuda().float()
        temp_stack_t = torch.transpose(temp_stack_t,0,2).cuda().float()

        output, hidden = Lstm(temp_stack_o,hidden)
        # output = torch.squeeze(output,1)
        # target, hidden_t = Lstm(target_stack)
        # output1=torch.squeeze(output,-1)
        # target1=torch.squeeze(temp_stack_t,-1)
        loss = criterion(output, temp_stack_t).double()
        # loss.backward()
        loss.backward(retain_graph=True)
        print("Loss: " + str(loss.item()))
        epoch_losses.append(loss.item())

        for p in model.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)

    return sum(epoch_losses) / len(epoch_losses)

def evaluate_GCRN(set,model,Lstm,criterion,criterion2,loss_function,seq_len):
    model.eval()
    Lstm.eval()
    hidden = RNN.init_hidden(Lstm)

    losses=[]
    losses2=[]


    if seq_len>= len(set):
        output_stack=[]
        target_stack=[]
        for i in set:
            target = i.y
            output = model(i)
            output_stack.append(output)
            target_stack.append(target)
        temp_stack_o = torch.stack(output_stack, 1).cpu().float()
        temp_stack_t = torch.stack(target_stack, 1).cpu().float()
        temp_stack_o = torch.transpose(temp_stack_o, 0, 1).cpu().float()
        temp_stack_t = torch.transpose(temp_stack_t, 0, 1).cpu().float()

        output, hidden = Lstm(temp_stack_o,hidden)
        # output = torch.squeeze(output,1)
        # target, hidden_t = Lstm(target_stack)

        val_loss = criterion(output, temp_stack_t).double()
        val_loss2 = criterion2(output, temp_stack_t).double()

        losses.append(val_loss)
        losses2.append(val_loss2)
        if loss_function == "MAE":
            print("MAE Loss: " + str(val_loss.item()))
            print("MSE Loss: " + str(val_loss2.item()))
        else:
            print("MSE Loss: " + str(val_loss.item()))
            print("MAE Loss: " + str(val_loss2.item()))
    else:
        for rolling in range(len(set)-seq_len):
            output_stack=[]
            target_stack=[]
            for i in range(rolling,rolling+seq_len):

                target=set.dataset[i].y
                output = model(set.dataset[i])
                output_stack.append(output)
                target_stack.append(target)

            temp_stack_o = torch.stack(output_stack,1).cuda().float()
            temp_stack_t = torch.stack(target_stack,1).cuda().float()
            temp_stack_o = torch.transpose(temp_stack_o,0,1).cuda().float()
            temp_stack_t = torch.transpose(temp_stack_t,0,1).cuda().float()

            output, hidden = Lstm(temp_stack_o,hidden)
            # output = torch.squeeze(output,1)
            # target, hidden_t = Lstm(target_stack)



            val_loss = criterion(output, temp_stack_t).double()
            val_loss2 = criterion2(output, temp_stack_t).double()

            losses.append(val_loss)
            losses2.append(val_loss2)
            if loss_function =="MAE":
                print("MAE Loss: " + str(val_loss.item()))
                print("MSE Loss: " + str(val_loss2.item()))
            else:
                print("MSE Loss: " + str(val_loss.item()))
                print("MAE Loss: " + str(val_loss2.item()))

    return sum(losses)/len(losses), sum(losses2)/len(losses2)

def train_DCRNN(loader,model,criterion,learning_rate):
    losses=[]
    model.train()
    for entry in loader:
        # print(" ")
        model.zero_grad()
        output = model(entry)
        loss = criterion(output, entry.y).double()
        loss.backward()
        print("Loss: " + str(loss.item()))
        losses.append(loss.item())

        for p in model.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)

    return sum(losses) / len(losses)

def train2(model, data_loader,criterion,learning_rate,seq_len ):
    losses=[]

    for rolling in range(len(data_loader)-seq_len):
        inputs=[]
        targets=[]
        adjacency=[]
        for i in range(rolling,rolling+seq_len):
            input=data_loader.dataset[i].x
            target=data_loader.dataset[i].y
            inputs.append(input)
            targets.append(target)
        adjacency=data_loader.dataset[i].edge_matrix
        stacked_input=torch.stack(inputs)
        stacked_target=torch.stack(targets)


        output=model(adjacency,stacked_input)
        loss = criterion(output, stacked_target).double()
        loss.backward()
        print("Loss: " + str(loss.item()))
        losses.append(loss.item())

        for p in model.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)


    return sum(losses)/len(losses)