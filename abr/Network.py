import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    # actornetwork pass the test
    def __init__(self, state_dim, action_dim, n_conv=128, n_fc=128, n_fc1=128):
        super(ActorNetwork, self).__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.vectorOutDim = n_conv
        self.scalarOutDim = n_fc
        self.numFcInput = 2 * self.vectorOutDim * (
                self.s_dim[1] - 4 + 1) + 3 * self.scalarOutDim + self.vectorOutDim * (self.a_dim - 4 + 1)
        self.numFcOutput = n_fc1

        # define layer
        self.tConv1d = nn.Conv1d(1, self.vectorOutDim, 4)
        self.dConv1d = nn.Conv1d(1, self.vectorOutDim, 4)
        self.cConv1d = nn.Conv1d(1, self.vectorOutDim, 4)

        self.bufferFc = nn.Linear(1, self.scalarOutDim)
        self.leftChunkFc = nn.Linear(1, self.scalarOutDim)
        self.bitrateFc = nn.Linear(1, self.scalarOutDim)

        self.fullyConnected = nn.Linear(self.numFcInput, self.numFcOutput)
        self.outputLayer = nn.Linear(self.numFcOutput, self.a_dim)

    def forward(self, inputs):
        bitrateFcOut = F.relu(self.bitrateFc(inputs[:, 0:1, -1]), inplace=True)
        bufferFcOut = F.relu(self.bufferFc(inputs[:, 1:2, -1]), inplace=True)

        tConv1dOut = F.relu(self.tConv1d(inputs[:, 2:3, :]), inplace=True)
        dConv1dOut = F.relu(self.dConv1d(inputs[:, 3:4, :]), inplace=True)
        cConv1dOut = F.relu(self.cConv1d(inputs[:, 4:5, :self.a_dim]), inplace=True)
        leftChunkFcOut = F.relu(self.leftChunkFc(inputs[:, 5:6, -1]), inplace=True)

        t_flatten = tConv1dOut.view(tConv1dOut.shape[0], -1)
        d_flatten = dConv1dOut.view(dConv1dOut.shape[0], -1)
        c_flatten = cConv1dOut.view(dConv1dOut.shape[0], -1)

        fullyConnectedInput = torch.cat([bitrateFcOut, bufferFcOut, t_flatten, d_flatten, c_flatten, leftChunkFcOut], 1)
        fcOutput = F.relu(self.fullyConnected(fullyConnectedInput), inplace=True)

        out = torch.softmax(self.outputLayer(fcOutput), dim=-1)
        return out


class CriticNetwork(nn.Module):
    # return a value V(s,a)
    # the dim of state is not considered
    def __init__(self, state_dim, a_dim, n_conv=128, n_fc=128, n_fc1=128):
        super(CriticNetwork, self).__init__()
        self.s_dim = state_dim
        self.a_dim = a_dim
        self.vectorOutDim = n_conv
        self.scalarOutDim = n_fc
        self.numFcInput = 2 * self.vectorOutDim * (
                self.s_dim[1] - 4 + 1) + 3 * self.scalarOutDim + self.vectorOutDim * (self.a_dim - 4 + 1)
        self.numFcOutput = n_fc1

        # define layer
        self.tConv1d = nn.Conv1d(1, self.vectorOutDim, 4)
        self.dConv1d = nn.Conv1d(1, self.vectorOutDim, 4)
        self.cConv1d = nn.Conv1d(1, self.vectorOutDim, 4)

        self.bufferFc = nn.Linear(1, self.scalarOutDim)
        self.leftChunkFc = nn.Linear(1, self.scalarOutDim)
        self.bitrateFc = nn.Linear(1, self.scalarOutDim)

        self.fullyConnected = nn.Linear(self.numFcInput, self.numFcOutput)
        self.outputLayer = nn.Linear(self.numFcOutput, 1)

    def forward(self, inputs):
        bitrateFcOut = F.relu(self.bitrateFc(inputs[:, 0:1, -1]), inplace=True)
        bufferFcOut = F.relu(self.bufferFc(inputs[:, 1:2, -1]), inplace=True)

        tConv1dOut = F.relu(self.tConv1d(inputs[:, 2:3, :]), inplace=True)
        dConv1dOut = F.relu(self.dConv1d(inputs[:, 3:4, :]), inplace=True)
        cConv1dOut = F.relu(self.cConv1d(inputs[:, 4:5, :self.a_dim]), inplace=True)
        leftChunkFcOut = F.relu(self.leftChunkFc(inputs[:, 5:6, -1]), inplace=True)

        t_flatten = tConv1dOut.view(tConv1dOut.shape[0], -1)
        d_flatten = dConv1dOut.view(dConv1dOut.shape[0], -1)
        c_flatten = cConv1dOut.view(dConv1dOut.shape[0], -1)

        fullyConnectedInput = torch.cat([bitrateFcOut, bufferFcOut, t_flatten, d_flatten, c_flatten, leftChunkFcOut], 1)
        fcOutput = F.relu(self.fullyConnected(fullyConnectedInput), inplace=True)

        out = self.outputLayer(fcOutput)
        return out
