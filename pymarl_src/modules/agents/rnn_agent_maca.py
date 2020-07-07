import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.conv1 = nn.Sequential(  # 100 * 100 * 3
            nn.Conv2d(
                in_channels=5,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(  # 50 * 50 * 16
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 25 * 25 * 32
        )

        # self.info_fc = nn.Sequential(
        #     nn.Linear(3, 256),
        #     nn.Tanh(),
        # )
        self.fc1 = nn.Linear(25 * 25 * 32, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, img, hidden_state):
        img_feature = self.conv1(img)
        img_feature = self.conv2(img_feature)
        # # info_feature = self.info_fc(info)
        # combined = torch.cat((img_feature.view(img_feature.size(0), -1), info_feature.view(info_feature.size(0), -1)),
        #                      dim=1)
        x = F.relu(self.fc1(img_feature.view(img_feature.size(0), -1)))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
