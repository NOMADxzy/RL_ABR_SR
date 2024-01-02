import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(torch.nn.Module):
    def __init__(self, action_space):
        super(Actor, self).__init__()
        self.input_channel = 1
        self.action_space = action_space
        channel_cnn = 128
        channel_fc = 128

        # self.bn = nn.BatchNorm1d(self.input_channel)

        self.actor_conv1 = nn.Conv1d(self.input_channel, channel_cnn, 4)  # L_out = 8 - (4-1) -1 + 1 = 5
        self.actor_conv2 = nn.Conv1d(self.input_channel, channel_cnn, 4)
        # self.actor_conv3 = nn.Conv1d(self.input_channel, channel_cnn, 4) # for available chunk sizes 6 version  L_out = 6 - (4-1) -1 + 1 = 3
        # self.actor_fc_1 = nn.Linear(self.input_channel, channel_fc)
        self.actor_fc_2 = nn.Linear(self.input_channel, channel_fc)
        self.actor_fc_3 = nn.Linear(self.input_channel, channel_fc)

        # ===================Hide layer=========================
        incoming_size = 2*channel_cnn*5 + 2 * channel_fc #
        # incoming_size = 2 * channel_cnn * 5 + channel_cnn * 3 + 3 * channel_fc  #

        self.fc1 = nn.Linear(in_features=incoming_size, out_features=channel_fc)
        # self.fc2 = nn.Linear(in_features=channel_fc, out_features=channel_fc)
        self.fc3 = nn.Linear(in_features=channel_fc, out_features=self.action_space)
        # self.fc4 = nn.Linear(in_features=channel_fc, out_features=1)

    def forward(self, inputs):  # (上个片比特率大小，缓冲区大小，吞吐量，时延，下个片的大小，剩余片数量)
        throughputs_batch = inputs[:, 2:3, :]  ## refer to env_train.py
        # throughputs_batch = self.bn(throughputs_batch)

        download_time_batch = inputs[:, 3:4, :]

        sizes_batch = inputs[:, 4:5, :self.action_space]

        x_1 = F.relu(self.actor_conv1(throughputs_batch))
        x_2 = F.relu(self.actor_conv2(download_time_batch))
        # x_3 = F.relu(self.actor_conv3(sizes_batch))
        x_4 = F.relu(self.actor_fc_3(inputs[:, 0:1, -1]))
        x_5 = F.relu(self.actor_fc_2(inputs[:, 1:2, -1]))
        # x_6 = F.relu(self.actor_fc_1(inputs[:, 5:6, -1]))

        x_1 = x_1.view(-1, self.num_flat_features(x_1))
        x_2 = x_2.view(-1, self.num_flat_features(x_2))
        # x_3 = x_3.view(-1, self.num_flat_features(x_3))
        x_4 = x_4.view(-1, self.num_flat_features(x_4))
        x_5 = x_5.view(-1, self.num_flat_features(x_5))
        # x_6 = x_6.view(-1, self.num_flat_features(x_6))

        x = torch.cat([x_1, x_2, x_4, x_5], 1)
        x = F.relu(self.fc1(x))
        # actor
        # actor = F.relu(self.fc1(x))
        # actor = F.relu(self.fc2(actor))
        actor = F.softmax(self.fc3(x), dim=1)
        return actor

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Critic(torch.nn.Module):
    def __init__(self, action_space):
        super(Critic, self).__init__()
        self.input_channel = 1
        self.action_space = action_space
        channel_cnn = 128
        channel_fc = 128

        # self.bn = nn.BatchNorm1d(self.input_channel)

        self.critic_conv1 = nn.Conv1d(self.input_channel, channel_cnn, 4)  # L_out = 8 - (4-1) -1 + 1 = 5
        self.critic_conv2 = nn.Conv1d(self.input_channel, channel_cnn, 4)
        # self.critic_conv3 = nn.Conv1d(self.input_channel, channel_cnn, 4) # for available chunk sizes 6 version  L_out = 6 - (4-1) -1 + 1 = 3
        # self.critic_fc_1 = nn.Linear(self.input_channel, channel_fc)
        self.critic_fc_2 = nn.Linear(self.input_channel, channel_fc)
        self.critic_fc_3 = nn.Linear(self.input_channel, channel_fc)

        # ===================Hide layer=========================
        incoming_size = 2 * channel_cnn * 5 + 2 * channel_fc  #
        # incoming_size = 2 * channel_cnn * 5 + channel_cnn*3 + 3 * channel_fc  #

        self.fc1 = nn.Linear(in_features=incoming_size, out_features=channel_fc)
        # self.fc2 = nn.Linear(in_features=channel_fc, out_features=channel_fc)
        self.fc3 = nn.Linear(in_features=channel_fc, out_features=1)

    def forward(self, inputs):
        throughputs_batch = inputs[:, 2:3, :]  ## refer to env_train.py
        # throughputs_batch = self.bn(throughputs_batch)

        download_time_batch = inputs[:, 3:4, :]
        # download_time_batch = self.bn(download_time_batch)

        sizes_batch = inputs[:, 4:5, :self.action_space]
        # sizes_batch = self.bn(sizes_batch)

        x_1 = F.relu(self.critic_conv1(throughputs_batch))
        x_2 = F.relu(self.critic_conv2(download_time_batch))
        # x_3 = F.relu(self.critic_conv3(sizes_batch))
        x_4 = F.relu(self.critic_fc_3(inputs[:, 0:1, -1]))
        x_5 = F.relu(self.critic_fc_2(inputs[:, 1:2, -1]))
        # x_6 = F.relu(self.critic_fc_3(inputs[:, 0:1, -1]))

        x_1 = x_1.view(-1, self.num_flat_features(x_1))
        x_2 = x_2.view(-1, self.num_flat_features(x_2))
        # x_3 = x_3.view(-1, self.num_flat_features(x_3))
        x_4 = x_4.view(-1, self.num_flat_features(x_4))
        x_5 = x_5.view(-1, self.num_flat_features(x_5))
        # x_6 = x_6.view(-1, self.num_flat_features(x_6))

        x = torch.cat([x_1, x_2, x_4, x_5, ], 1)
        x = F.relu(self.fc1(x))
        # critic
        # critic = F.relu(self.fc1(x))
        # critic = F.relu(self.fc2(critic))
        critic = self.fc3(x)
        return critic

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PlaceActor(torch.nn.Module):
    def __init__(self, action_space):
        super(PlaceActor, self).__init__()
        self.input_channel = 1
        self.action_space = action_space
        channel_cnn = 128
        channel_fc = 128

        self.actor_conv1 = nn.Conv1d(self.input_channel, channel_cnn, 4) # L_out = 8 - (4-1) -1 + 1 = 5
        # self.actor_conv2 = nn.Conv1d(self.input_channel, channel_cnn, 4)

        # self.actor_fc_1 = nn.Linear(self.input_channel, channel_fc)
        # self.actor_fc_2 = nn.Linear(self.input_channel, channel_fc)
        self.actor_fc_3 = nn.Linear(self.input_channel, channel_fc)
        self.actor_fc_4 = nn.Linear(self.input_channel, channel_fc)

        # ===================Hide layer=========================
        # incoming_size = 2 * channel_cnn * 5 + 4 * channel_fc  #
        incoming_size = 1 * channel_cnn * 5 + 2 * channel_fc  #

        self.fc1 = nn.Linear(in_features=incoming_size, out_features=channel_fc)
        self.fc3 = nn.Linear(in_features=channel_fc, out_features=self.action_space)

    def forward(self, inputs):  # (上个片比特率大小，缓冲区大小，吞吐量，时延，下个片的大小，剩余片数量)
        throughputs_batch = inputs[:, 2:3, :]  ## 过去8个时间的吞吐量
        download_time_batch = inputs[:, 3:4, :]

        x_1 = F.relu(self.actor_conv1(throughputs_batch))
        # x_2 = F.relu(self.actor_conv2(download_time_batch))
        # x_3 = F.relu(self.actor_fc_1(inputs[:, 1:2, -1]))
        # x_4 = F.relu(self.actor_fc_2(inputs[:, 2:3, -1]))
        x_5 = F.relu(self.actor_fc_3(inputs[:, 6:7, -1]))
        x_6 = F.relu(self.actor_fc_4(inputs[:, 7:8, -1]))

        x_1 = x_1.view(-1, self.num_flat_features(x_1))
        # x_2 = x_2.view(-1, self.num_flat_features(x_2))
        # x_3 = x_3.view(-1, self.num_flat_features(x_3))
        # x_4 = x_4.view(-1, self.num_flat_features(x_4))
        x_5 = x_5.view(-1, self.num_flat_features(x_5))
        x_6 = x_6.view(-1, self.num_flat_features(x_6))

        x = torch.cat([x_1, x_5, x_6], 1)
        x = F.relu(self.fc1(x))
        # actor
        # actor = F.relu(self.fc1(x))
        # actor = F.relu(self.fc2(actor))
        actor = F.softmax(self.fc3(x), dim=1)
        return actor

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PlaceCritic(torch.nn.Module):
    def __init__(self, action_space):
        super(PlaceCritic, self).__init__()
        self.input_channel = 1
        self.action_space = action_space
        channel_cnn = 128
        channel_fc = 128

        # self.bn = nn.BatchNorm1d(self.input_channel)

        self.critic_conv1 = nn.Conv1d(self.input_channel, channel_cnn, 4)  # L_out = 8 - (4-1) -1 + 1 = 5
        # self.critic_conv2 = nn.Conv1d(self.input_channel, channel_cnn, 4)

        # self.critic_fc_1 = nn.Linear(self.input_channel, channel_fc)
        # self.critic_fc_2 = nn.Linear(self.input_channel, channel_fc)
        self.critic_fc_3 = nn.Linear(self.input_channel, channel_fc)
        self.critic_fc_4 = nn.Linear(self.input_channel, channel_fc)

        # ===================Hide layer=========================
        # incoming_size = 2 * channel_cnn * 5 + 4 * channel_fc  #
        incoming_size = 1 * channel_cnn * 5 + 2 * channel_fc  #

        self.fc1 = nn.Linear(in_features=incoming_size, out_features=channel_fc)
        # self.fc2 = nn.Linear(in_features=channel_fc, out_features=channel_fc)
        self.fc3 = nn.Linear(in_features=channel_fc, out_features=self.action_space)
        # self.fc4 = nn.Linear(in_features=channel_fc, out_features=1)

    def forward(self, inputs):  # (上个片比特率大小，缓冲区大小，吞吐量，时延，下个片的大小，剩余片数量)
        throughputs_batch = inputs[:, 2:3, :]  ## 过去8个时间的吞吐量
        download_time_batch = inputs[:, 3:4, :]

        x_1 = F.relu(self.critic_conv1(throughputs_batch))
        # x_2 = F.relu(self.critic_conv2(download_time_batch))
        # x_3 = F.relu(self.critic_fc_1(inputs[:, 1:2, -1]))
        # x_4 = F.relu(self.critic_fc_2(inputs[:, 2:3, -1]))
        x_5 = F.relu(self.critic_fc_3(inputs[:, 6:7, -1]))
        x_6 = F.relu(self.critic_fc_4(inputs[:, 7:8, -1]))

        x_1 = x_1.view(-1, self.num_flat_features(x_1))
        # x_2 = x_2.view(-1, self.num_flat_features(x_2))
        # x_3 = x_3.view(-1, self.num_flat_features(x_3))
        # x_4 = x_4.view(-1, self.num_flat_features(x_4))
        x_5 = x_5.view(-1, self.num_flat_features(x_5))
        x_6 = x_6.view(-1, self.num_flat_features(x_6))

        x = torch.cat([x_1, x_5, x_6], 1)
        x = F.relu(self.fc1(x))
        # actor
        # actor = F.relu(self.fc1(x))
        # actor = F.relu(self.fc2(actor))
        actor = F.softmax(self.fc3(x), dim=1)
        return actor

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


ABR_DIM = 6
SPACE_DIM = 2
class Agent:
    def __init__(self):
        abr_actor = Actor(ABR_DIM)
        place_actor = PlaceActor(SPACE_DIM)
        self.abr_actor = abr_actor
        self.place_actor = place_actor
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    def load(self, abr_epoch=0, place_epoch=0, path="./Results/checkpoints"):
        if abr_epoch > 0:
            self.abr_actor.load_state_dict(torch.load(path + "/abr_actor" + str(abr_epoch) + ".pt"))
        if place_epoch > 0:
            self.place_actor.load_state_dict(torch.load(path + "/place_actor" + str(place_epoch) + ".pt"))

    def get_exploration_action(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            abr_prob = self.abr_actor(state[:].unsqueeze(0).type(self.dtype))
            place_prob = self.place_actor(state[:].unsqueeze(0).type(self.dtype))
        abr_action = abr_prob.multinomial(num_samples=1).detach()
        place_action = place_prob.multinomial(num_samples=1).detach()
        bit_rate = int(abr_action.squeeze().cpu().numpy())
        place_choice = int(place_action.squeeze().cpu().numpy())
        return bit_rate,place_choice

