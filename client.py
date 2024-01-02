from __future__ import division

import random

import numpy as np

# import buffer
# import rl_abr.envs as abr_env
import utils,os,time
import matplotlib.pyplot as plt
import model_ac_torch as ac
import env, env_wrapper, load_trace
from variant_vmaf.utils.helper import check_folder
from torch.utils.tensorboard import SummaryWriter
import torch
import linearNet.agent as linear_agent

VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY_LOG = 2.66  # 1 sec rebuffering -> 3 Mbps
REBUF_PENALTY_LIN = 4.3
SMOOTH_PENALTY = 0.2
DEFAULT_QUALITY = 1
S_INFO = 8
S_LEN = 8
args = {"test":True}

SUMMARY_DIR = "./Results/sim"
add_str = "maml"
summary_dir = os.path.join(*[SUMMARY_DIR, add_str])
check_folder(summary_dir)
log_str = "log"
summary_dir = os.path.join(*[summary_dir, log_str])
check_folder(summary_dir)
ts = time.strftime("%b%d-%H:%M:%S", time.gmtime())
log_file_path = os.path.join(*[summary_dir, ts])
check_folder(log_file_path)
log_file_name = log_file_path + "/log"
writer = SummaryWriter(log_file_path)

all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace()
train_env1 = env.Environment(
    all_cooked_time=all_cooked_time,
    all_cooked_bw=all_cooked_bw,
    all_file_names=all_file_names,
    a2br=True,
)
train_env2 = env.Environment(
    all_cooked_time=all_cooked_time,
    all_cooked_bw=all_cooked_bw,
    all_file_names=all_file_names,
    a2br=True,
)



MAX_BUFFER = 1000000
# MAX_TOTAL_REWARD = 300
ABR_DIM = 6
ABR_STATE_DIM = 7
PLACE_STATE_DIM = 3
A_MAX = 1

# ram1 = buffer.MemoryBuffer(MAX_BUFFER)
# ram2 = buffer.MemoryBuffer(MAX_BUFFER)
# abr_trainer = train.Trainer(ABR_STATE_DIM, ABR_DIM, A_MAX, ram1)
# place_trainer = train.Trainer(PLACE_STATE_DIM, SPACE_DIM, A_MAX, ram2)
#
# abr_trainer.load_models(msg="abr", episode=500)
# place_trainer.load_models(msg="place", episode=600)

class Client:
    def __init__(self, agent, env:env_wrapper.VirtualPlayer):
        self.agent = agent
        self.env = env
        self.K_RANGE = [5, 10]

        self.base_sr_delay = 2

        self.abr_reward_list = []
        self.place_reward_list = []
        self.full_reward_list = []
        self.maybe_do_sr_reward_list = []

        self.t1 = []
        self.t2 = []

        # 画图
        self.plt_interval = 1

    def reset_plot(self):
        self.abr_reward_list = []
        self.place_reward_list = []
        self.full_reward_list = []
        self.maybe_do_sr_reward_list = []

    def add_reward(self, abr_reward, place_reward):
        self.abr_reward_list.append(abr_reward)
        self.place_reward_list.append(place_reward)
        self.full_reward_list.append(abr_reward + place_reward)

    def maybe_do_sr(self, past_chunk_throughput, new_bitrate):
        cur_compute_resource = utils.get_random_gauss_val(self.K_RANGE[0], self.K_RANGE[1])
        t1 = 30 * ((new_bitrate + ABR_DIM) / ABR_DIM) / cur_compute_resource * self.base_sr_delay / 6  # 单个视野是整个数据量的1/6
        t2 = min(new_bitrate + 2, ABR_DIM - 1) * 1e5 / past_chunk_throughput
        self.t1.append(t1)
        self.t2.append(t2)
        if t1 > t2:
            t1, t2 = t2, t1
        if random.randint(0, 10) / 10 < 0.6:
            self.maybe_do_sr_reward_list.append(t2 - t1)
        else:
            self.maybe_do_sr_reward_list.append(0)

        return t1, t2 - t1  # 实际花费时间，节省时间量

    def run(self, max_steps, trace_idx):
        # self.env.reset()
        self.env.env.set_trace_idx(trace_idx)
        state = init_state(self.env.env)
        for r in range(max_steps):

            bit_rate, place_choice = self.agent.get_exploration_action(state)
            if place_choice == 0:
                source_bitrate = max(bit_rate - 2, -1)
                new_bitrate = bit_rate
            else:
                source_bitrate = bit_rate
                new_bitrate = min(ABR_DIM - 1, bit_rate + 2)

            (
                delay,
                sr_delay,
                sleep_time,
                buffer_size,
                rebuf,
                video_chunk_size,
                next_video_chunk_sizes,
                end_of_video,
                video_chunk_remain,
                place_reward,
                edge_k,
                client_k
            ) = self.env.env.get_video_chunk(
                source_bitrate, place_choice
            )  ## sample in the environment of virtual player
            trans_delay = delay - sr_delay

            reward = self.env.bitrate_versions[bit_rate] / 1000
            reward -= self.env.rebuff_p * rebuf
            reward -= 0.5 * self.env.smooth_p * np.abs(self.env.bitrate_versions[bit_rate] - self.env.bitrate_versions[self.env.last_bit_rate]) / 1000
            self.video_chunk_remain = video_chunk_remain

            self.env.last_bit_rate = bit_rate

            print(str(bit_rate),'\t', str(int(trans_delay)),'\t', str(sleep_time)[:4],'\t', str(buffer_size)[:4],'\t', str(rebuf)[:4],'\t', str(int(state[6, -1]))[:4],'\t', str(int(state[7, -1]))[:4],
                  '\t', place_choice,'\t', place_reward,'\t', reward)
            self.add_reward(reward, place_reward)
            self.maybe_do_sr(past_chunk_throughput=float(video_chunk_size) / float(trans_delay) / M_IN_K, new_bitrate=new_bitrate)

            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(
                np.max(VIDEO_BIT_RATE)
            )  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = (
                    float(video_chunk_size) / float(trans_delay) / M_IN_K
            )  # kilo byte / ms
            state[3, -1] = float(trans_delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :ABR_DIM] = (
                    np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K / 10
            )  # mega byte
            state[5, -1] = np.minimum(
                video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP
            ) / float(CHUNK_TIL_VIDEO_END_CAP)
            state[6, -1] = float(edge_k)
            state[7, -1] = client_k

            # state = torch.from_numpy(state)

            if end_of_video:
                break
        # self.plot_all()

    def plot_all(self):
        self.sample_plot(self.abr_reward_list, self.plt_interval, "abr_reward")
        self.sample_plot(self.place_reward_list, self.plt_interval, "place_reward")
        self.sample_plot(self.full_reward_list, self.plt_interval, "full_reward")
        self.sample_plot(self.maybe_do_sr_reward_list, self.plt_interval, "maybe_do_sr_reward")
        self.sample_plot(self.t1, self.plt_interval, "t1")
        self.sample_plot(self.t2, self.plt_interval, "t2")

    def sample_plot(self, vals, interval, msg):  # 按interval采样vals并绘图
        plt.figure()
        new_vals = self.resize(vals, interval)
        num_of_x = len(new_vals)
        plt.plot([i * interval for i in range(num_of_x)],
                 new_vals, label=msg)
        plt.legend()
        plt.xlabel("chunk_idx")
        plt.ylabel(msg)
        plt.savefig('./plot_results/' + msg + '.png')

    def resize(self, val_list, a):
        new_list = []
        for idx, e in enumerate(val_list):
            if idx % a == 0:
                new_list.append(e)
        return new_list

def init_state(net_env):
    state = np.zeros((S_INFO, S_LEN))
    # state = torch.from_numpy(state)
    edge_k, client_k = net_env.reset(net=False)
    state[6, -1] = float(edge_k)
    state[7, -1] = client_k
    return state

def plot_step_of_diff_algo(y_lable, step_values, algo_names, pre_msg):
    plt.figure()
    for i, single_algo_step_values in enumerate(step_values):
        sample_plot(vals=single_algo_step_values, msg=algo_names[i])
        plt.legend()
    plt.xlabel("Chunk Idx")
    plt.ylabel(y_lable)
    plt.savefig('./plot_results/' + pre_msg + '.png')


def sample_plot(vals, msg):
    plt.plot([i for i in range(len(vals))],
             vals, label=msg)

if __name__ == "__main__":
    with open(log_file_name + "_record", "w") as log_file:
        env1 = env_wrapper.VirtualPlayer(args, train_env1, log_file)
        env2 = env_wrapper.VirtualPlayer(args, train_env2, log_file)

        maml_agent = ac.Agent()
        maml_agent.load(abr_epoch=340, place_epoch=3150)
        normal_agent = linear_agent.Agent()
        normal_agent.load(abr_epoch=500, place_epoch=600)


        algo_names = ["MAML", "NORMAL"]
        client1 = Client(maml_agent, env1)
        client2 = Client(normal_agent, env2)
        start = 50
        for i in range(start, start + 5):
            client1.run(10000, trace_idx=i)
            client2.run(10000, trace_idx=i)

            plot_step_of_diff_algo("abr_reward", [client1.abr_reward_list, client2.abr_reward_list],
                                   algo_names, pre_msg="abr_reward_net" + str(i) )
            plot_step_of_diff_algo("place_reward", [client1.place_reward_list, client2.place_reward_list],
                                   algo_names, pre_msg="place_reward_net" + str(i) )
            client1.reset_plot()
            client2.reset_plot()

