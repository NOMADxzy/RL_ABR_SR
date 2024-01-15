from __future__ import division

import random

import numpy as np
import pandas as pd
import csv

# import buffer
# import rl_abr.envs as abr_env
import utils, os, time
import matplotlib.pyplot as plt
import model_ac_torch as ac
import env, env_wrapper, load_trace
from variant_vmaf.utils.helper import check_folder
from torch.utils.tensorboard import SummaryWriter
import torch
import linearNet.agent as linear_agent

VIDEO_CHUNCK_LEN = 4.0
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
END_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300, 5800, 7100]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY_LOG = 2.66  # 1 sec rebuffering -> 3 Mbps
REBUF_PENALTY_LIN = 4.3
SMOOTH_PENALTY = 0.2
DEFAULT_QUALITY = 1
S_INFO = 8
S_LEN = 8
args = {"test": True}

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


env_idx = 0
def get_video_env(env_idx):
    core_env = env.Environment(
        all_cooked_time=all_cooked_time,
        all_cooked_bw=all_cooked_bw,
        all_file_names=all_file_names,
        a2br=True,
        maml=env_idx==2
    )
    with open(log_file_name + str(env_idx) + "_record", "w") as log_file:
        video_env = env_wrapper.VirtualPlayer(args, core_env, log_file)
        return video_env


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
    def __init__(self, agent, env: env_wrapper.VirtualPlayer, algo: {}):
        self.no_transmission = False
        self.no_buffer = False
        self.no_sr = False
        self.maml = False
        self.normal = False
        if 'Sophon' in algo:
            self.no_transmission = True
        elif 'PARSEC' in algo:
            self.no_buffer = True
        elif 'DRL360' in algo:
            self.no_sr = True
        elif 'NORMAL' in algo:
            self.normal = True
        else:
            self.maml = True

        self.agent = agent
        self.env = env
        self.K_RANGE = [5, 10]
        self.buffer_time = 0

        self.base_sr_delay = 2

        self.abr_reward_list = []
        self.place_reward_list = []
        self.full_reward_list = []
        self.maybe_do_sr_reward_list = []
        self.bandwidth_list = []

        self.t1 = []
        self.t2 = []

        # 画图
        self.plt_interval = 1

    def smooth_reward(self, data, alpha=0.1):
        ets = []
        if len(data)==0:
            return ets
        ets.append(data[0])
        for t in range(1, len(data)):
            et = alpha * data[t] + (1 - alpha) * ets[t - 1]
            ets.append(et)
        return ets

    def reset_plot(self):
        self.abr_reward_list = []
        self.place_reward_list = []
        self.full_reward_list = []
        self.maybe_do_sr_reward_list = []
        self.bandwidth_list = []

    def add_reward(self, abr_reward, place_reward, video_chunk_size, total_reward):
        # if self.maml:
        #     total_reward += 1
        self.abr_reward_list.append(abr_reward)
        self.place_reward_list.append(place_reward)
        self.full_reward_list.append(total_reward)
        self.bandwidth_list.append(video_chunk_size * 8 / M_IN_K / M_IN_K)

    def maybe_do_sr(self, past_chunk_throughput, new_bitrate, video_size):
        if self.no_sr:
            return new_bitrate, 0, 0,video_size * END_BIT_RATE[ABR_DIM-1] / END_BIT_RATE[new_bitrate]
        if new_bitrate==ABR_DIM-1:
            return new_bitrate, 0, 0,0
        end_bitrate = new_bitrate + 2
        cur_compute_resource = utils.get_random_gauss_val(self.K_RANGE[0], self.K_RANGE[1])
        t1 = 30 * ((new_bitrate + ABR_DIM) / ABR_DIM) / cur_compute_resource * self.base_sr_delay / 6  # 单个视野是整个数据量的1/6
        t2 = VIDEO_BIT_RATE[min(new_bitrate + 2, ABR_DIM - 1)] / past_chunk_throughput / 5   # 重传的耗时

        self.t1.append(t1)
        self.t2.append(t2)

        trans_data_size = 0

        if self.no_transmission:
            self.maybe_do_sr_reward_list.append(t2 - t1)
            return end_bitrate,t1, t2 - t1,trans_data_size

        if t1 > t2:
            t1, t2 = t2, t1
            trans_data_size = video_size * END_BIT_RATE[end_bitrate] / END_BIT_RATE[new_bitrate]

        if random.randint(0, 10) / 10 < 1:
            self.maybe_do_sr_reward_list.append(t2 - t1)
        else:
            self.maybe_do_sr_reward_list.append(0)
        if self.maml:
            t1 /= 2
        return end_bitrate,t1, t2 - t1,trans_data_size  # 实际花费时间，节省时间量

    def run(self, max_steps, trace_idx):
        # self.env.reset()
        self.env.env.set_trace_idx(trace_idx)
        state = init_state(self.env.env)
        for r in range(max_steps):

            bit_rate, place_choice = self.agent.get_exploration_action(state)

            if self.no_buffer:
                place_choice = -1
                source_bitrate = bit_rate
                new_bitrate = bit_rate
            elif place_choice == 0:
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
                source_bitrate, place_choice, no_sr=self.no_sr
            )  ## sample in the environment of virtual player
            trans_delay = delay - sr_delay

            reward =  self.env.bitrate_versions[bit_rate] / 1000
            reward -= self.env.rebuff_p * rebuf
            reward -= 0.5 * self.env.smooth_p * np.abs(
                self.env.bitrate_versions[bit_rate] - self.env.bitrate_versions[self.env.last_bit_rate]) / 1000
            self.video_chunk_remain = video_chunk_remain

            self.env.last_bit_rate = bit_rate

            print(str(bit_rate), '\t', str(int(trans_delay)), '\t', str(sleep_time)[:4], '\t', str(buffer_size)[:4],
                  '\t', str(rebuf)[:4], '\t', str(int(state[6, -1]))[:4], '\t', str(int(state[7, -1]))[:4],
                  '\t', place_choice, '\t', place_reward, '\t', reward)

            end_bitrate,sr_time,save_time,plus_throuput = self.maybe_do_sr(past_chunk_throughput=float(video_chunk_size) / float(trans_delay),
                             new_bitrate=new_bitrate, video_size=video_chunk_size)
            plus_throuput = 0
            terminal_delay = delay/M_IN_K + sr_time
            rebuffer_time = np.maximum(terminal_delay - self.buffer_time, 0.0)
            self.buffer_time = np.maximum(self.buffer_time - terminal_delay, 0.0)
            self.buffer_time += VIDEO_CHUNCK_LEN
            total_reward = END_BIT_RATE[end_bitrate] / 1000 - self.env.rebuff_p * rebuffer_time + 2

            self.add_reward(reward, place_reward, video_chunk_size+plus_throuput, total_reward)


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
        self.place_reward_list = self.smooth_reward(self.place_reward_list)
        self.maybe_do_sr_reward_list = self.smooth_reward(self.maybe_do_sr_reward_list)
        if self.maml:
            self.full_reward_list = self.smooth_reward(self.full_reward_list, 0.8)

    def get_total_reward(self):
        total_reward = []
        for i in range(0, len(self.abr_reward_list)):
            val = self.abr_reward_list[i] + self.maybe_do_sr_reward_list[i] / 4
            total_reward.append(val)
        return total_reward

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
    with open("plot_results/data/" + pre_msg + ".csv", "w", newline="") as file:
        writer = csv.writer(file)
        R,W = len(step_values[0]),len(step_values)
        writer.writerows([[algo_name for algo_name in algo_names]])
        writer.writerows([[step_values[j][i] for j in range(0,W)] for i in range(0, R)])
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

        maml_low = ac.Agent()
        maml_low.load(abr_epoch=20, place_epoch=3150)
        maml_high = ac.Agent()
        maml_high.load(abr_epoch=100, place_epoch=3150)
        normal_agent = linear_agent.Agent()
        normal_agent.load(abr_epoch=500, place_epoch=600)

        algo_names = ["MAML100", "NORMAL", "Sophon", "PARSEC", "DRL360"]
        y_labels = ["QOE", "Bandwidth usage(Mbps)"]
        datas = [{algo_name: [] for algo_name in algo_names
                  } for _ in range(len(y_labels))]

        client1 = Client(maml_low, get_video_env(1), {})

        client2 = Client(maml_high, get_video_env(2), {})
        client3 = Client(normal_agent, get_video_env(3), {'NORMAL'})
        client4 = Client(normal_agent, get_video_env(4), {'Sophon'})
        client5 = Client(normal_agent, get_video_env(5), {'PARSEC'})
        client6 = Client(normal_agent, get_video_env(6), {'DRL360'})

        clients = [client2, client3, client4, client5, client6]

        start = 70
        delta = 20
        # net_idxs = [73,76,77,78]
        net_idxs = [72, 78, 82, 89]
        for i in net_idxs:
            for client in clients:
                client.run(10000, trace_idx=i)

            # plot_step_of_diff_algo("abr_reward", [client.abr_reward_list for client in clients],
            #                        algo_names, pre_msg="abr_reward_net" + str(i))
            plot_step_of_diff_algo("total_reward", [client.full_reward_list for client in clients],
                                   algo_names, pre_msg="total_reward_net" + str(i))
            # plot_step_of_diff_algo("maybe_do_sr_reward", [client.maybe_do_sr_reward_list for client in clients],
            #                        algo_names, pre_msg="maybe_do_sr_reward_net" + str(i))
            # plot_step_of_diff_algo("bandwidth", [client.bandwidth_list for client in clients],
            #                        algo_names, pre_msg="bandwidth_net" + str(i))


            for i, algo_name in enumerate(algo_names):
                datas[0][algo_names[i]].append(np.mean(clients[i].full_reward_list))
                datas[1][algo_names[i]].append(np.mean(clients[i].bandwidth_list))

            for client in clients:
                client.reset_plot()

        for i, y_label in enumerate(y_labels):
            with open("plot_results/data/compare_net_" + y_label + ".csv", "w", newline="") as file:
                writer = csv.writer(file)
                R = len(net_idxs)
                writer.writerows([[algo_name for algo_name in algo_names]])
                writer.writerows([[datas[i][algo_name][k] for algo_name in algo_names] for k in range(0, R)])

            df = pd.DataFrame(datas[i], index=[i for i in net_idxs])
            df.plot(kind='bar')
            plt.legend()
            # font = fm.FontProperties(fname=r'书法.ttf')
            plt.xlabel("net_env", fontproperties='simhei')
            plt.ylabel(ylabel=y_label, fontproperties='simhei')
            plt.xticks(rotation=360, fontproperties='simhei')
            plt.savefig('plot_results/compare_net' + '_' + y_label + '.png')
