from __future__ import division

import random

import numpy as np

import train
import buffer
import rl_abr.envs as abr_env
import utils
import matplotlib.pyplot as plt

env = abr_env.ABRSimEnv(trace_type = 'n_train', obs_chunk_len=4, normalize_obs=True)


class Client:
    def __init__(self, abr_trainer:train.Trainer, place_trainer:train.Trainer):
        self.abr_trainer = abr_trainer
        self.place_trainer = place_trainer
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


    def add_reward(self, reward):
        r1,r2 = reward
        self.abr_reward_list.append(r1)
        self.place_reward_list.append(r2)
        self.full_reward_list.append(r1 + r2)

    def maybe_do_sr(self, past_chunk_throughput, new_bitrate):
        cur_compute_resource = utils.get_random_gauss_val(self.K_RANGE[0], self.K_RANGE[1])
        t1 = 30 * ((new_bitrate+ABR_DIM)/ABR_DIM) / cur_compute_resource*self.base_sr_delay / 6  # 单个视野是整个数据量的1/6
        t2 = min(new_bitrate + 2, ABR_DIM - 1) * 1e5 / past_chunk_throughput
        self.t1.append(t1)
        self.t2.append(t2)
        if t1 > t2:
            t1, t2 = t2, t1
        if random.randint(0, 10)/10 < 0.6:
            self.maybe_do_sr_reward_list.append(t2 - t1)
        else:
            self.maybe_do_sr_reward_list.append(0)

        return t1, t2 - t1 # 实际花费时间，节省时间量

    def run(self, max_steps):
        observation = env.reset()
        for r in range(max_steps):
            env.render()
            # time.sleep(1)
            state = np.float32(observation)

            _, abr_action, _ = self.abr_trainer.get_exploration_action(state[:ABR_STATE_DIM])
            _, place_action, _ = self.place_trainer.get_exploration_action(np.asarray([state[0], state[-2], state[-1]]))
            full_action = np.asarray([abr_action, place_action])
            if place_action==0:
                full_action[0] = max(0, abr_action - 2)
            # if _ep%5 == 0:
            # 	# validate every 5th episode
            # 	action = trainer.get_exploitation_action(state)
            # else:
            # 	# get action based on observation, use exploration policy here
            # 	action = trainer.get_exploration_action(state)

            new_observation, reward, done, info = env.step(full_action)
            print(reward, info)
            self.add_reward(reward)
            self.maybe_do_sr(past_chunk_throughput=info['throughput'], new_bitrate=info['new_bitrate'])


            observation = new_observation

            if done:
                break
        self.plot_all()

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
        plt.savefig('./model/results/' + msg + '.png')

    def resize(self, val_list, a):
        new_list = []
        for idx, e in enumerate(val_list):
            if idx % a == 0:
                new_list.append(e)
        return new_list

if __name__ == "__main__":
    client = Client(abr_trainer, place_trainer)
    client.run(10000)
