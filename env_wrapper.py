import numpy as np
import os
import torch
import torch.optim as optim
from torch.autograd import Variable

DEFAULT_QUALITY = 1
M_IN_K = 1000.0
BUFFER_NORM_FACTOR = 10.0
# DB_NORM_FACTOR = 100.0

VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
S_INFO = 8
S_LEN = 8
A_DIM = 6
REBUF_PENALTY_LOG = 2.66  # 1 sec rebuffering -> 3 Mbps
REBUF_PENALTY_LIN = 4.3
SMOOTH_PENALTY = 1

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


class VirtualPlayer:
    def __init__(self, args, env, log_file):
        self.env = env
        self.args = args
        self.task_list = env.task_list

        ## get the information of virtual players (personality)
        # s_info, s_len, c_len, total_chunk_num, bitrate_versions, \
        #     quality_penalty, rebuffer_penalty, smooth_penalty_p, smooth_penalty_n \
        #         = env.get_env_info()
        # Video information
        self.s_info, self.s_len, self.quality_p, self.smooth_p = (
            S_INFO,
            S_LEN,
            1,
            SMOOTH_PENALTY,
        )
        self.bitrate_versions = VIDEO_BIT_RATE
        self.rebuff_p = REBUF_PENALTY_LOG
        self.br_dim = len(self.bitrate_versions)

        # QoE reward scaling
        self.scaling_lb = -4 * self.rebuff_p
        self.scaling_r = self.rebuff_p

        # define the state for rl agent
        self.state = np.zeros((self.s_info, self.s_len))

        # information of emulating the video playing
        self.last_bit_rate = DEFAULT_QUALITY
        self.time_stamp = 0.0
        self.end_flag = True
        # log files, recoding the video playing
        self.log_file = log_file

        # information of action mask
        self.past_errors = []
        self.past_bandwidth_ests = []
        self.video_chunk_sizes = env.get_video_size()
        self.total_chunk_num = len(self.video_chunk_sizes[0]) - 1
        self.video_chunk_remain = self.total_chunk_num

    def step(self, trans_bitrate, sr_place):
        # execute a step forward
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
        ) = self.env.get_video_chunk(trans_bitrate, sr_place)

        trans_delay = delay - sr_delay

        if trans_bitrate<0:
            trans_bitrate = 0
            new_bitrate = 0
        elif sr_place==1:
            new_bitrate = min(trans_bitrate + 2, len(VIDEO_BIT_RATE) - 1)
        else:
            new_bitrate = trans_bitrate

        # compute reward of Quality of experience
        reward = (
                10 * self.bitrate_versions[trans_bitrate] / M_IN_K
                - self.rebuff_p * rebuf
                - self.smooth_p * np.abs(self.bitrate_versions[trans_bitrate] - self.bitrate_versions[self.last_bit_rate]) / M_IN_K
        )
        # if self.time_stamp < 0.1:
        #     reward += self.rebuff_p * rebuf

        lazy_reward = 0
        if sleep_time > 0 and new_bitrate < len(VIDEO_BIT_RATE) - 1:
            lazy_reward = -sleep_time * (len(VIDEO_BIT_RATE) - 1 - new_bitrate)
        reward += lazy_reward

        rew_ = float(max(reward, self.scaling_lb) / self.scaling_r)
        # reward_norm = self.reward_filter(rew_)
        reward_norm = reward

        # compute and record the reward of current chunk
        self.time_stamp += delay  # in ms
        # self.time_stamp += sleep_time  # in ms

        self.video_chunk_remain = video_chunk_remain

        self.last_bit_rate = trans_bitrate

        # -------------- logging -----------------
        # log time_stamp, trans_bitrate, buffer_size, reward
        self.log_file.write(
            str(self.time_stamp / M_IN_K)[:4] + "\t"
            + str(VIDEO_BIT_RATE[trans_bitrate] / M_IN_K) + "\t"
            + str(VIDEO_BIT_RATE[new_bitrate] / M_IN_K) + "\t"
            + str(buffer_size)[:4] + "\t"
            + str(rebuf)[:4] + "\t"
            + str(video_chunk_size) + "\t"
            + str(delay)[:4] + "\t"
            + str(self.state[6, -1])[:4] + "\t"
            + str(self.state[7, -1])[:4] + "\t"
            + str(sr_place) + "\t"
            + str(lazy_reward)[:4] + "\t"
            + str(place_reward)[:4] + "\t"
            + str(reward)[:4]
            + "\n"
        )
        self.log_file.flush()

        ## dequeue history record
        self.state = np.roll(self.state, -1, axis=1)

        # this should be S_INFO number of terms 6个状态 (比特率大小，缓冲区大小，剩余片数量，时延，下个片的大小，吞吐量)
        self.state[0, -1] = self.bitrate_versions[trans_bitrate] / float(
            self.bitrate_versions[-1]
        )  # last quality
        self.state[1, -1] = float(buffer_size / BUFFER_NORM_FACTOR)  # 10 sec
        self.state[2, -1] = (
                float(video_chunk_size) / float(trans_delay) / M_IN_K
        )  # kilo byte / ms
        self.state[3, -1] = float(trans_delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        self.state[4, : self.br_dim] = (
            np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K / 10
        )  # mega byte
        self.state[5, -1] = np.minimum(
            video_chunk_remain, self.total_chunk_num
        ) / float(self.total_chunk_num)
        self.state[6, -1] = float(edge_k)
        self.state[7, -1] = client_k

        next_state = np.array([self.state])
        next_state = torch.from_numpy(next_state).type(dtype)

        self.end_flag = end_of_video
        if self.end_flag:
            self.reset_play()
        return next_state, reward_norm, place_reward, end_of_video

    def set_task(self, idx):
        self.env.set_task(idx)

    def reset(self):
        edge_k, client_k = self.env.reset()
        self.state[6, -1] = float(edge_k)
        self.state[7, -1] = client_k
        return self.state

    def reset_play(self):
        self.state = np.zeros((self.s_info, self.s_len))

        self.last_bit_rate = DEFAULT_QUALITY
        self.video_chunk_remain = self.total_chunk_num
        self.time_stamp = 0.0

        self.past_bandwidth_ests = []
        self.past_errors = []

        self.log_file.write("\n")
        self.log_file.flush()

    def clean_file_cache(self, file_name, max_file_size=4.096e7):
        file_size = os.stat(file_name).st_size
        if file_size > max_file_size:
            self.log_file.seek(0)
            self.log_file.truncate()
