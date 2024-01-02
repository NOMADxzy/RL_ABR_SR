import time
from collections import deque
import numpy as np
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F

from model_ac_torch import Actor, Critic, PlaceActor, PlaceCritic
from env_wrapper import VirtualPlayer
import env as env_valid
import load_trace

RANDOM_SEED = 42
S_INFO = 8
S_LEN = 8
A_DIM = 6
P_DIM = 2
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY_LOG = 2.66  # 1 sec rebuffering -> 3 Mbps
REBUF_PENALTY_LIN = 4.3
SMOOTH_PENALTY = 0.2
DEFAULT_QUALITY = 1
TEST_TRACES_VALID = "./cooked_test_traces/"
SUMMARY_DIR = "./Results/sim"
# LOG_FILE = "./Results/sim/ppo/log"
# TEST_LOG_FOLDER = "./Results/sim/ppo/test_results/"

# LOG_FILE_VALID = "./Results/sim/ppo/test_results/log_valid_ppo"
# TEST_LOG_FOLDER_VALID = "./Results/sim/ppo/test_results/"

# LOG_FILE_TEST = './Results/test/BC/log_hybrid_ppo'
# SUMMARY_DIR = './Results/test/BC/'

# Log_path = "./Results/sim/ppo"

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
dshorttype = torch.cuda.ShortTensor if torch.cuda.is_available() else torch.ShortTensor


def evaluation(
    abr_model, place_model, log_path_ini, net_env, all_file_name, detail_log=True, q_lin=False
):
    # all_file_name = net_env.get_file_name()

    state = init_state(net_env)
    # reward_sum = 0
    done = True
    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY
    # model.load_state_dict(model.state_dict())
    log_path = log_path_ini + "_" + all_file_name[net_env.trace_idx]
    log_file = open(log_path, "w")

    time_stamp = 0
    for video_count in tqdm(range(len(all_file_name))):
        while True:

            with torch.no_grad():
                abr_prob = abr_model(state[:].unsqueeze(0).type(dtype))
                place_prob = place_model(state[:].unsqueeze(0).type(dtype))
            abr_action = abr_prob.multinomial(num_samples=1).detach()
            place_action = place_prob.multinomial(num_samples=1).detach()
            bit_rate = int(abr_action.squeeze().cpu().numpy())
            place_choice = int(place_action.squeeze().cpu().numpy())
            if place_choice==0:
                source_bitrate = max(bit_rate - 2, -1)
                new_bitrate = bit_rate
            else:
                source_bitrate = bit_rate
                new_bitrate = min(len(VIDEO_BIT_RATE)-1, bit_rate + 2)

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
            ) = net_env.get_video_chunk(
                source_bitrate, place_choice
            )  ## sample in the environment of virtual player
            if source_bitrate<0:
                source_bitrate = 0
                new_bitrate = 0
            trans_delay = delay - sr_delay

            time_stamp += delay  # in ms
            # time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty - smooth penalty
            # compute reward of Quality of experience
            # -- lin scale reward --
            reward = (
                    10 * VIDEO_BIT_RATE[bit_rate] / M_IN_K
                    - REBUF_PENALTY_LIN * rebuf
                    - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
            )
            lazy_reward = 0
            if sleep_time > 0 and new_bitrate < len(VIDEO_BIT_RATE) - 1:
                lazy_reward = -sleep_time * (len(VIDEO_BIT_RATE) - 1 - new_bitrate)
            reward += lazy_reward

            last_bit_rate = bit_rate

            log_file.write(
                str(time_stamp / M_IN_K)[:4] + "\t"
                + str(VIDEO_BIT_RATE[bit_rate] / M_IN_K) + "\t"
                + str(VIDEO_BIT_RATE[new_bitrate] / M_IN_K) + "\t"
                + str(buffer_size)[:4] + "\t"
                + str(rebuf)[:4] + "\t"
                + str(video_chunk_size) + "\t"
                + str(delay)[:4] + "\t"
                + str(int(state[6, -1]))[:4] + "\t"
                + str(int(state[7, -1]))[:4] + "\t"
                + str(place_choice) + "\t"
                + str(lazy_reward)[:4] + "\t"
                + str(place_reward)[:4] + "\t"
                + str(reward)[:4]
                + "\n"
            )
            log_file.flush()

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms、
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(
                np.max(VIDEO_BIT_RATE)
            )  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = (
                    float(video_chunk_size) / float(trans_delay) / M_IN_K
            )  # kilo byte / ms
            state[3, -1] = float(trans_delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = (
                    np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K / 10
            )  # mega byte
            state[5, -1] = np.minimum(
                video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP
            ) / float(CHUNK_TIL_VIDEO_END_CAP)
            state[6, -1] = float(edge_k)
            state[7, -1] = client_k

            state = torch.from_numpy(state)

            if end_of_video:
                state = init_state(net_env)
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY

                log_file.write("\n")
                log_file.close()
                time_stamp = 0

                if video_count + 1 >= len(all_file_name):
                    break
                else:
                    log_path = log_path_ini + "_" + all_file_name[net_env.trace_idx]
                    log_file = open(log_path, "w")
                    break

def init_state(net_env):
    state = np.zeros((S_INFO, S_LEN))
    state = torch.from_numpy(state)

    # if net_env.edge_k==0:
    #
    edge_k, client_k = net_env.reset()
    state[6, -1] = float(edge_k)
    state[7, -1] = client_k
    return state


def valid(args, abr_shared_model, place_shared_model, epoch, log_file, test_log_path):
    res_folder = os.path.join(*[test_log_path, "test_results"])
    os.system("rm -r " + res_folder)
    os.system("mkdir " + res_folder)

    abr_model = Actor(A_DIM).type(dtype)
    abr_model.eval()
    abr_model.load_state_dict(abr_shared_model.state_dict())
    place_model = PlaceActor(P_DIM).type(dtype)
    place_model.eval()
    place_model.load_state_dict(place_shared_model.state_dict())

    log_path_ini = f"{res_folder}/log_valid_maml"
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(
        TEST_TRACES_VALID
    )
    env = env_valid.Environment(
        all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw
    )
    evaluation(abr_model, place_model, log_path_ini, env, all_file_names, False, q_lin=args.lin)

    abr_rewards = []
    place_rewards = []
    test_log_files = os.listdir(res_folder)
    for test_log_file in test_log_files:
        abr_reward = []
        place_reward = []
        with open(f"{res_folder}/{test_log_file}", "rb") as f:
            for line in f:
                parse = line.split()
                try:
                    abr_reward.append(float(parse[-1]))
                    place_reward.append(float(parse[-2]))
                except IndexError:
                    break
        abr_rewards.append(np.mean(abr_reward[1:]))
        place_rewards.append(np.mean(place_reward[1:]))

    abr_rewards = np.array(abr_rewards)

    abr_rewards_min = np.min(abr_rewards)
    abr_rewards_5per = np.percentile(abr_rewards, 5)
    abr_rewards_mean = np.mean(abr_rewards)
    abr_rewards_median = np.percentile(abr_rewards, 50)
    abr_rewards_95per = np.percentile(abr_rewards, 95)
    abr_rewards_max = np.max(abr_rewards)

    place_rewards_min = np.min(place_rewards)
    place_rewards_5per = np.percentile(place_rewards, 5)
    place_rewards_mean = np.mean(place_rewards)
    place_rewards_median = np.percentile(place_rewards, 50)
    place_rewards_95per = np.percentile(place_rewards, 95)
    place_rewards_max = np.max(place_rewards)


    log_file.write(
        str(int(epoch))
        + "\t"
        + str(abr_rewards_min)
        + "\t"
        + str(abr_rewards_5per)
        + "\t"
        + str(abr_rewards_mean)
        + "\t"
        + str(abr_rewards_median)
        + "\t"
        + str(abr_rewards_95per)
        + "\t"
        + str(abr_rewards_max)
        + "\n"
    )
    log_file.write(
        str(int(epoch))
        + "\t"
        + str(place_rewards_min)
        + "\t"
        + str(place_rewards_5per)
        + "\t"
        + str(place_rewards_mean)
        + "\t"
        + str(place_rewards_median)
        + "\t"
        + str(place_rewards_95per)
        + "\t"
        + str(place_rewards_max)
        + "\n"
    )
    log_file.flush()
    return abr_rewards_mean, place_rewards_mean

