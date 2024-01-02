"""
In this file, maml_ppo algorithm is adopted to fine-tune the policy of rate adaptation, gae advantage function and multi-step return are used to calculate the gradients.

Add the reward normalization, using vmaf quality metric

Designed by kannw_1230@sjtu.edu.cn

"""

import os
import numpy as np
import random

import torch, time
import torch.optim as optim
from torch.autograd import Variable
import logging
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from maml_ppo import MAMLPPO
from test_maml_torch import valid
import env
import load_trace
from env_wrapper import VirtualPlayer
from variant_vmaf.utils.helper import check_folder

RANDOM_SEED = 28
DEFAULT_QUALITY = int(1)  # default video quality without agent

SUMMARY_DIR = "./Results/sim"

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

ABR_STATE_DIM = 6
PLACE_STATE_DIM = 3


def train_maml_ppo(args):
    add_str = "maml"
    summary_dir = os.path.join(*[SUMMARY_DIR, add_str])
    check_folder(summary_dir)
    log_str = "lin" if args.lin else "log"
    summary_dir = os.path.join(*[summary_dir, log_str])
    check_folder(summary_dir)
    ts = time.strftime("%b%d-%H:%M:%S", time.gmtime())
    log_file_path = os.path.join(*[summary_dir, ts])
    check_folder(log_file_path)
    log_file_name = log_file_path + "/log"
    writer = SummaryWriter(log_file_path)
    abr_mean_value = 0
    place_mean_value = 0

    # define the parameters of ABR environments
    # _, _, _, _, bitrate_versions, _, _, _, _ = train_env.get_env_info()
    br_dim = 6
    place_dim = 2
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace()
    train_env = env.Environment(
        all_cooked_time=all_cooked_time,
        all_cooked_bw=all_cooked_bw,
        all_file_names=all_file_names,
        a2br=True,
    )

    torch.manual_seed(RANDOM_SEED)

    with open(log_file_name + "_record", "w") as log_file, open(
        log_file_name + "_test", "w"
    ) as test_log_file:
        abr_update,place_update = True,True
        agent = MAMLPPO(args, br_dim, ABR_STATE_DIM, place_dim, PLACE_STATE_DIM)
        agent.load(abr_epoch=18420)
        if args.init:
            init_ckpt_path = os.path.join(
                *[summary_dir, "init_ckpt"]
            )  # Notice: ensure the correct model!
            agent.load(init_ckpt_path)

        steps_in_episode = args.ro_len

        vp_env = VirtualPlayer(args, train_env, log_file)
        task_num = len(vp_env.env.task_list)

        # while True:
        for epoch in range(int(1e5)):
            # agent.model_eval()
            # vp_env.reset_reward()

            # ---------- limit the file size -----------
            vp_env.clean_file_cache(log_file_name + "_record")

            abr_iteration_replays = []
            abr_iteration_policies = []
            place_iteration_replays = []
            place_iteration_policies = []

            for _ in range(task_num):
                cloned_abr_actor = deepcopy(agent.abr_actor)
                cloned_place_actor = deepcopy(agent.place_actor)
                vp_env.env.set_task(_)
                abr_task_replay = []
                place_task_replay = []

                # Fast Adapt
                for _ in range(args.adapt_steps): # 4*47*6*8 train_episodes 47个chunk、6 + 2个特征、8个历史长度
                    abr_train_episodes, place_train_episodes = agent.collect_steps(
                        cloned_abr_actor, cloned_place_actor, vp_env, n_episodes=steps_in_episode
                    ) # abr_train_episodes : [abr_states, abr_actions, abr_returns, abr_advantages]
                    _, _, cloned_abr_actor = agent.fast_adapt(
                        cloned_abr_actor, abr_train_episodes, first_order=True
                    )
                    _, _, cloned_place_actor = agent.fast_adapt(
                        cloned_place_actor, place_train_episodes, first_order=True
                    )
                    abr_task_replay.append(abr_train_episodes)
                    place_task_replay.append(place_train_episodes)

                # Compute Validation Loss
                abr_valid_episodes, place_valid_episodes = agent.collect_steps(
                    cloned_abr_actor, cloned_place_actor, vp_env, n_episodes=steps_in_episode
                )

                abr_task_replay.append(abr_valid_episodes)
                place_task_replay.append(place_valid_episodes)

                abr_iteration_replays.append(abr_task_replay)
                abr_iteration_policies.append(cloned_abr_actor)
                place_iteration_replays.append(place_task_replay)
                place_iteration_policies.append(cloned_place_actor)

            # training the models
            if abr_update:
                abr_policy_loss_, abr_entropy_loss_ = agent.meta_optimize(
                    abr_iteration_replays, abr_iteration_policies, agent.abr_actor
                )
                writer.add_scalar("Avg_Abr_Policy_loss", abr_policy_loss_, epoch)
                writer.add_scalar("Avg_Abr_Entropy_loss", abr_entropy_loss_, epoch)
            if place_update:
                place_policy_loss_, place_entropy_loss_ = agent.meta_optimize(
                    place_iteration_replays, place_iteration_policies, agent.place_actor
                )
                writer.add_scalar("Avg_Place_Policy_loss", place_policy_loss_, epoch)
                writer.add_scalar("Avg_Place_Entropy_loss", place_entropy_loss_, epoch)



            if epoch % 20 == 0 and epoch > 0:
                abr_model = agent.abr_actor
                place_model = agent.place_actor
                abr_mean_value, place_mean_value = valid(
                    args, abr_model, place_model, epoch, test_log_file, log_file_path
                )

                save_folder = os.path.join(*[log_file_path, "checkpoints"])
                check_folder(save_folder)
                agent.save(save_folder, epoch)
            writer.add_scalar("Avg_Abr_Return", abr_mean_value, epoch)
            writer.add_scalar("Avg_Place_Return", place_mean_value, epoch)
            writer.flush()

            if epoch % int(100) == 0 and epoch > 0:
                agent.ent_coeff_decay()

        writer.close()
