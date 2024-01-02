import argparse
import os
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

import logging
import learn2learn as l2l
from torch import autograd
from variant_vmaf.utils.replay_memory import ReplayMemory
from model_ac_torch import Actor, Critic, PlaceActor, PlaceCritic

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

class MAMLPPO:
    def __init__(
        self,
        args,
        a_dim,
        a_state_dim,
        p_dim,
        p_state_dim,
        seed=42,
        device=None,
        name="MAMLPPO",
        tensorboard_log="./logs",
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.a_dim = a_dim
        self.p_dim = p_dim
        self.a_state_dim = a_state_dim
        self.p_state_dim = p_state_dim

        self.gamma = args.gae_gamma
        self.tau = args.gae_lambda
        self.adapt_lr = args.lr_adapt
        self.meta_lr = args.lr_meta
        self.adapt_steps = args.adapt_steps
        self.policy_clip = args.clip
        self.ppo_steps = args.ppo_ups
        self.ent_coeff = args.ent_coeff
        self.ent_decay = args.ent_decay
        self.dual_adv_w = args.dual_adv_w

        # ---- initialize models ----
        self.abr_actor = Actor(a_dim).type(dtype)
        self.abr_critic = Critic(a_dim).type(dtype)
        self.place_actor = PlaceActor(p_dim).type(dtype)
        self.place_critic = PlaceCritic(p_dim).type(dtype)

        # ---- set optimizer for actor and critic
        self.abr_optimizer = torch.optim.Adam(self.abr_actor.parameters(), self.meta_lr)
        self.abr_optimizer_critic = torch.optim.Adam(self.abr_critic.parameters(), self.meta_lr)
        self.place_optimizer = torch.optim.Adam(self.place_actor.parameters(), self.meta_lr)
        self.place_optimizer_critic = torch.optim.Adam(self.place_critic.parameters(), self.meta_lr)

    def save(self, path="./", epoch=0):
        torch.save(self.abr_critic.state_dict(), path + "/abr_critic" + str(epoch) + ".pt")
        torch.save(self.abr_actor.state_dict(), path + "/abr_actor" + str(epoch) + ".pt")
        torch.save(self.place_critic.state_dict(), path + "/place_critic" + str(epoch) + ".pt")
        torch.save(self.place_actor.state_dict(), path + "/place_actor" + str(epoch) + ".pt")

    def load(self, abr_epoch = 0, place_epoch = 0, path="./Results/checkpoints"):
        if abr_epoch>0 :
            self.abr_critic.load_state_dict(torch.load(path + "/abr_critic"+str(abr_epoch)+".pt"))
            self.abr_actor.load_state_dict(torch.load(path + "/abr_actor"+str(abr_epoch)+".pt"))
        if place_epoch>0 :
            self.place_critic.load_state_dict(torch.load(path + "/place_critic"+str(place_epoch)+".pt"))
            self.place_actor.load_state_dict(torch.load(path + "/place_actor"+str(place_epoch)+".pt"))

    def ent_coeff_decay(self):
        self.ent_coeff = self.ent_decay * self.ent_coeff

    def compute_adv(self, done, value, values, rewards):
        "Calculates the advantages and returns for a trajectories."
        gamma, gae_param = self.gamma, self.tau
        advantages = []
        returns = []

        # ==================== finish one episode ===================
        # one last step
        R = torch.zeros(1, 1)
        if done == False:
            v = value.cpu()
            R = v.data

        values.append(Variable(R).type(dtype))
        R = Variable(R).type(dtype)
        A = Variable(torch.zeros(1, 1)).type(dtype)

        rewards_ = np.array(rewards)
        rewards_ = torch.from_numpy(rewards_).type(dtype)

        for i in reversed(range(len(rewards))):
            td = (
                rewards_[i].data
                + gamma * values[i + 1].data[0, 0]
                - values[i].data[0, 0]
            )
            A = td + gamma * gae_param * A
            advantages.insert(0, A)
            # R = A + values[i]
            R = gamma * R + rewards_[i].data
            returns.insert(0, R)

        return advantages, returns

    def collect_steps(self, abr_actor, place_actor, env, n_episodes):
        done = True
        abr_states = []
        abr_actions = []
        abr_rewards = []
        abr_values = []
        abr_memory = ReplayMemory(500)
        place_states = []
        place_actions = []
        place_rewards = []
        place_values = []
        place_memory = ReplayMemory(500)

        explo_bit_rate = 1
        explo_place_action = 1
        state = env.reset()
        state = torch.from_numpy(np.array([state])).type(dtype)
        for _ in range(n_episodes):

            abr_state, place_state = state[:, :self.a_state_dim], state[:, :]

            with torch.no_grad():
                abr_prob = abr_actor.forward(abr_state)
                abr_value = self.abr_critic(abr_state)
                abr_action = abr_prob.multinomial(num_samples=1)

                place_prob = place_actor.forward(place_state)
                place_value = self.place_critic(place_state)
                place_action = place_prob.multinomial(num_samples=1)

            # value, action = agent.explore(ob_, state_)
            explo_bit_rate = int(abr_action.squeeze().cpu().numpy())
            explo_place_action = int(place_action.squeeze().cpu().numpy())

            if explo_place_action == 0:
                trans_bit_rate = max(-1, explo_bit_rate - 2)
            else:
                trans_bit_rate = explo_bit_rate

            next_state, abr_reward, place_reward, done = env.step(trans_bit_rate, explo_place_action)

            # record the current state, observation and action
            abr_states.append(abr_state)
            abr_actions.append(abr_action)
            abr_values.append(abr_value)
            place_states.append(place_state)
            place_actions.append(place_action)
            place_values.append(place_value)
            abr_rewards.append(abr_reward)
            place_rewards.append(place_reward)

            state = next_state

            # value, action = actor.explore(ob_, state_, action_mask_)
            if done:
                break

        # compute returns and GAE(lambda) advantages:
        if len(abr_states) != len(abr_rewards):
            if len(abr_states) + 1 == len(abr_rewards):
                abr_rewards = abr_rewards[1:]
            else:
                print("error in length of states!")
        if len(place_states) != len(place_rewards):
            if len(place_states) + 1 == len(place_rewards):
                place_rewards = place_rewards[1:]
            else:
                print("error in length of states!")
        abr_advantages, abr_returns = self.compute_adv(done, abr_value, abr_values, abr_rewards)
        abr_replay = [abr_states, abr_actions, abr_returns, abr_advantages]
        abr_memory.push(abr_replay)
        place_advantages, place_returns = self.compute_adv(done, place_value, place_values, place_rewards)
        place_replay = [place_states, place_actions, place_returns, place_advantages]
        place_memory.push(place_replay)

        # ----- update critic ----
        abr_batch_states, _, abr_batch_returns, _ = abr_memory.sample_cuda(abr_memory.return_size())
        abr_v_pre = self.abr_critic(abr_batch_states)
        # value loss
        abr_vfloss1 = (abr_v_pre - abr_batch_returns.type(dtype)) ** 2
        abr_loss_value = 0.5 * torch.mean(abr_vfloss1)
        self.abr_optimizer_critic.zero_grad()
        # loss_actor.backward(retain_graph=False)
        abr_loss_value.backward()
        # clip_grad_norm_(self.critic.parameters(), max_norm = 3., norm_type = 2)
        self.abr_optimizer_critic.step()

        place_batch_states, _, place_batch_returns, _ = place_memory.sample_cuda(place_memory.return_size())
        place_v_pre = self.place_critic(place_batch_states)
        # value loss
        place_vfloss1 = (place_v_pre - place_batch_returns.type(dtype)) ** 2
        place_loss_value = 0.5 * torch.mean(place_vfloss1)
        self.place_optimizer_critic.zero_grad()
        # loss_actor.backward(retain_graph=False)
        place_loss_value.backward()
        # clip_grad_norm_(self.critic.parameters(), max_norm = 3., norm_type = 2)
        self.place_optimizer_critic.step()

        del abr_memory
        del place_memory
        return abr_replay, place_replay

    def dual_ppo_loss(self, train_episodes, old_policy, new_policy):
        memory = ReplayMemory(500)
        memory.push(train_episodes)
        batch_states, batch_actions, _, batch_advantages = memory.sample_cuda(
            memory.return_size()
        )
        # old_prob
        probs_old = old_policy(batch_states).detach()
        prob_value_old = torch.gather(
            probs_old, dim=1, index=batch_actions.type(dlongtype)
        ).detach()
        # new prob
        probs = new_policy(batch_states)
        prob_value = torch.gather(probs, dim=1, index=batch_actions.type(dlongtype))

        # ratio
        ratio = prob_value / (1e-6 + prob_value_old)

        # clip loss
        surr1 = ratio * batch_advantages.type(
            dtype
        )  # surrogate from conservative policy iteration
        surr2 = ratio.clamp(
            1 - self.policy_clip, 1 + self.policy_clip
        ) * batch_advantages.type(dtype)
        loss_clip_ = torch.min(surr1, surr2)
        loss_clip_dual = torch.where(
            torch.lt(batch_advantages.type(dtype), 0.0),
            torch.max(loss_clip_, self.dual_adv_w * batch_advantages.type(dtype)),
            loss_clip_,
        )
        loss_clip_actor = -torch.mean(loss_clip_dual)

        # entropy loss
        ent_latent = self.ent_coeff * torch.mean(probs * torch.log(probs + 1e-5))
        del memory
        return loss_clip_actor, ent_latent
        # return loss_clip_actor

    def maml_a2c_loss(self, memory, actor):
        # obtain policy loss
        batch_states, batch_actions, _, batch_advantages = memory.sample_cuda(
            memory.return_size()
        )
        probs = actor(batch_states)
        prob_value = torch.gather(probs, dim=1, index=batch_actions.type(dlongtype))
        loss = -torch.mean(prob_value * batch_advantages.type(dtype))
        ent = self.ent_coeff * torch.mean(probs * torch.log(probs + 1e-5))
        return loss, ent

    def fast_adapt(self, clone, train_episodes, first_order=False):
        memory = ReplayMemory(500)
        memory.push(train_episodes)
        second_order = not first_order
        loss_a, loss_e = self.maml_a2c_loss(memory, clone)
        loss = loss_a + loss_e
        gradients = autograd.grad(
            loss,
            clone.parameters(),
            retain_graph=second_order,
            create_graph=second_order,
        )

        del memory
        return (
            loss_a,
            loss_e,
            l2l.algorithms.maml.maml_update(clone, self.adapt_lr, gradients),
        )

    def meta_loss(self, iteration_replays, iteration_policies, policy):
        mean_loss_a = 0.0
        mean_loss_e = 0.0
        for _ in range(len(iteration_replays)):
            task_replays = iteration_replays[_]
            old_policy = iteration_policies[_]
            train_replays = task_replays[:-1]
            valid_episodes = task_replays[-1]
            new_policy = l2l.clone_module(policy)

            # Fast Adapt
            for _ in range(len(train_replays)):
                train_episodes = train_replays[_]
                _, _, new_policy = self.fast_adapt(
                    new_policy, train_episodes, first_order=False
                )

            # Compute Surrogate Loss
            loss_a, loss_e = self.dual_ppo_loss(valid_episodes, old_policy, new_policy)
            # surr_loss = loss_a + loss_e
            mean_loss_a += loss_a
            mean_loss_e += loss_e
        mean_loss_a /= len(iteration_replays)
        mean_loss_e /= len(iteration_replays)
        return mean_loss_a, mean_loss_e

    def meta_optimize(self, iteration_replays, iteration_policies, actor):
        for _ in range(self.ppo_steps):
            loss_a, loss_e = self.meta_loss(
                iteration_replays, iteration_policies, actor
            )

            self.abr_optimizer.zero_grad()
            loss = loss_a + loss_e
            loss.backward()
            self.abr_optimizer.step()

            ## --------- update ---------
            # this part will take higher order gradients through the inner loop:
            # grads = torch.autograd.grad(loss, self.actor.parameters())
            # grads = parameters_to_vector(grads)
            # old_params = parameters_to_vector(self.actor.parameters())
            # vector_to_parameters(old_params -  self.meta_lr * grads, self.actor.parameters())

        return loss_a, loss_e
