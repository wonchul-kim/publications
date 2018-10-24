from __future__ import print_function
import os
import time
from collections import deque
import pickle
import random
from ddpg import DDPG
from util import mpi_mean, mpi_std, mpi_max, mpi_sum
import tf_util as U

import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI
import matplotlib.pyplot as plt
import glob

from imitation_learning import imitationLearning

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

def get_reward(y, dy, ddy, target, targetT, cur_step_num, waypoint):
    if cur_step_num == waypoint[0] - 1:
        term_w = -1*(y - waypoint[1])**2
    else:
        term_w = 0
    
    if cur_step_num == targetT:
        term1 = -1*(y - target)**2
    else:
        term1 = 0
        
    term2 = -1*sum(dy**2)
    term3 = -1*sum(ddy**2)
    
    return term1, term2, term3, term_w

def check_done(y, target, targetT, cur_step_num):
    if cur_step_num == targetT and (y - target)**2 < 1e-3:
        return 1
    else:
        return 0

def check_sub_done(y, waypoint, cur_step_num):
    if cur_step_num == waypoint[0] - 1 and (y - waypoint[1])**2 < 1e-3:
        return 1
    else:
        return 0

def train(dmp, path, imitation_flag, observation_shape, action_shape, nb_epochs, nb_epoch_cycles, reward_scale, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, imitation_lr, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, batch_size, memory, 
    tau=0.01, param_noise_adaption_interval=50):
    rank = MPI.COMM_WORLD.Get_rank()

    agent = DDPG(actor, critic, memory, observation_shape, action_shape, \
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)

    # # Set up logging stuff only for a single worker.
    # saver = tf.train.Saver()

    step = 0
    episode = 0
    # episode_rewards_history = deque(maxlen=100)

    sys_traj_cur_y = []
    sys_traj_canocial = []

    waypoint = [40, 0.35]

    figs_path = './results/figs/'
    # delete the previous figs in results folder
    figs = glob.glob(figs_path + '*')
    for ff in figs:
        os.remove(ff)

    with U.single_threaded_session() as sess:
        # Prepare everything.
        print(">>> Initialized DDPG .....")
        agent.initialize(sess)
        sess.graph.finalize()

        print('>>> Imitation starts .....')
        imitation = imitationLearning(dmp, path, agent)
        data_X, train_X, mean_X, std_X,data_Y, train_Y, mean_Y, std_Y = imitation.get_data()
        if imitation_flag is True:
            f_target = imitation.run()
            imitation.eval(f_target, True)
        else:
            imitation.restore_model()
            
        print('\n>>> DDPG starts ..........................')
        for epoch in range(nb_epochs):
            for cycle in range(nb_epoch_cycles): # Episode
                # Perform rollouts.
                agent.reset()            
                dmp.reset()
                obs = (dmp.cs.step() - mean_X)/std_X
                obs = np.array([obs])
                done = False
                episode_reward = 0.
                episode_step = 0
                episodes = 0
                t = 0

                # start_time = time.time()

                # epoch_episode_rewards = []
                # epoch_episode_steps = []
                # epoch_start_time = time.time()
                epoch_actions = []
                epoch_qs = []
                epoch_episodes = 0


                cur_y_traj = []
                canonical_traj = []

                    
                for t_rollout in range(dmp.n_step): # step
                    # Predict next action.
                    canonical_traj.append(obs)
                    assert obs.shape == observation_shape
                    action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    force = action*std_Y + mean_Y
                    cur_y, cur_dy, cur_ddy = dmp.ddpg_step(force, path)
                    r1, r2, r3, r_w = get_reward(cur_y, cur_dy, cur_ddy, path[-1], dmp.n_step - 1, t_rollout, waypoint)
                    r = 100*r1 + 0.01*r2 + 0.0*r3 + 100*r_w

                    new_obs = np.array([(dmp.cs.x - mean_X)/std_X])

                    done = check_done(cur_y, path[-1], dmp.n_step - 1, t_rollout)
                    sub_done = check_sub_done(cur_y, waypoint, t_rollout)
                    if sub_done:
                        print('\nooooooo sub goal !!!!!!! ooooooo')
                    t += 1

                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                       
                    agent.store_transition(obs, action, r, new_obs, done)
                    obs = new_obs
                    cur_y_traj.append(cur_y[0])

                    print('\rEpoch: {} | Ep: {} | Step: {} | R: {} | Y: {} | CS: {} | G: {} '.format(\
                                        epoch, cycle, t_rollout,episode_reward, cur_y[0], obs*std_X + mean_X, dmp.goal[0]), end='')
                    # print('\rEp: {} | Step: {} | R: {} | Y: {} | CS: {}'.format(\
                    #                     cycle, t_rollout,episode_reward, cur_y[0], obs*std_X + mean_X), end='')

                    
                # Episode done.
                # epoch_episode_rewards.append(episode_reward)
                # episode_rewards_history.append(episode_reward)
                # epoch_episode_steps.append(episode_step)
                episode_reward = 0.
                episode_step = 0
                epoch_episodes += 1
                episodes += 1

                # agent.reset()

                if done:
                    print('\nSuccess >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
                else:
                    print('\nxxx fail xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n')

                        
                # Train.
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                for t_train in range(nb_train_steps):
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and t % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    cl, al = agent.train()
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()
            
                sys_traj_canocial.append(canonical_traj)
                sys_traj_cur_y.append(cur_y_traj)
                plt.clf()
                plt.plot(sys_traj_cur_y[cycle], 'b', lw=2)
                plt.plot(path, 'r--', lw=3)
                plt.scatter(dmp.n_step - 1, path[-1], s=150)
                plt.scatter(waypoint[0] - 1, waypoint[1], s=150)
                plt.legend(['dmp', 'true'])
                plt.title('%i epoch %i episode' %(epoch, cycle))
                plt.savefig(figs_path + '%i_%i.png' %(epoch, cycle))
                # plt.title('%i episode' %(cycle))
                # plt.savefig(figs_path + '%i.png' %(cycle))
                # plt.pause(0.01)

                # Log stats.
                # epoch_train_duration = time.time() - epoch_start_time
                # duration = time.time() - start_time
                # stats = agent.get_stats()
                # combined_stats = {}
                # for key in sorted(stats.keys()):
                #     combined_stats[key] = mpi_mean(stats[key])

                # # Rollout statistics.
                # combined_stats['rollout/return'] = mpi_mean(epoch_episode_rewards)
                # combined_stats['rollout/return_history'] = mpi_mean(np.mean(episode_rewards_history))
                # combined_stats['rollout/episode_steps'] = mpi_mean(epoch_episode_steps)
                # combined_stats['rollout/episodes'] = mpi_sum(epoch_episodes)
                # combined_stats['rollout/actions_mean'] = mpi_mean(epoch_actions)
                # combined_stats['rollout/actions_std'] = mpi_std(epoch_actions)
                # combined_stats['rollout/Q_mean'] = mpi_mean(epoch_qs)

                # # Train statistics.
                # combined_stats['train/loss_actor'] = mpi_mean(epoch_actor_losses)
                # combined_stats['train/loss_critic'] = mpi_mean(epoch_critic_losses)
                # combined_stats['train/param_noise_distance'] = mpi_mean(epoch_adaptive_distances)



                # # Total statistics.
                # combined_stats['total/duration'] = mpi_mean(duration)
                # combined_stats['total/steps_per_second'] = mpi_mean(float(t) / float(duration))
                # combined_stats['total/episodes'] = mpi_mean(episodes)
                # combined_stats['total/epochs'] = epoch + 1
                # combined_stats['total/steps'] = t

            
                        

