#!/usr/bin/env python
from __future__ import print_function

import actionlib
from actionlib import SimpleActionClient, SimpleGoalState
import rospy
from control_msgs.msg import JointTrajectoryControllerState
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from gazebo_msgs.msg import LinkStates
from time import sleep 
import numpy as np
import math
import os

ARM_JOINTS = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
              'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

INITIAL_STATES = [0, -3.1415*(1./3), 3.1415*(2./3), 
                  -3.1415*(1./3), 3.1415*(1./2), 0]
TARGET1 = [0.4, 0.0, 1.4]
TARGET2 = [0.4, 0.3, 1.4]
DT_DELTA_ANGLE = 0.2 # while running
DT_ANGLE = 3.0 # while initializing
LOG_PATH = '/home/icsl/catkin_ws/src/ddpg_ur3/scripts/Results/'
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)
LOG1 = LOG_PATH+'log1.txt'
LOG2 = LOG_PATH+'log2.txt'
LOG3 = LOG_PATH+'log3.csv'

def deleteContent(fName):
    with open(fName, "w"):
        pass

def getSign(data):
    if data > 0:
        return 1
    elif data < 0:
        return -1
    else:
        return 0


class UR3_ARM:
    def __init__(self):
        self.jta = actionlib.SimpleActionClient('arm_controller/follow_joint_trajectory',
                                        FollowJointTrajectoryAction)
        rospy.loginfo('Waiting for joint trajectory action')
        self.jta.wait_for_server()
        rospy.loginfo('Found joint trajectory action!!')
        # global TARGET
        # TARGET = [0.4, 0.15, 1.3] 

        self.reset(0)

    def move1(self, angles):
        """ Creates an action from the trajectory and sends it to the server"""
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = ARM_JOINTS

        point = JointTrajectoryPoint()
        point.positions = angles
        point.time_from_start = rospy.Duration(DT_ANGLE)
        goal.trajectory.points.append(point)
        self.jta.send_goal_and_wait(goal)
        # self.jta.send_goal(goal)

    def move2(self, angles):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = ARM_JOINTS

        point = JointTrajectoryPoint()
        global states

        # check constraint!!!!!!!!!!!!!!!!!!!!!
        angles = self.constraint(states, angles)

        for x, y in zip(states, angles):
            point.positions.append(x + y)
        # a = states[0] + angles[0]
        # b = states[1] + angles[1]
        # c = states[2] + angles[2]
        # d = states[3] + angles[3]
        # e = states[4] + angles[4]
        # f = states[5] + angles[5]    
        # point.positions = [a, b, c, d, e, f] 
        point.time_from_start = rospy.Duration(DT_DELTA_ANGLE)
        goal.trajectory.points.append(point)
        self.jta.send_goal_and_wait(goal)

    def is_move_done(self):

        return self.jta.get_state()

    def reset(self, num_episode):
        print("Reset Environment >>>>>>>>>>>>>>>>>>>>")
        # global episode, reward_per_ep, step, done, reward
        # episode += 1
        # reward_per_ep = 0
        # step = 0
        # done = False
        # reward = 0

        self.move1(INITIAL_STATES)
        while(self.is_move_done() == 0 and rospy.is_shutdown()):
            # rospy.spinOnce()
            sleep(0.5)

        print("<<<<<<<<<<<<<<<<<<< Completed environment reset")

        global states, endPosition
        direction_x = getSign(endPosition[0] - TARGET1[0])
        direction_y = getSign(endPosition[1] - TARGET1[1])
        direction_z = getSign(endPosition[2] - TARGET1[2])
        direction_vector = (direction_x, direction_y, direction_z)
        obs = (states[0], )
        obs += direction_vector
        obs += (TARGET1[0], TARGET1[1], TARGET1[2])

        return obs 

    def step(self, action, flag):
        self.move2(action)
        while(self.is_move_done() == 0 and rospy.is_shutdown()):
            # rospy.spinOnce()
            sleep(0.5)
        global states, endPosition
        reward, done, flag = self.get_reward(endPosition, action, flag)
        
        if flag == True: 
            direction_x = getSign(endPosition[0] - TARGET1[0])
            direction_y = getSign(endPosition[1] - TARGET1[1])
            direction_z = getSign(endPosition[2] - TARGET1[2])
        else:
            direction_x = getSign(endPosition[0] - TARGET2[0])
            direction_y = getSign(endPosition[1] - TARGET2[1])
            direction_z = getSign(endPosition[2] - TARGET2[2])
            
        direction_vector = (direction_x, direction_y, direction_z)
        observation = (states[0], )
        observation += direction_vector
        
        if flag == True:
            observation += (TARGET2[0], TARGET2[1], TARGET2[2])
        else:
            observation += (TARGET1[0], TARGET1[1], TARGET1[2])

        return observation, reward, done, flag

    def get_reward(self, end_position, action, flag):
        # print(end_position, action)
        distance1 = np.sqrt((TARGET1[0]-end_position[0])**2 + (TARGET1[1]-end_position[1])**2 + (TARGET1[2]-end_position[2])**2)
        distance2 = np.sqrt((TARGET2[0]-end_position[0])**2 + (TARGET2[1]-end_position[1])**2 + (TARGET2[2]-end_position[2])**2)

        reward1 = -1*distance1 - 0.1*np.linalg.norm(action)
        reward2 = -1*distance2 - 0.1*np.linalg.norm(action)

        if flag == False:
            reward = reward1 + 2*reward2
            if distance1 < 0.1:
                flag = True
                done = False
            else:
                done = False
        else:
            reward = 2*reward2
            if distance2 < 0.1:
                done = True
                reward = 10
            else:
                done = False

        return reward, done, flag

    def constraint(self, states, action):
        a = action
        s = list(states)

        INITIAL_STATES = [0, -3.1415*(1./3), 3.1415*(2./3), 
                        -3.1415*(1./3), 3.1415*(1./2), 0]
        
        if s[0] >= 1.5:
            if a[0] >= 0:
                a[0] = 0
        if s[0] <= -1.5:
            if a[0] <= 0:
                a[0] = 0

        if s[1] >= -0.4:
            if s[2] > 2.1 and a[2] > 0:
                a[2] = 0
            elif s[2] > 2.1 and a[1] > 0:
                a[1] = 0
        if s[1] <= -1.5:
            if a[1] <= 0:
                a[1] = 0

        if s[2] >= -1.0:
            if s[1] > -0.4 and a[1] > 0:
                a[1] = 0
            elif s[1] > -0.4 and a[2] > 0:
                a[2] = 0        
        if s[2] <= 1.5:
            if a[2] < 0:
                a[2] = 0

        if s[3] >= 1.5:
            if a[3] > 0:
                a[3] = 0
        if s[3] >= -1.5:
            if a[3] < 0:
                a[3] = 0

        if s[4] >= 1.5:
            if a[3] > 0:
                a[3] = 0
        if s[3] >= -1.5:
            if a[3] < 0:
                a[3] = 0

        if s[4] >= 1.9 and s[3] >= 0.65:
            if a[3] >= 0:
                a[3] = 0
            if a[4] >= 0:
                a[4] = 0



        for i in range(6):
            if s[i] >= 3.1:
                if a[i] >= 0:
                    a[i] = 0
            if s[i] <= -3.1:
                if [i] <= 0:
                    a[i] = 0

        return a
# =============================================================================
# ====================== CALLBACK FUNC. =======================================
def arm_state_callback(data):
    global states
    states = data.actual.positions
    # print(states)

def link_state_callback(data):
    global endPosition
    endPosition = [data.pose[7].position.x, data.pose[7].position.y, 
                                            data.pose[7].position.z]
    # print(endPosition)


# =============================================================================
# ======================= HELPER FUNC. ========================================
import argparse
import time
import os
from tempfile import mkdtemp
import sys
import subprocess
import threading
import json

from mpi_fork import mpi_fork
# import logger
from misc_util import (
    set_global_seeds,
    boolean_flag,
    SimpleMonitor
)
# import training as training
from models import Actor, Critic
from memory import Memory
from noise import *

from puma560 import arm
import tensorflow as tf
from mpi4py import MPI

def parse_args():
    parser = argparse.ArgumentParser()
    
    # parser.add_argument('--env-id', type=str, default='HalfCheetah-v1')
    # boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'eval-display-on', default = False)
    boolean_flag(parser, 'layer-norm', default=True)
    # boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'display-on', default=False)
    parser.add_argument('--num-cpu', type=int, default=1)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=500)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default= 'ou_0.3') # 'adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    # parser.add_argument('--noise-type', type=str, default= 'adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none

    parser.add_argument('--logdir', type=str, default='./log')
    # boolean_flag(parser, 'gym-monitor', default=False)
    boolean_flag(parser, 'evaluation', default=False)
    boolean_flag(parser, 'bind-to-core', default=False)

    return vars(parser.parse_args())


def run(#env_id, 
        seed, noise_type, num_cpu, layer_norm, logdir, #gym_monitor, 
            evaluation, bind_to_core, **kwargs):
    kwargs['logdir'] = logdir
    whoami = mpi_fork(num_cpu, bind_to_core=bind_to_core)
    if whoami == 'parent':
        sys.exit(0)

    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        # Write to temp directory for all non-master workers.
        actual_dir = None
        # Logger.CURRENT.close()
        # Logger.CURRENT = Logger(dir=mkdtemp(), output_formats=[])
        # logger.set_level(logger.DISABLED)
    
    # Create envs.
    if rank == 0:
        # env = gym.make(env_id)
        env = UR3_ARM()
        # if gym_monitor and logdir:
            # env = gym.wrappers.Monitor(env, os.path.join(logdir, 'gym_train'), force=True)
        # env = SimpleMonitor(env)

        if evaluation:
            # eval_env = gym.make(env_id)
            eval_env = UR3_ARM()
            # if gym_monitor and logdir:
                # eval_env = gym.wrappers.Monitor(eval_env, os.path.join(logdir, 'gym_eval'), force=True)
            # eval_env = SimpleMonitor(eval_env)
        else:
            eval_env = None
    else:
        # env = gym.make(env_id)
        env = UR3_ARM()
        if evaluation:
            # eval_env = gym.make(env_id)
            eval_env = UR3_ARM()
        else:
            eval_env = None

    # Parse noise_type
    action_noise = None
    param_noise = None
    # nb_actions = env.action_space.shape[-1]
    nb_actions = 6
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    # memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    memory = Memory(limit=int(1e6), action_shape = (6,), observation_shape = (7,))
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    # logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    # env.seed(seed)
    # if eval_env is not None:
        # eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()

    ###########################################################################    
    train(env=env, eval_env=eval_env, param_noise=param_noise,
        action_noise=action_noise, actor=actor, critic=critic, memory=memory, **kwargs)
       
    # env.close()
    # if eval_env is not None:
        # eval_env.close()
    # Logger.CURRENT.close()
    # if rank == 0:
        # logger.info('total runtime: {}s'.format(time.time() - start_time))


# =============================================================================
# ======================= TRAINING FUNC. ======================================

import os
import time
from collections import deque
import pickle

from ddpg import DDPG
from util import mpi_mean, mpi_std, mpi_max, mpi_sum
import tf_util as U

# import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI
import matplotlib.pyplot as plt



def train(env, nb_epochs, nb_epoch_cycles, eval_display_on, #render_eval, 
    reward_scale, display_on, #render,
    param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise, logdir,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
    tau=0.01, eval_env=None, param_noise_adaption_interval=50):
    
    # LOG_PATH = '/home/icsl/catkin_ws/src/ddpg_ur3/scripts/Results/'
    # if not os.path.exists(LOG_PATH):
    #     os.makedirs(LOG_PATH)
        
    global states, endPosition

    rank = MPI.COMM_WORLD.Get_rank()

    # assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    # max_action = env.action_space.high
    max_action = np.ones(6)*0.1
    # logger.info('scaling actions by {} before executing in env'.format(max_action))
    # agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
    #     gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
    #     batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
    #     actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
    #     reward_scale=reward_scale)
    agent = DDPG(actor, critic, memory, (7, ), (6, ),
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
    # logger.info('Using agent with the following configuration:')
    # logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver()
    else:
        saver = None
    
    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)
        sess.graph.finalize()

        agent.reset()
        global TARGET
        # TARGET = [0.4, 0.15, 1.3] 
        # m = 0
        obs = env.reset(0)
       
        # print('initial obs', states)

        if eval_env is not None:
            eval_obs = eval_env.reset()

        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0

        epoch = 0
        start_time = time.time()

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0
        for epoch in range(nb_epochs):
            # if epoch > 10:
                # eval_display_on = True
            for cycle in range(nb_epoch_cycles):
                # Perform rollouts.
                # if m == 0:
                #     TARGET = [0.4, 0.15, 1.3]
                #     m += 1
                # else:
                #     TARGET = [0.4 + 0.2*np.random.rand() - 0.1 , 
                #               0.15 + 0.3*np.random.rand() - 0.15, 
                #               1.3 + 0.2*np.random.rand() - 0.1]   
                flag = False
                eval_flag = False

                for t_rollout in range(nb_rollout_steps):
                    # Predict next action.
                    action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    # assert action.shape == env.action_space.shape
                    assert action.shape == (6, )

                    # Execute next action.
                    # if rank == 0 and render:
                        # env.render()
                    # assert max_action.shape == action.shape
                    assert max_action.shape == (6, )
                    # print('before A', states)
                    new_obs, r, done, flag = env.step(max_action * action, flag)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    # print('After A', states)
                    # a = []
                    # for x, y in zip(obs, action):
                    #     a.append(x + y)
                    # print('sum', a)
                    global TARGET

                    with open(LOG_PATH+'log1.txt', 'a') as out_file:
                        out_string = "-----------------------" + str(t_rollout) + "----------------------------\n"
                        out_string = out_string + str(obs) + "\n" + str(action) + "\n" + str(new_obs) + "\n\n"
                        out_file.write(out_string) 
                    
                    # global endPosition
                    with open(LOG_PATH+'log2.txt', 'a') as out_file:
                        out_string = str(cycle) + "(" + str(nb_epoch_cycles) + ") \t" + str(t_rollout) + "(" + str(nb_rollout_steps) + ")" + str(episode_reward) + ' \t' 
                        out_string = out_string + str(q) + '      \t' + str(TARGET) + '\t' + str(endPosition) + '\t' + str(done)
                        out_string += "\n"
                        out_file.write(out_string)

                    t += 1
                    # if rank == 0 and render:
                        # env.render()
                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    agent.store_transition(obs, action, r, new_obs, done)
                    obs = new_obs

                    print("\rRoll {}/{} @ Cycle{}/{} @ Episode {}/{} ||| R: {} ||| {} ||{}".format(t_rollout + 1, 
                        nb_rollout_steps, cycle + 1, nb_epoch_cycles, epoch, nb_epochs, episode_reward, endPosition, flag), end="")                  
                    
                    if done:
                        print('\n0000000000000000000000000000000000')

                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0.
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1

                        agent.reset()
                        obs = env.reset(cycle)
                        break
                print('')

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


                ########################################################
                # epoch_episode_rewards.append(episode_reward)
                # episode_rewards_history.append(episode_reward)
                # epoch_episode_steps.append(episode_step)
                agent.reset()
                obs = env.reset(cycle)
                episode_reward = 0
                episode_step = 0
                ########################################################


                # Evaluate.
                eval_episode_rewards = []
                eval_qs = []
                if eval_env is not None:
                    eval_episode_reward = 0.
                    for t_rollout in range(nb_eval_steps):
                        eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                        new_eval_obs = eval_obs + eval_action*max_action
                        eval_r, eval_endP, eval_done = eval_env.step(new_eval_obs)  

                        # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                        # if render_eval:
                        #     eval_env.render()
                        eval_episode_reward += eval_r
                        eval_obs = new_eval_obs

                        eval_qs.append(eval_q)
                        if eval_done:
                            eval_endP, eval_obs = eval_env.reset(ax, eval_display_on)
                            eval_episode_rewards.append(eval_episode_reward)
                            eval_episode_rewards_history.append(eval_episode_reward)
                            eval_episode_reward = 0.
                            break
                            
                    ####################################################################
                    # eval_endP, eval_obs = eval_env.reset(ax, eval_display_on)
                    # eval_episode_rewards.append(eval_episode_reward)
                    # eval_episode_rewards_history.append(eval_episode_reward)
                    # eval_episode_reward = 0.    
                    ###################################################################


            # Log stats.
            epoch_train_duration = time.time() - epoch_start_time
            duration = time.time() - start_time
            stats = agent.get_stats()
            combined_stats = {}
            for key in sorted(stats.keys()):
                combined_stats[key] = mpi_mean(stats[key])


        
            with open(LOG_PATH+'log3.csv', 'w') as out_file:
                for m in range(len(epoch_episode_rewards)):
                    # out_string = ""
                    out_string =  str(epoch_episode_steps[m]) + '\t' + str(epoch_episode_rewards[m]) + '\t' + str(epoch_qs[m]) 
                    out_string += "\n"
                    out_file.write(out_string) 

            fig = plt.figure(2)
            plt.plot(range(len(epoch_episode_rewards)), epoch_episode_rewards)
            plt.ylabel('reward')
            plt.xlabel('episode')
            plt.savefig(LOG_PATH + 'reward.png')
            print('\n saved reward figure')

            # Rollout statistics.
            combined_stats['rollout/return'] = mpi_mean(epoch_episode_rewards)
            combined_stats['rollout/return_history'] = mpi_mean(np.mean(episode_rewards_history))
            combined_stats['rollout/episode_steps'] = mpi_mean(epoch_episode_steps)
            combined_stats['rollout/episodes'] = mpi_sum(epoch_episodes)
            combined_stats['rollout/actions_mean'] = mpi_mean(epoch_actions)
            combined_stats['rollout/actions_std'] = mpi_std(epoch_actions)
            combined_stats['rollout/Q_mean'] = mpi_mean(epoch_qs)
    
            # Train statistics.
            combined_stats['train/loss_actor'] = mpi_mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = mpi_mean(epoch_critic_losses)
            combined_stats['train/param_noise_distance'] = mpi_mean(epoch_adaptive_distances)

            # Evaluation statistics.
            if eval_env is not None:
                combined_stats['eval/return'] = mpi_mean(eval_episode_rewards)
                combined_stats['eval/return_history'] = mpi_mean(np.mean(eval_episode_rewards_history))
                combined_stats['eval/Q'] = mpi_mean(eval_qs)
                combined_stats['eval/episodes'] = mpi_mean(len(eval_episode_rewards))

            # Total statistics.
            combined_stats['total/duration'] = mpi_mean(duration)
            combined_stats['total/steps_per_second'] = mpi_mean(float(t) / float(duration))
            combined_stats['total/episodes'] = mpi_mean(episodes)
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t
            
            # for key in sorted(combined_stats.keys()):
                # logger.record_tabular(key, combined_stats[key])
            # logger.dump_tabular()
            # logger.info('')

            # if rank == 0 and logdir:
            #     if hasattr(env, 'get_state'):
            #         with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
            #             pickle.dump(env.get_state(), f)
            #     if eval_env and hasattr(eval_env, 'get_state'):
            #         with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
            #             pickle.dump(eval_env.get_state(), f)


# =============================================================================
# ======================= MAIN FUNC. ==========================================
def main():

    deleteContent(LOG_PATH+'log1.txt')
    deleteContent(LOG_PATH+'log2.txt')
    deleteContent(LOG_PATH+'log3.csv')

    args = parse_args()

    # # Figure out what logdir to use.
    # if args['logdir'] is None:
    #     args['logdir'] = os.getenv('OPENAI_LOGDIR')
    
    # # Print and save arguments.
    # logger.info('Arguments:')
    # for key in sorted(args.keys()):
    #     logger.info('{}: {}'.format(key, args[key]))
    # logger.info('')
    # if args['logdir']:
    #     with open(os.path.join(args['logdir'], 'args.json'), 'w') as f:
    #         json.dump(args, f)

    # # Run actual script.
    run(**args)


    # arm = UR3_ARM(TARGET)
    # global states
    # for i in range(4):
    #     print(i, states)
    #     arm.step([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5], 0.3)
    #     while(arm.is_move_done() == 0 and rospy.is_shutdown()):
    #         sleep(0.05)
    #     print(i, states, 'complete---')
    print("=== COMPLETE !!! ===")

if __name__ == '__main__':
    # global parameters
    states = []
    action = []
    endPosition = [0, 0, 0]
    TARGET = []

    # MAXEPISODE = 500
    # MAXSTEP = 100
    # ACTION_BOUND = 0.1
    # episode = 0
    # step = 0
    # reward_per_ep = 0
    # reward = 0

    done = False
    isGetAction = False

    # node declaration
    rospy.init_node('joint_position')

    # callback func. declaration
    rospy.Subscriber('arm_controller/state', JointTrajectoryControllerState, arm_state_callback)
    rospy.Subscriber('gazebo/link_states', LinkStates, link_state_callback)
        
    # main func. starts    
    main()
    
