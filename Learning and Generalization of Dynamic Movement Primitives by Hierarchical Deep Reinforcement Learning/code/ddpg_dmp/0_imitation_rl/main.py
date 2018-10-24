import argparse
import time
import os
from tempfile import mkdtemp
import sys
import subprocess
import threading
import json
import gym

from mpi_fork import mpi_fork
from misc_util import (
    set_global_seeds,
    boolean_flag,
    SimpleMonitor
)
import training as training
from models import Actor, Critic
from memory import Memory
from noise import *

import tensorflow as tf
from mpi4py import MPI

from dmp.dmp_discrete import DMP_discrete

def run(seed, noise_type, layer_norm, **kwargs):
    # dmp ----------------------------------------------------------------------
    runTime = 1.0
    dt = 0.01
    n_dmps = 1

    # path = 0.5*np.arange(0, runTime, dt)
    path = np.sin(5*np.arange(0, runTime, dt))
    dmp = DMP_discrete(n_dmps=n_dmps, runTime=runTime, dt=dt)

    imitation_flag = False

    # ddpg ---------------------------------------------------------------------
    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = 1
    nb_observations = 1
    action_shape = (nb_actions, )
    observation_shape = (nb_observations, )

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
    memory = Memory(limit=int(1e6), action_shape=action_shape, observation_shape=observation_shape)
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    seed = seed + 1000000
    tf.reset_default_graph()
    set_global_seeds(seed)

    # Disable logging for rank != 0 to avoid noise.
    start_time = time.time()
    training.train(dmp=dmp, path=path, imitation_flag=imitation_flag, observation_shape=observation_shape, action_shape=action_shape, param_noise=param_noise,
        action_noise=action_noise, actor=actor, critic=critic, memory=memory, **kwargs)
    # -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--imitation-lr', type=float, default=1e-3)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=500)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
    # parser.add_argument('--noise-type', type=str, default= 'ou_0.3') # 'adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--noise-type', type=str, default= 'ou_0.3')  # choices are adaptive-param_xx, ou_xx, normal_xx, none

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()

    # Run actual script.
    run(**args)
