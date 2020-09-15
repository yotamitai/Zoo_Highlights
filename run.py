import argparse
import os

from highlights import create_highlights, get_multiple_highlights
from utils import make_dirs, find_features_layer
from rl_baselines_zoo.utils import ALGOS
from get_traces import load_agent
from environments import Evnironments
from agent_comparisons import compare_agents

import logging

if __name__ == '__main__':
    # TODO parser args:
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', type=str, default='CartPole-v1')
    parser.add_argument('-f', '--folder', help='Log folder', type=str, default='trained_agents')
    parser.add_argument('--algo', help='RL Algorithm', default='ppo2', type=str, required=False,
                        choices=list(ALGOS.keys()))
    parser.add_argument('-n', '--n-timesteps', help='number of timesteps', default=1000, type=int)
    parser.add_argument('--n-envs', help='number of environments', default=1, type=int)
    parser.add_argument('--exp-id', help='Experiment ID (default: -1, no exp folder, 0: latest)', default=-1, type=int)
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1, type=int)
    parser.add_argument('--no-render', action='store_true', default=False,
                        help='Do not render the environment (useful for tests)')
    parser.add_argument('--deterministic', action='store_true', default=False, help='Use deterministic actions')
    parser.add_argument('--stochastic', action='store_true', default=False,
                        help='Use stochastic actions (for DDPG/DQN/SAC)')
    parser.add_argument('--norm-reward', action='store_true', default=False,
                        help='Normalize reward if applicable (trained with VecNormalize)')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--reward-log', help='Where to log reward', default='', type=str)
    parser.add_argument('--gym-packages', type=str, nargs='+', default=[],
                        help='Additional external Gym environment package modules to import (e.g. gym_minigrid)')

    args = parser.parse_args()

    """Model Parameters"""
    args.env = 'MsPacmanNoFrameskip-v4'  # SeaquestNoFrameskip-v4, MsPacmanNoFrameskip-v4
    args.algo = 'acktr'  # 'a2c', 'ppo2' , 'acktr', 'dqn'
    args.stats_path = None
    args.log_dir = None
    args.hyperparams = {}
    args.deterministic = False
    args.stochastic = False
    args.max_trace_timesteps = 5000
    args.verbose = 1
    args.n_envs = 1
    args.is_atari = True
    args.no_render = True

    """Highlights Parameters"""
    args.summary_traj_budget = 10
    args.context_length = 2 * args.summary_traj_budget  # must be even number
    assert args.context_length % 2 == 0, "The context range of a state must be an even number"
    args.minimum_gap = 10
    args.trajectory_importance = "max_min"  # avg , max_minus_avg, avg_delta, max_min, single_state
    args.state_importance = "second"  # worst, second
    args.similarity_limit = 0  # 0 , int(args.context_length / 3)

    "Agent Comparison Parameters"
    args.important_state_percentage = 0.1

    """Experiment parameters"""
    args.load_traces = False
    # args.load_trajectories = False
    args.random_noop_init = True
    args.random_noop_range = 40

    args.built_in_noop = Evnironments[args.env]["built_in_noop"]
    # needed for creating a list of random noop starts. list must be longer than num of traces
    args.crop_top = Evnironments[args.env]["crop_top"]
    args.crop_bottom = Evnironments[args.env]["crop_bottom"]
    args.single_life_trace = True  # trace = single life or full game
    args.num_traces = 20
    assert args.num_traces < args.random_noop_range

    """Directory Parameters"""
    args.agents_dir = "rl_baselines_zoo/trained_agents"
    args.env_dir = os.path.join("output", args.env)
    args.stt_dir = os.path.join(args.env_dir, "states_traces_trajectories")
    args.video_dir = os.path.join(args.env_dir, "videos")
    make_dirs(args.env_dir)
    make_dirs(args.stt_dir)
    make_dirs(args.video_dir)
    args.file_name = str(args.num_traces) + ".pkl"
    args.traces_file = os.path.join(args.stt_dir, args.algo, "Traces:" + args.file_name)
    args.state_file = os.path.join(args.stt_dir, args.algo, "States:" + args.file_name)
    args.trajectories_file = os.path.join(args.stt_dir, args.algo, "Trajectories:" + args.file_name)

    """Bad Result Experiments"""
    args.map_action_reduction = False  # results are much worse hen True
    args.rand_step = 0  # not sure if we want this

    """RUN"""
    create_highlights(args)

    """MULTIPLE RUNS"""
    # get_multiple_highlights(args)

    """LOADING AN AGENT"""
    # environment, agent = load_agent(args)
    """features"""
    # features_layer = find_features_layer(agent)

    """LOADING MULTIPLE AGENTS"""
    compare_agents(args)

    print()
