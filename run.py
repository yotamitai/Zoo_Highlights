import argparse
import os
import pandas as pd
from rl_baselines_zoo.utils import ALGOS
from highlights_state_selection import compute_states_importance, highlights
from get_traces import load_traces, get_traces, load_agent
from get_trajectories import get_trajectory_images, create_video, trajectories_by_importance, states_to_trajectories
from tools import make_dirs
from environments import Evnironments


def experiment(kargs):
    # environments = ['SeaquestNoFrameskip-v4', 'MsPacmanNoFrameskip-v4']
    algos = ['a2c', 'ppo2', 'acktr', 'dqn']
    state_importance = ["second", "worst"]
    trajectory_importance = ["avg", "max_minus_avg", "avg_delta", "max_min", "single_state"]
    kargs.verbose = False

    print("Starting Experiments:")
    # for env in environments:
    #     print(f"\tEnvironment: {env}")
    for algo in algos:
        print(f"\t\tAlgorithm: {algo}")
        kargs.algo = algo
        for s_i in state_importance:
            args.load_traces = False  # need to save new trajectories
            print(f"\t\t\tState Importance: {s_i}")
            kargs.state_importance = s_i
            for t_i in trajectory_importance:
                print(f"\t\t\t\tTrajectory Importance: {t_i}")
                kargs.trajectory_importance = t_i
                main(kargs)
                print(f"\t\t\t\t....Completed")
                args.load_traces = True  # use saved trajectories

    print("Experiments Completed")


def main(params):
    """load pre-trained agent from RL-Zoo and retrieve execution traces and states"""

    """RL-Zoo"""
    if params.load_traces:
        traces, states = load_traces(params)
    else:
        traces, states = get_traces(params)

    """HIGHLIGHTS"""
    data = {
        'state': list(states.keys()),
        'q_values': [x.observed_actions for x in states.values()]
    }
    q_values_df = pd.DataFrame(data)

    """importance by state"""
    q_values_df = compute_states_importance(q_values_df, compare_to=params.state_importance)
    highlights_df = q_values_df
    state_importance_dict = dict(zip(highlights_df["state"], highlights_df["importance"]))

    """get highlights"""
    if params.trajectory_importance == "single_state":
        """highlights importance by single state importance"""
        summary_states = highlights(highlights_df, traces, params.summary_traj_budget, params.context_length,
                                    params.minimum_gap)
        summary_trajectories = states_to_trajectories(summary_states, state_importance_dict)
    else:
        """highlights importance by trajectory"""
        summary_trajectories = trajectories_by_importance(params.trajectory_importance, traces,
                                                          params.context_length, params.load_traces,
                                                          params.trajectories_file, state_importance_dict,
                                                          params.similarity_limit, params.summary_traj_budget)
    if params.verbose: print('HIGHLIGHTS obtained')

    """make video"""
    dir_name = os.path.join(params.video_dir, params.algo, params.state_importance +
                            "_state_importance", params.trajectory_importance)
    get_trajectory_images(summary_trajectories, states, dir_name)
    create_video(dir_name)
    if params.verbose: print("HIGHLIGHTS Video Obtained")
    return


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
    agents_dir = "rl_baselines_zoo/trained_agents"
    args.model_path = os.path.join(agents_dir, args.algo, args.env + ".pkl")
    args.stats_path = None
    args.log_dir = None
    args.hyperparams = {}
    args.deterministic = False
    args.stochastic = False
    args.max_trace_timesteps = 5000
    args.verbose = 1
    args.n_envs = 1
    args.is_atari = True
    args.no_render = False

    """Highlights Parameters"""
    args.summary_traj_budget = 10
    args.context_length = 20  # must be even number
    assert args.context_length % 2 == 0, "The context range of a state must be an even number"
    args.minimum_gap = 10
    args.trajectory_importance = "avg"  # avg , max_minus_avg, avg_delta, max_min, single_state
    args.state_importance = "second"  # worst, second
    args.similarity_limit = int(args.context_length / 4)

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
    args.env_dir = os.path.join("output", args.env)
    args.stt_dir = os.path.join(args.env_dir, "states_traces_trajectories")
    args.video_dir = os.path.join(args.env_dir, "videos")
    make_dirs(args.env_dir)
    make_dirs(args.stt_dir)
    make_dirs(args.video_dir)
    file_name = str(args.num_traces) + '_' + args.algo + ".pkl"
    args.traces_file = os.path.join(args.stt_dir, "Traces:" + file_name)
    args.state_file = os.path.join(args.stt_dir, "States:" + file_name)
    args.trajectories_file = os.path.join(args.stt_dir, "Trajectories:" + file_name)

    """Bad Result Experiments"""
    args.map_action_reduction = False  # results are much worse hen True
    args.rand_step = 0  # not sure if we want this

    """RUN"""
    main(args)

    """EXPERIMENT"""
    # experiment(args)

    """LOADING AN AGENT"""
    # environment, agent = load_agent(args)
    # """features"""
    # features = get_features(environment, agent)

    print()
