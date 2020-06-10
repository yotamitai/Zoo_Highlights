import os
import pandas as pd
from get_traces import load_traces, get_traces
from highlights_state_selection import compute_states_importance, highlights
from get_trajectories import get_trajectory_images, create_video, trajectories_by_importance, states_to_trajectories


def create_highlights(args):
    """
    load pre-trained agent from RL-Zoo.
    retrieve execution traces and states.
    Obtain trajectories and create highlights video.
    """

    """RL-Zoo"""
    if args.load_traces:
        traces, states = load_traces(args)
    else:
        traces, states = get_traces(args)

    """HIGHLIGHTS"""
    data = {
        'state': list(states.keys()),
        'q_values': [x.observed_actions for x in states.values()]
    }
    q_values_df = pd.DataFrame(data)

    """importance by state"""
    q_values_df = compute_states_importance(q_values_df, compare_to=args.state_importance)
    highlights_df = q_values_df
    state_importance_dict = dict(zip(highlights_df["state"], highlights_df["importance"]))

    """get highlights"""
    if args.trajectory_importance == "single_state":
        """highlights importance by single state importance"""
        summary_states = highlights(highlights_df, traces, args.summary_traj_budget, args.context_length,
                                    args.minimum_gap)
        summary_trajectories = states_to_trajectories(summary_states, state_importance_dict)
    else:
        """highlights importance by trajectory"""
        summary_trajectories = trajectories_by_importance(args.trajectory_importance, traces,
                                                          args.context_length, args.load_traces,
                                                          args.trajectories_file, state_importance_dict,
                                                          args.similarity_limit, args.summary_traj_budget)
    if args.verbose: print('HIGHLIGHTS obtained')

    """make video"""
    dir_name = os.path.join(args.video_dir, args.algo, args.state_importance +
                            "_state_importance", args.trajectory_importance)
    get_trajectory_images(summary_trajectories, states, dir_name)
    create_video(dir_name)
    if args.verbose: print("HIGHLIGHTS Video Obtained")
    return


def get_multiple_highlights(args):
    # environments = ['SeaquestNoFrameskip-v4', 'MsPacmanNoFrameskip-v4']
    algos = ['a2c', 'ppo2', 'acktr', 'dqn']
    state_importance = ["second", "worst"]
    trajectory_importance = ["avg", "max_minus_avg", "avg_delta", "max_min", "single_state"]
    args.verbose = False

    print("Starting Experiments:")
    # for env in environments:
    #     print(f"\tEnvironment: {env}")
    for algo in algos:
        print(f"\t\tAlgorithm: {algo}")
        args.algo = algo
        args.traces_file = os.path.join(args.stt_dir, args.algo, "Traces:" + args.file_name)
        args.state_file = os.path.join(args.stt_dir, args.algo, "States:" + args.file_name)
        args.trajectories_file = os.path.join(args.stt_dir, args.algo, "Trajectories:" + args.file_name)
        for s_i in state_importance:
            args.load_traces = False  # need to save new trajectories
            print(f"\t\t\tState Importance: {s_i}")
            args.state_importance = s_i
            for t_i in trajectory_importance:
                print(f"\t\t\t\tTrajectory Importance: {t_i}")
                args.trajectory_importance = t_i
                create_highlights(args)
                print(f"\t\t\t\t....Completed")
                args.load_traces = True  # use saved trajectories

    print("Experiments Completed")
